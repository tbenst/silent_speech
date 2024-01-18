##
# 2023-07-25_dtw_speech_silent_emg.py : best sEMG results
# 2023-08-24_brain_to_text_comp_split.py : most recent brain-to-text results, uses MONA name
2
##
# %load_ext autoreload
# %autoreload 2
##
import os, subprocess

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "backend:cudaMallocAsync" # no OOM
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
hostname = subprocess.run("hostname", capture_output=True)
ON_SHERLOCK = hostname.stdout[:2] == b"sh"

import pytorch_lightning as pl, pickle
import sys, warnings
import numpy as np
import logging
import torchmetrics
import random, typer
from tqdm.auto import tqdm
from typing import List
from dataclasses import dataclass
import torch
from torch import nn
from torch.utils.data import DistributedSampler
import torch.nn.functional as F

# horrible hack to get around this repo not being a proper python package
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(SCRIPT_DIR)

from read_emg import (
    EMGDataset,
    PreprocessedEMGDataset,
    PreprocessedSizeAwareSampler,
    EMGDataModule,
    ensure_folder_on_scratch,
)
from architecture import Model, S4Model, H3Model, ResBlock, MONAConfig, MONA
from data_utils import combine_fixed_length, decollate_tensor
from transformer import TransformerEncoderLayer
from pytorch_lightning.loggers import NeptuneLogger

# import neptune, shutil
import neptune.new as neptune, shutil
import typer
from datetime import datetime
from pytorch_lightning.callbacks import ModelCheckpoint, GradientAccumulationScheduler
from pytorch_lightning.profilers import (
    SimpleProfiler,
    AdvancedProfiler,
    PyTorchProfiler,
    PassThroughProfiler,
)
from pytorch_lightning.strategies import DDPStrategy
from data_utils import TextTransform, in_notebook
from typing import List
from collections import defaultdict
from enum import Enum
from magneto.preprocessing import ensure_data_on_scratch
from dataloaders import (
    LibrispeechDataset,
    EMGAndSpeechModule,
    DistributedStratifiedBatchSampler,
    StratifiedBatchSampler,
    cache_dataset,
    split_batch_into_emg_neural_audio,
    DistributedSizeAwareStratifiedBatchSampler,
    SizeAwareStratifiedBatchSampler,
    collate_gaddy_or_speech,
    collate_gaddy_speech_or_neural,
    DistributedSizeAwareSampler,
    T12DataModule,
    T12Dataset,
    NeuralDataset,
    T12CompDataModule,
)
from functools import partial
from contrastive import (
    cross_contrastive_loss,
    var_length_cross_contrastive_loss,
    nobatch_cross_contrastive_loss,
    supervised_contrastive_loss,
)
import glob, scipy
from helpers import load_npz_to_memory

DEBUG = False
# DEBUG = True
RESUME = False
# RESUME = True

torch.set_float32_matmul_precision("high")
# torch.set_float32_matmul_precision("medium" | "high")

if RESUME:
    # TODO: make an auto-resume feature...? or at least find ckpt_path from run_id
    # to think about: can we do this automatically on gaia/sherlock if OOM..? (maybe we don't care / can do manually)
    # INFO: when resuming logging to Neptune, we might repeat some steps,
    # e.g. if epoch 29 was lowest WER, but we resume at epoch 31, we will
    # log epoch 30 & 31 twice. mainly an issue for publication plots
    # ckpt_path = '/scratch/2023-07-10T12:20:43.920850_gaddy/SpeechOrEMGToText-epoch=29-val/wer=0.469.ckpt'
    ckpt_path = "/scratch/2023-08-03T21:30:03.418151_gaddy/SpeechOrEMGToText-epoch=15-val/wer=0.547.ckpt"
    run_id = "GAD-493"

per_index_cache = True  # read each index from disk separately
# per_index_cache = False # read entire dataset from disk

isotime = datetime.now().isoformat()

if DEBUG:
    NUM_GPUS = 1
    limit_train_batches = 2
    limit_val_batches = 2  # will not run on_validation_epoch_end
    # NUM_GPUS = 2
    # limit_train_batches = None
    # limit_val_batches = None
    log_neptune = False
    n_epochs = 2
    # precision = "32"
    precision = "16-mixed"
    num_sanity_val_steps = 2
    grad_accum = 1
    logger_level = logging.DEBUG
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

else:
    NUM_GPUS = 1
    grad_accum = 2 # might need if run on 1 GPU
    # grad_accum = 1
    precision = "16-mixed"
    limit_train_batches = None
    limit_val_batches = None
    log_neptune = True
    # log_neptune = False
    n_epochs = 200
    num_sanity_val_steps = 0  # may prevent crashing of distributed training
    logger_level = logging.WARNING


assert (
    os.environ["NEPTUNE_ALLOW_SELF_SIGNED_CERTIFICATE"] == "TRUE"
), "run this in shell: export NEPTUNE_ALLOW_SELF_SIGNED_CERTIFICATE='TRUE'"

# load our data file paths and metadata:

if per_index_cache:
    cache_suffix = "_per_index"
else:
    cache_suffix = ""
if ON_SHERLOCK:
    sessions_dir = "/oak/stanford/projects/babelfish/magneto/"
    # TODO: bechmark SCRATCH vs LOCAL_SCRATCH ...?
    scratch_directory = os.environ["SCRATCH"]
    # scratch_directory = os.environ["LOCAL_SCRATCH"]
    gaddy_dir = "/oak/stanford/projects/babelfish/magneto/GaddyPaper/"
    scratch_lengths_pkl = os.path.join(
        scratch_directory, "2023-07-25_emg_speech_dset_lengths.pkl"
    )
    tmp_lengths_pkl = os.path.join("/tmp", "2023-07-25_emg_speech_dset_lengths.pkl")
    if os.path.exists(scratch_lengths_pkl) and not os.path.exists(tmp_lengths_pkl):
        shutil.copy(scratch_lengths_pkl, tmp_lengths_pkl)
    t12_npz_path = os.path.join(scratch_directory, "2023-08-21_T12_dataset.npz")
    T12_dir = os.path.join(scratch_directory, "T12_data_v4")
    if len(os.sched_getaffinity(0)) > 16:
        print(
            "WARNING: if you are running more than one script, you may want to use `taskset -c 0-16` or similar"
        )
else:
    # on my local machine
    sessions_dir = "/data/magneto/"
    scratch_directory = "/scratch"
    gaddy_dir = "/scratch/GaddyPaper/"
    # t12_npz_path = "/data/data/T12_data_v4/synthetic_audio/2023-08-21_T12_dataset_per_sentence_z-score.npz"
    t12_npz_path = "/data/data/T12_data_v4/synthetic_audio/2023-08-22_T12_dataset_gaussian-smoothing.npz"
    T12_dir = "/data/data/T12_data_v4/"

print(f"CPU affinity: {os.sched_getaffinity(0)}")

data_dir = os.path.join(gaddy_dir, "processed_data/")
lm_directory = os.path.join(gaddy_dir, "pretrained_models/librispeech_lm/")
normalizers_file = os.path.join(SCRIPT_DIR, "normalizers.pkl")

if ON_SHERLOCK:
    lm_directory = ensure_folder_on_scratch(lm_directory, scratch_directory)

gpu_ram = torch.cuda.get_device_properties(0).total_memory / 1024**3
assert gpu_ram > 70, "needs A100 80GB"
base_bz = 16  # decent compromise between class balance.
# base_bz = 48
val_bz = 16
# max_len = 48000 # from best perf with 4 x V100
# max_len = 128000 # OOM on A100 80GB
# max_len = 64000
max_len = 96000

##

app = typer.Typer()

togglePhones = False
learning_rate = 3e-4
seqlen = 600
white_noise_sd = 0
constant_offset_sd = 0
use_dtw = True
use_crossCon = True
use_supCon = True


@app.command()
def update_configs(
    constant_offset_sd_cli: float = typer.Option(0, "--constant-offset-sd"),
    white_noise_sd_cli: float = typer.Option(0, "--white-noise-sd"),
    learning_rate_cli: float = typer.Option(3e-4, "--learning-rate"),
    debug_cli: bool = typer.Option(False, "--debug/--no-debug"),
    phonemes_cli: bool = typer.Option(False, "--phonemes/--no-phonemes"),
    resume_cli: bool = typer.Option(RESUME, "--resume/--no-resume"),
    use_dtw_cli: bool = typer.Option(use_dtw, "--dtw/--no-dtw"),
    use_crossCon_cli: bool = typer.Option(use_crossCon, "--crossCon/--no-crossCon"),
    use_supCon_cli: bool = typer.Option(use_supCon, "--supCon/--no-supCon"),
    grad_accum_cli: int = typer.Option(grad_accum, "--grad-accum"),
    precision_cli: str = typer.Option(precision, "--precision"),
    logger_level_cli: str = typer.Option("WARNING", "--logger-level"),
    base_bz_cli: int = typer.Option(base_bz, "--base-bz"),
    val_bz_cli: int = typer.Option(val_bz, "--val-bz"),
    max_len_cli: int = typer.Option(max_len, "--max-len"),
    seqlen_cli: int = typer.Option(seqlen, "--seqlen"),
    # devices_cli: str = typer.Option(devices, "--devices"),
):
    """Update configurations with command-line values."""
    global constant_offset_sd, white_noise_sd, DEBUG, RESUME, grad_accum
    global precision, logger_level, base_bz, val_bz, max_len, seqlen
    global learning_rate, devices, togglePhones, use_dtw, use_crossCon, use_supCon

    # devices = devices_cli
    # try:
    #     devices = int(devices) # eg "2" -> 2
    # except:
    #     pass
    use_dtw = use_dtw_cli
    use_crossCon = use_crossCon_cli
    use_supCon = use_supCon_cli
    togglePhones = phonemes_cli
    learning_rate = learning_rate_cli
    constant_offset_sd = constant_offset_sd_cli
    white_noise_sd = white_noise_sd_cli
    DEBUG = debug_cli
    RESUME = resume_cli
    grad_accum = grad_accum_cli
    precision = precision_cli
    logger_level = getattr(logging, logger_level_cli.upper())
    base_bz = base_bz_cli
    val_bz = val_bz_cli
    max_len = max_len_cli
    seqlen = seqlen_cli
    print("Updated configurations using command-line arguments.")


if __name__ == "__main__" and not in_notebook():
    try:
        app()
    except SystemExit as e:
        pass

# needed for using CachedDataset
emg_datamodule = EMGDataModule(
    data_dir,
    togglePhones,
    normalizers_file,
    max_len=max_len,
    collate_fn=collate_gaddy_or_speech,
    pin_memory=(not DEBUG),
    batch_size=val_bz,
)
emg_train = emg_datamodule.train

mfcc_norm, emg_norm = pickle.load(open(normalizers_file, "rb"))

if NUM_GPUS > 1:
    strategy = DDPStrategy(gradient_as_bucket_view=True, find_unused_parameters=True)
elif NUM_GPUS == 1:
    strategy = "auto"
else:
    strategy = "auto"

devices = NUM_GPUS

if ON_SHERLOCK:
    # TODO: should we just use the scratch directory over LOCAL_SCRATCH?
    output_directory = os.path.join(os.environ["SCRATCH"], f"{isotime}_gaddy")
else:
    output_directory = os.path.join(scratch_directory, f"{isotime}_gaddy")

logging.basicConfig(
    handlers=[logging.StreamHandler()],
    level=logger_level,
    format="%(message)s",
    force=True,
)

logging.debug("DEBUG mode")
if not log_neptune:
    logging.warning("not logging to neptune")
##
# TODO: From DTW notebook: do i need this block??

if NUM_GPUS > 1:
    num_workers = 0  # nccl backend doesn't support num_workers>0
    rank_key = "RANK" if "RANK" in os.environ else "LOCAL_RANK"
    bz = base_bz * NUM_GPUS
    if rank_key not in os.environ:
        rank = 0
    else:
        rank = int(os.environ[rank_key])
    logging.info(f"SETTING CUDA DEVICE ON RANK: {rank}")

    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()
    # we cannot call DistributedSampler before pytorch lightning trainer.fit() is called,
    # or we get this error:
    # RuntimeError: Default process group has not been initialized, please make sure to call init_process_group.
    # always include at least one example of class 0 (silent EMG & parallel Audio) in batch
    # always include at least one example of class 1 (EMG & Audio) in batch
    # TrainBatchSampler = partial(Distributed`SizeAwareStratifiedBatchSampler,
    #     num_replicas=NUM_GPUS, max_len=max_len//8, always_include_class=1)
    # TrainBatchSampler = partial(DistributedStratifiedBatchSampler,
    #     num_replicas=NUM_GPUS)
    TrainBatchSampler = partial(
        DistributedSizeAwareStratifiedBatchSampler,
        num_replicas=NUM_GPUS,
        max_len=max_len // 8,
        always_include_class=0,
    )
    ValSampler = lambda: DistributedSampler(
        emg_datamodule.val, shuffle=False, num_replicas=NUM_GPUS
    )
    TestSampler = lambda: DistributedSampler(
        emg_datamodule.test, shuffle=False, num_replicas=NUM_GPUS
    )
else:
    # TrainBatchSampler = SizeAwareStratifiedBatchSampler
    TrainBatchSampler = partial(
        DistributedSizeAwareStratifiedBatchSampler,
        num_replicas=NUM_GPUS,
        max_len=max_len // 8,
        always_include_class=0,
    )
    # num_workers=32
    num_workers = 0  # prob better now that we're caching
    bz = base_bz
    ValSampler = None
    TestSampler = None
    rank = 0

if rank == 0:
    os.makedirs(output_directory, exist_ok=True)

##

# must run 2023-07-17_cache_dataset_with_attrs_.py first
librispeech_train_cache = os.path.join(
    scratch_directory, "librispeech", "librispeech_960_train_phoneme_cache"
)
librispeech_val_cache = os.path.join(
    scratch_directory, "librispeech", "librispeech_val_phoneme_cache"
)
librispeech_test_cache = os.path.join(
    scratch_directory, "librispeech", "librispeech_test_phoneme_cache"
)

speech_val = cache_dataset(librispeech_val_cache, LibrispeechDataset, per_index_cache)()
speech_train = cache_dataset(
    librispeech_train_cache, LibrispeechDataset, per_index_cache
)()
speech_train.len = 281185  # TODO: recompute cache and remove this hack
speech_test = cache_dataset(
    librispeech_test_cache, LibrispeechDataset, per_index_cache
)()


datamodule = EMGAndSpeechModule(
    emg_datamodule.train,
    emg_datamodule.val,
    emg_datamodule.test,
    speech_train,
    speech_val,
    speech_test,
    bz=bz,
    val_bz=val_bz,
    num_replicas=NUM_GPUS,
    pin_memory=(not DEBUG),
    num_workers=num_workers,
    TrainBatchSampler=TrainBatchSampler,
    ValSampler=ValSampler,
    TestSampler=TestSampler,
)
steps_per_epoch = len(datamodule.TrainBatchSampler) // grad_accum

# assert steps_per_epoch > 100, "too few steps per epoch"
# assert steps_per_epoch < 1000, "too many steps per epoch"
##
text_transform = TextTransform(togglePhones=togglePhones)
os.makedirs(output_directory, exist_ok=True)

n_chars = len(text_transform.chars)
num_outs = n_chars + 1  # +1 for CTC blank token ( i think? )
config = MONAConfig(
    steps_per_epoch=steps_per_epoch,
    lm_directory=lm_directory,
    num_outs=num_outs,
    precision=precision,
    gradient_accumulation_steps=grad_accum,
    learning_rate=learning_rate,
    audio_lambda=1.0,
    # neural_input_features=datamodule.train.n_features,
    neural_input_features=1,
    seqlen=seqlen,
    max_len=max_len,
    batch_size=base_bz,
    white_noise_sd=white_noise_sd,
    constant_offset_sd=constant_offset_sd,
    num_train_epochs=n_epochs,
    togglePhones=togglePhones,
    use_dtw=use_dtw,
    use_crossCon=use_crossCon,
    use_supCon=use_supCon,
    # d_inner=8,
    # d_model=8,
    fixed_length=True,
)

model = MONA(config, text_transform, no_neural=True)

##
logging.info("made model")

callbacks = [
    # starting at epoch 0, accumulate this many batches of gradients
    GradientAccumulationScheduler(scheduling={0: config.gradient_accumulation_steps})
]

if log_neptune:
    # need to store credentials in your shell env
    nep_key = os.environ["NEPTUNE_API_TOKEN"]
    neptune_kwargs = {
        "project": "neuro/Gaddy",
        "name": model.__class__.__name__,
        "tags": [
            model.__class__.__name__,
            isotime,
            f"fp{config.precision}",
        ],
    }
    if RESUME:
        neptune_logger = NeptuneLogger(
            run=neptune.init_run(
                with_id=run_id,
                api_token=os.environ["NEPTUNE_API_TOKEN"],
                **neptune_kwargs,
            ),
            log_model_checkpoints=False,
        )
    else:
        neptune_logger = NeptuneLogger(
            api_key=nep_key, **neptune_kwargs, log_model_checkpoints=False
        )
        neptune_logger.log_hyperparams(vars(config))
        neptune_logger.experiment["isotime"] = isotime
        neptune_logger.experiment["hostname"] = hostname.stdout.decode().strip()
        neptune_logger.experiment["output_directory"] = output_directory
        if "SLURM_JOB_ID" in os.environ:
            neptune_logger.experiment["SLURM_JOB_ID"] = os.environ["SLURM_JOB_ID"]

    checkpoint_callback = ModelCheckpoint(
        monitor="val/wer",
        mode="min",
        dirpath=output_directory,
        save_top_k=10,  # TODO: try averaging weights afterwards to see if improve WER..?
        filename=model.__class__.__name__ + "-{epoch:02d}-{val/wer:.3f}",
    )
    callbacks.extend(
        [
            checkpoint_callback,
            pl.callbacks.LearningRateMonitor(logging_interval="epoch"),
            # pl.callbacks.LearningRateMonitor(logging_interval="step"), # good for troubleshooting warmup
        ]
    )
else:
    neptune_logger = None

trainer = pl.Trainer(
    max_epochs=config.num_train_epochs,
    devices=devices,
    accelerator="gpu",
    # accelerator="cpu",
    gradient_clip_val=1,  # was 0.5 for best 26.x% run, gaddy used 10, llama 2 uses 1.0
    logger=neptune_logger,
    default_root_dir=output_directory,
    callbacks=callbacks,
    precision=config.precision,
    limit_train_batches=limit_train_batches,
    limit_val_batches=limit_val_batches,
    # strategy=strategy,
    # use_distributed_sampler=True,
    # use_distributed_sampler=False, # we need to make a custom distributed sampler
    # num_sanity_val_steps=num_sanity_val_steps,
    sync_batchnorm=True,
    strategy=strategy,
    # strategy='fsdp', # errors on CTC loss being used on half-precision.
    # also model only ~250MB of params, so fsdp may be overkill
    # check_val_every_n_epoch=10 # should give speedup of ~30% since validation is bz=1
    num_sanity_val_steps=0,
    # https://lightning.ai/docs/pytorch/stable/debug/debugging_intermediate.html#detect-autograd-anomalies
    # detect_anomaly=True # slooooow
)
##
logging.info("about to fit")
print(f"Sanity check: {len(datamodule.train)} training samples")
print(f"Sanity check: {len(datamodule.train_dataloader())} training batches")
# epoch of 242 if only train...
if RESUME:
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)
else:
    trainer.fit(model, datamodule=datamodule)

if log_neptune:
    ckpt_path = os.path.join(
        output_directory, f"finished-training_epoch={model.current_epoch}.ckpt"
    )
    trainer.save_checkpoint(ckpt_path)
    print(f"saved checkpoint to {ckpt_path}")

##
# dl = datamodule.train_dataloader()
dl = datamodule.val_dataloader()
for b in dl:
    break
b
##
split_batch_into_emg_neural_audio(b)
##
