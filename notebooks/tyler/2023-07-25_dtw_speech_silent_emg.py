##
2
##
# %load_ext autoreload
# %autoreload 2
##
import os, subprocess
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "backend:cudaMallocAsync" # no OOM
hostname = subprocess.run("hostname", capture_output=True)
ON_SHERLOCK = hostname.stdout[:2] == b"sh"
if ON_SHERLOCK:
    os.environ["SLURM_JOB_NAME"] = "interactive" # best practice for pytorch lightning...
    os.environ["SLURM_NTASKS"] = "1" # best practice for pytorch lightning...
    # best guesses
    os.environ["SLURM_LOCALID"] = "0" # Migtht be used by pytorch lightning...
    os.environ["SLURM_NODEID"] = "0" # Migtht be used by pytorch lightning...
    os.environ["SLURM_NTASKS_PER_NODE"] = "1" # Migtht be used by pytorch lightning...
    os.environ["SLURM_PROCID"] = "0" # Migtht be used by pytorch lightning...

# from pl source code
# "SLURM_NODELIST": "1.1.1.1, 1.1.1.2",
# "SLURM_JOB_ID": "0001234",
# "SLURM_NTASKS": "20",
# "SLURM_NTASKS_PER_NODE": "10",
# "SLURM_LOCALID": "2",
# "SLURM_PROCID": "1",
# "SLURM_NODEID": "3",
# "SLURM_JOB_NAME": "JOB",

import pytorch_lightning as pl, pickle
import sys, warnings
import numpy as np
import logging
import torchmetrics
import random
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

from read_emg import EMGDataset, SizeAwareSampler, PreprocessedEMGDataset, \
    PreprocessedSizeAwareSampler, EMGDataModule, ensure_folder_on_scratch
from architecture import Model, S4Model, H3Model, ResBlock, SpeechOrEMGToTextConfig, SpeechOrEMGToText
from data_utils import combine_fixed_length, decollate_tensor
from transformer import TransformerEncoderLayer
from pytorch_lightning.loggers import NeptuneLogger
# import neptune, shutil
import neptune.new as neptune, shutil
from datetime import datetime
from pytorch_lightning.callbacks import ModelCheckpoint, GradientAccumulationScheduler
from pytorch_lightning.profilers import SimpleProfiler, AdvancedProfiler, PyTorchProfiler, PassThroughProfiler
from pytorch_lightning.strategies import DDPStrategy
from data_utils import TextTransform
from typing import List
from collections import defaultdict
from enum import Enum
from magneto.preprocessing import ensure_data_on_scratch
from dataloaders import LibrispeechDataset, EMGAndSpeechModule, \
    DistributedStratifiedBatchSampler, StratifiedBatchSampler, cache_dataset, \
    split_batch_into_emg_audio, DistributedSizeAwareStratifiedBatchSampler, \
    SizeAwareStratifiedBatchSampler, collate_gaddy_or_speech
from functools import partial
from contrastive import cross_contrastive_loss, var_length_cross_contrastive_loss, \
    nobatch_cross_contrastive_loss, supervised_contrastive_loss

DEBUG = False
# DEBUG = True
RESUME = False
# RESUME = True

if RESUME:
    # TODO: make an auto-resume feature...? or at least find ckpt_path from run_id
    # to think about: can we do this automatically on gaia/sherlock if OOM..? (maybe we don't care / can do manually)
    # INFO: when resuming logging to Neptune, we might repeat some steps,
    # e.g. if epoch 29 was lowest WER, but we resume at epoch 31, we will
    # log epoch 30 & 31 twice. mainly an issue for publication plots
    # ckpt_path = '/scratch/2023-07-10T12:20:43.920850_gaddy/SpeechOrEMGToText-epoch=29-val/wer=0.469.ckpt'
    ckpt_path = '/scratch/2023-08-03T21:30:03.418151_gaddy/SpeechOrEMGToText-epoch=15-val/wer=0.547.ckpt'
    run_id = 'GAD-493'
    

per_index_cache = True # read each index from disk separately
# per_index_cache = False # read entire dataset from disk


isotime = datetime.now().isoformat()

if DEBUG:
    NUM_GPUS = 1
    limit_train_batches = 2
    limit_val_batches = 2 # will not run on_validation_epoch_end
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
    NUM_GPUS = 2
    # grad_accum = 3
    grad_accum = 2 # EMG only, 128000 max_len
    precision = "16-mixed"

    if ON_SHERLOCK:
        NUM_GPUS = 4
        grad_accum = 1
        # precision = "32"
    # variable length batches are destroying pytorch lightning
    # limit_train_batches = 900 # validation loop doesn't run at 900 ?! wtf
    # limit_train_batches = 100 # validation loop runs at 100
    # limit_train_batches = 500
    limit_train_batches = None
    limit_val_batches = None
    log_neptune = True
    # log_neptune = False
    n_epochs = 200
    num_sanity_val_steps = 0 # may prevent crashing of distributed training
    # grad_accum = 2 # NaN loss at epoch 67 with BatchNorm, two gpu, grad_accum=2, base_bz=16
    
    # if BatchNorm still causes issues can try RunningBatchNorm (need to implement for distributed)
    # https://youtu.be/HR0lt1hlR6U?t=7543
    logger_level = logging.WARNING


assert os.environ["NEPTUNE_ALLOW_SELF_SIGNED_CERTIFICATE"] == 'TRUE', "run this in shell: export NEPTUNE_ALLOW_SELF_SIGNED_CERTIFICATE='TRUE'"

# load our data file paths and metadata:

if per_index_cache:
    cache_suffix = "_per_index"
else:
    cache_suffix = ""
if ON_SHERLOCK:
    sessions_dir = '/oak/stanford/projects/babelfish/magneto/'
    # TODO: bechmark SCRATCH vs LOCAL_SCRATCH ...?
    scratch_directory = os.environ["SCRATCH"]
    # scratch_directory = os.environ["LOCAL_SCRATCH"]
    gaddy_dir = '/oak/stanford/projects/babelfish/magneto/GaddyPaper/'
    scratch_lengths_pkl = os.path.join(scratch_directory, "2023-07-25_emg_speech_dset_lengths.pkl")
    tmp_lengths_pkl = os.path.join("/tmp", "2023-07-25_emg_speech_dset_lengths.pkl")
    if os.path.exists(scratch_lengths_pkl) and not os.path.exists(tmp_lengths_pkl):
        shutil.copy(scratch_lengths_pkl, tmp_lengths_pkl)
else:
    # on my local machine
    sessions_dir = '/data/magneto/'
    scratch_directory = "/scratch"
    gaddy_dir = '/scratch/GaddyPaper/'
    
data_dir = os.path.join(gaddy_dir, 'processed_data/')
lm_directory = os.path.join(gaddy_dir, 'pretrained_models/librispeech_lm/')
normalizers_file = os.path.join(SCRIPT_DIR, "normalizers.pkl")
togglePhones = False

if ON_SHERLOCK:
    lm_directory = ensure_folder_on_scratch(lm_directory, scratch_directory)
    
gpu_ram = torch.cuda.get_device_properties(0).total_memory / 1024**3

if gpu_ram < 24:
    # Titan RTX
    base_bz = 12
    # TODO: need to fix by using size-aware sampling for dataloader
    # base_bz = 4
    # base_bz = 16 # OOM epoch 9 with Titan RTX for batch-level infoNCE
    # val_bz = base_bz
    val_bz = 8
    max_len = 48000 # works for supNCE on Titan RTX
    # max_len = 128000 # for emg only, no contrastive loss
    # max_len = 2 * 128000 # for emg only, no contrastive loss, 1 GPU
    # assert NUM_GPUS == 2
elif gpu_ram > 30:
    # V100
    # base_bz = 24
    base_bz = 12 # don't think does anything..?
    val_bz = 8
    # max_len = 64000 # OOM epoch 32
    # max_len = 56000
    max_len = 48000 # possibly better performance than 56000, def less memory
    # assert NUM_GPUS == 4
else:
    raise ValueError("Unknown GPU")


##
# needed for using CachedDataset
emg_datamodule = EMGDataModule(data_dir, togglePhones, normalizers_file, max_len=max_len,
    collate_fn=collate_gaddy_or_speech,
    pin_memory=(not DEBUG), batch_size=val_bz)
emg_train = emg_datamodule.train

mfcc_norm, emg_norm = pickle.load(open(normalizers_file,'rb'))

if ON_SHERLOCK:
    # TODO: should we just use the scratch directory over LOCAL_SCRATCH?
    output_directory = os.path.join(os.environ["SCRATCH"], f"{isotime}_gaddy")
else:
    output_directory = os.path.join(scratch_directory, f"{isotime}_gaddy")

logging.basicConfig(handlers=[
        logging.StreamHandler()
        ], level=logger_level, format="%(message)s",
        force=True)

logging.debug("DEBUG mode")
if not log_neptune:
    logging.warning("not logging to neptune")
##
auto_lr_find = False

# learning_rate = 3e-4
learning_rate = 1.5e-4
# 3e-3 leads to NaNs, prob need to have slower warmup in this case
togglePhones = False
text_transform = TextTransform(togglePhones = togglePhones)
##

n_chars = len(text_transform.chars)

if NUM_GPUS > 1:
    num_workers=0 # nccl backend doesn't support num_workers>0
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
    # TrainBatchSampler = partial(DistributedSizeAwareStratifiedBatchSampler,
    #     num_replicas=NUM_GPUS, max_len=max_len//8, always_include_class=1)
    # TrainBatchSampler = partial(DistributedStratifiedBatchSampler,
    #     num_replicas=NUM_GPUS)
    TrainBatchSampler = partial(DistributedSizeAwareStratifiedBatchSampler,
        num_replicas=NUM_GPUS, max_len=max_len//8, always_include_class=0)
    ValSampler = lambda: DistributedSampler(emg_datamodule.val,
        shuffle=False, num_replicas=NUM_GPUS)
    TestSampler = lambda: DistributedSampler(emg_datamodule.test,
        shuffle=False, num_replicas=NUM_GPUS)
else:
    # TrainBatchSampler = SizeAwareStratifiedBatchSampler
    TrainBatchSampler = partial(DistributedSizeAwareStratifiedBatchSampler,
        num_replicas=NUM_GPUS, max_len=max_len//8, always_include_class=0)
    # num_workers=32
    num_workers=0 # prob better now that we're caching
    bz = base_bz
    ValSampler = None
    TestSampler = None
    rank = 0

if rank == 0:
    os.makedirs(output_directory, exist_ok=True)


# must run 2023-07-17_cache_dataset_with_attrs_.py first
librispeech_train_cache = os.path.join(scratch_directory, "librispeech", "librispeech_960_train_phoneme_cache")
librispeech_val_cache = os.path.join(scratch_directory, "librispeech", "librispeech_val_phoneme_cache")
librispeech_test_cache = os.path.join(scratch_directory, "librispeech", "librispeech_test_phoneme_cache")

speech_val = cache_dataset(librispeech_val_cache, LibrispeechDataset, per_index_cache)()
speech_train = cache_dataset(librispeech_train_cache, LibrispeechDataset, per_index_cache)()
speech_train.len = 281185 # TODO: recompute cache and remove this hack
speech_test = cache_dataset(librispeech_test_cache, LibrispeechDataset, per_index_cache)()
##
datamodule =  EMGAndSpeechModule(emg_datamodule.train,
    emg_datamodule.val, emg_datamodule.test,
    speech_train, speech_val, speech_test,
    bz=bz, val_bz=val_bz, num_replicas=NUM_GPUS, pin_memory=(not DEBUG),
    num_workers=num_workers,
    TrainBatchSampler=TrainBatchSampler,
    ValSampler=ValSampler,
    TestSampler=TestSampler
)
steps_per_epoch = len(datamodule.TrainBatchSampler) // grad_accum
# steps_per_epoch = len(datamodule.train_dataloader()) # may crash if distributed

num_outs = n_chars + 1 # +1 for CTC blank token ( i think? )
config = SpeechOrEMGToTextConfig(steps_per_epoch, lm_directory, num_outs, precision=precision)

model = SpeechOrEMGToText(config, text_transform)
logging.info('made model')

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
        "tags": [model.__class__.__name__,
                isotime,
                f"fp{config.precision}",
                ],
    }
    if RESUME:
        neptune_logger = NeptuneLogger(
            run = neptune.init_run(with_id=run_id,
                api_token=os.environ["NEPTUNE_API_TOKEN"],
                **neptune_kwargs),
            log_model_checkpoints=False
        )
    else:
        neptune_logger = NeptuneLogger(api_key=nep_key,
            **neptune_kwargs,
            log_model_checkpoints=False
        )
        neptune_logger.log_hyperparams(vars(config))
        neptune_logger.experiment["isotime"] = isotime
        neptune_logger.experiment["hostname"] = hostname
        neptune_logger.experiment["output_directory"] = output_directory

    checkpoint_callback = ModelCheckpoint(
        monitor="val/wer",
        mode="min",
        dirpath=output_directory,
        filename=model.__class__.__name__+"-{epoch:02d}-{val/wer:.3f}",
    )
    callbacks.extend([
        checkpoint_callback,
        pl.callbacks.LearningRateMonitor(logging_interval="epoch"),
        # pl.callbacks.LearningRateMonitor(logging_interval="step"), # good for troubleshooting warmup
    ])
else:
    neptune_logger = None

if NUM_GPUS > 1:
    devices = 'auto'
    strategy=DDPStrategy(gradient_as_bucket_view=True, find_unused_parameters=True)
elif NUM_GPUS == 1:
    devices = [0]
    strategy = "auto"
else:
    devices = 'auto'
    strategy = "auto"
trainer = pl.Trainer(
    max_epochs=config.num_train_epochs,
    devices=devices,
    accelerator="gpu",
    # accelerator="cpu",
    gradient_clip_val=1, # was 0.5 for best 26.x% run, gaddy used 10, llama 2 uses 1.0
    logger=neptune_logger,
    default_root_dir=output_directory,
    callbacks=callbacks,
    precision=config.precision,
    limit_train_batches=limit_train_batches,
    limit_val_batches=limit_val_batches,
    # strategy=strategy,
    use_distributed_sampler=False, # we need to make a custom distributed sampler
    # num_sanity_val_steps=num_sanity_val_steps,
    sync_batchnorm=True,
    strategy=strategy,
    # strategy='fsdp', # errors on CTC loss being used on half-precision.
    # also model only ~250MB of params, so fsdp may be overkill
    # check_val_every_n_epoch=10 # should give speedup of ~30% since validation is bz=1
    num_sanity_val_steps=0,
)

if auto_lr_find:
    # TODO: might be deprecated
    # https://lightning.ai/docs/pytorch/stable/upgrade/from_1_9.html
    # https://lightning.ai/docs/pytorch/stable/advanced/training_tricks.html#learning-rate-finder
    tuner = pl.tuner.Tuner(trainer)
    tuner.lr_find(model, datamodule)
        
logging.info('about to fit')
# epoch of 242 if only train...
if RESUME:
    trainer.fit(model, datamodule=datamodule,
        ckpt_path=ckpt_path)
else:
    trainer.fit(model, datamodule=datamodule)
    
if log_neptune:
    ckpt_path = os.path.join(output_directory,f"finished-training_epoch={model.current_epoch}.ckpt")
    trainer.save_checkpoint(ckpt_path)
    print(f"saved checkpoint to {ckpt_path}")
##
