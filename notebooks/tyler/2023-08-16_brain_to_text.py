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

from read_emg import EMGDataset, PreprocessedEMGDataset, \
    PreprocessedSizeAwareSampler, EMGDataModule, ensure_folder_on_scratch
from architecture import Model, S4Model, H3Model, ResBlock, MONAConfig, MONA
from data_utils import combine_fixed_length, decollate_tensor
from transformer import TransformerEncoderLayer
from pytorch_lightning.loggers import NeptuneLogger
# import neptune, shutil
import neptune.new as neptune, shutil
import typer
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
    split_batch_into_emg_neural_audio, DistributedSizeAwareStratifiedBatchSampler, \
    SizeAwareStratifiedBatchSampler, collate_gaddy_or_speech, \
    collate_gaddy_speech_or_neural, DistributedSizeAwareSampler
from functools import partial
from contrastive import cross_contrastive_loss, var_length_cross_contrastive_loss, \
    nobatch_cross_contrastive_loss, supervised_contrastive_loss

DEBUG = False
# DEBUG = True
RESUME = False
# RESUME = True


constant_offset_sd = 0.2
white_noise_sd = 1
# constant_offset_sd = 0
# white_noise_sd = 0
seqlen = 600
auto_lr_find = False

# see https://github.com/fwillett/speechBCI/blob/main/NeuralDecoder/neuralDecoder/configs/config.yaml
# learning_rate = 1e-3 # frank used 1e-2. but we saw lar spike from 3 to 8 in validation...
learning_rate = 3e-4
# learning_rate = 1.5e-4
togglePhones = False


app = typer.Typer()

@app.command()
def update_configs(
    constant_offset_sd_cli: float = typer.Option(0.2, "--constant-offset-sd"),
    white_noise_sd_cli: float = typer.Option(1, "--white-noise-sd"),
    debug_cli: bool = typer.Option(False, "--debug"),
    resume_cli: bool = typer.Option(False, "--resume"),
    grad_accum_cli: int = typer.Option(1, "--grad-accum"),
    precision_cli: str = typer.Option("16-mixed", "--precision"),
    logger_level_cli: str = typer.Option("WARNING", "--logger-level"),
    base_bz_cli: int = typer.Option(24, "--base-bz"),
    val_bz_cli: int = typer.Option(8, "--val-bz"),
    max_len_cli: int = typer.Option(48000, "--max-len"),
    seqlen_cli: int = typer.Option(300, "--seqlen")
):
    """Update configurations with command-line values."""
    global constant_offset_sd, white_noise_sd, DEBUG, RESUME, grad_accum
    global precision, logger_level, base_bz, val_bz, max_len, seqlen

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


if RESUME:
    # TODO: make an auto-resume feature...? or at least find ckpt_path from run_id
    # to think about: can we do this automatically on gaia/sherlock if OOM..? (maybe we don't care / can do manually)
    # INFO: when resuming logging to Neptune, we might repeat some steps,
    # e.g. if epoch 29 was lowest WER, but we resume at epoch 31, we will
    # log epoch 30 & 31 twice. mainly an issue for publication plots
    ckpt_path = '/scratch/2023-'
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
    # grad_accum = 2 # EMG only, 128000 max_len
    grad_accum = 1
    precision = "16-mixed"

    if ON_SHERLOCK:
        NUM_GPUS = 2
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
    t12_npz_path = os.path.join(scratch_directory, "2023-08-21_T12_dataset.npz")
    if len(os.sched_getaffinity(0)) > 16:
        print("WARNING: if you are running more than one script, you may want to use `taskset -c 0-16` or similar")
else:
    # on my local machine
    sessions_dir = '/data/magneto/'
    scratch_directory = "/scratch"
    gaddy_dir = '/scratch/GaddyPaper/'
    t12_npz_path = "/data/data/T12_data/synthetic_audio/2023-08-21_T12_dataset.npz"

print(f"CPU affinity: {os.sched_getaffinity(0)}")

data_dir = os.path.join(gaddy_dir, 'processed_data/')
lm_directory = os.path.join(gaddy_dir, 'pretrained_models/librispeech_lm/')
normalizers_file = os.path.join(SCRIPT_DIR, "normalizers.pkl")
togglePhones = False

if ON_SHERLOCK:
    lm_directory = ensure_folder_on_scratch(lm_directory, scratch_directory)
    
gpu_ram = torch.cuda.get_device_properties(0).total_memory / 1024**3

if gpu_ram < 24:
    # Titan RTX
    # val_bz = 16 # OOM
    # base_bz = 32
    base_bz = 24
    val_bz = 8
    # max_len = 24000 # OOM
    max_len = 12000 # approx 11000 / 143 = 77 bz. 75 * 2 GPU = 150 bz. still high..?
    # max_len = 18000 # no OOM, approx 110 bz (frank used 64)
    # assert NUM_GPUS == 2
elif gpu_ram > 30:
    # V100
    base_bz = 24
    # base_bz = 32 # OOM epoch ~4
    val_bz = 8
    max_len = 48000
    # assert NUM_GPUS == 4
else:
    raise ValueError("Unknown GPU")

if __name__ == "__main__" and "--cli" in sys.argv:
    app()


##
def load_npz_to_memory(npz_path, **kwargs):
    npz = np.load(npz_path, **kwargs)
    loaded_data = {k: npz[k] for k in npz}
    npz.close()
    return loaded_data

t12_npz = load_npz_to_memory(t12_npz_path, allow_pickle=True)
##
tot_trials = len(t12_npz['spikePow'])
missing_phones = np.sum(np.array([p is None for p in t12_npz['aligned_phonemes']]))
silent_trials = np.sum(np.array([p is None for p in t12_npz['mspecs']]))
missing_synth_audio = np.sum(np.array([p is None for p in t12_npz['tts_mspecs']]))
train_trials = np.sum(t12_npz['dataset_partition'] == 'train')
val_trials = np.sum(t12_npz['dataset_partition'] == 'test')

print("tot_trials:", tot_trials)
print("train_trials:", train_trials)
print("val_trials:", val_trials)
print("missing_phones:", missing_phones)
print("silent_trials:", silent_trials)
print("missing_synth_audio:", missing_synth_audio)
##

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
class NeuralDataset(torch.utils.data.Dataset):
    def __init__(self, neural, audio, phonemes, sentences, text_transform,
                 white_noise_sd=0, constant_offset_sd=0):
        self.neural = neural
        self.audio = audio
        self.phonemes = phonemes
        self.sentences = sentences
        self.text_transform = text_transform
        self.n_features = neural[0].shape[1]
        self.white_noise_sd = white_noise_sd
        self.constant_offset_sd = constant_offset_sd
        super().__init__()
    
    def __getitem__(self, idx):
        text_int = np.array(self.text_transform.text_to_int(self.sentences[idx]), dtype=np.int64)
        aud = self.audio[idx]
        aud = aud if aud is None else torch.from_numpy(aud)
        phon = self.phonemes[idx]
        phon = phon if phon is None else torch.from_numpy(phon)
        nf = torch.from_numpy(self.neural[idx])
        if self.white_noise_sd > 0:
            nf += torch.randn_like(nf) * self.white_noise_sd
        if self.constant_offset_sd > 0:
            nf += torch.randn(1) * self.constant_offset_sd
        return {
            "audio_features": aud,
            "neural_features": nf,
            "text": self.sentences[idx],
            "text_int": torch.from_numpy(text_int),
            "phonemes": phon,
        }
        
    def __len__(self):
        return len(self.neural)

class T12Dataset(NeuralDataset):
    def __init__(self, t12_npz, partition="train",
            audio_type="tts_mspecs", white_noise_sd=0, constant_offset_sd=0):
                #  audio_type="tts_mspecs"):
        """T12 BCI dataset.
        
        partition: train or test
        audio_type: mspecs, tts_mspecs, or aligned_tts_mspecs
        
        """
        idx = np.where(t12_npz["dataset_partition"] == partition)[0]
        neural = []
        audio = []
        spikePow = t12_npz["spikePow"]
        tx1 = t12_npz["tx1"]
        tx2 = t12_npz["tx2"]
        tx3 = t12_npz["tx3"]
        tx4 = t12_npz["tx4"]
        # mean, variance per block
        aud = t12_npz[audio_type]
        for i in tqdm(idx, desc="concatenating neural data"):
            block_idx = t12_npz["block"][i][0]
            session = t12_npz["session"][i]
            # print(session, block_idx)

            neural.append(np.concatenate([
                    # np.log10(spikePow[i][:,:128]+1) / 4, # map to approx 0-1
                    spikePow[i][:,:128],
                    tx1[i][:,:128],
                    # tx1[i][:,:128] / 25, # max val is 56
                    # tx2[i] / 25,
                    # tx3[i] / 25,
                    # tx4[i] / 25
                    ] # max val is 52
                , axis=1).astype(np.float32))
            if aud[i] is None:
                # for example, if audio_type is "mspecs" then we have no
                # audio for the silent trials
                if audio_type == "tts_mspecs":
                    print(f"WARNING: no audio for index {i}")
                audio.append(None)
            else:
                audio.append((aud[i]+5) / 5) # TODO: match librispeech
        phonemes = t12_npz["aligned_phonemes"][idx]
        sentences = t12_npz["sentences"][idx]
        text_transform = TextTransform(togglePhones = False)
        super().__init__(neural, audio, phonemes, sentences, text_transform,
            white_noise_sd=white_noise_sd, constant_offset_sd=constant_offset_sd)
        
class T12DataModule(pl.LightningDataModule):
    def __init__(self, t12_npz, audio_type="tts_mspecs", max_len=32000,
                 num_replicas=1, train_bz:int=32, val_bz:int=16, fixed_length=False,
                 white_noise_sd=1.0, constant_offset_sd=0.2):
        super().__init__()
        self.train = T12Dataset(t12_npz, partition="train", audio_type="tts_mspecs",
                white_noise_sd=white_noise_sd, constant_offset_sd=constant_offset_sd)
        self.val = T12Dataset(t12_npz, partition="test", audio_type="tts_mspecs")
        self.collate_fn = collate_gaddy_speech_or_neural

        self.train_bz = train_bz
        self.val_bz = val_bz
        self.fixed_length = fixed_length
        
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
                self.train,
                collate_fn=self.collate_fn,
                pin_memory=True,
                num_workers=0,
                batch_size=self.train_bz,
            )
        
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val,
            collate_fn=self.collate_fn,
            pin_memory=True,
            num_workers=0,
            batch_size=self.val_bz,
        )
        
    def test_dataloader(self):
        return None

    
# train_dset = T12Dataset(t12_npz, partition="train", audio_type="tts_mspecs")
datamodule = T12DataModule(t12_npz, audio_type="tts_mspecs",
    num_replicas=NUM_GPUS, max_len=max_len, train_bz=base_bz, val_bz=val_bz*NUM_GPUS,
    white_noise_sd=white_noise_sd, constant_offset_sd=constant_offset_sd)
##
# TODO: why is this super slow on sherlock now (5 minutes..?) It was seconds before...
# we must be hitting filesystem...
for t in tqdm(datamodule.train, desc="checking for NaNs"):
    if torch.any(torch.isnan(t['neural_features'])):
        print("got NaN for neural_features")
        break
    if t['audio_features'] is not None and torch.any(torch.isnan(t['audio_features'])):
        print("got NaN for audio_features")
        break
    if t['phonemes'] is not None and torch.any(torch.isnan(t['phonemes'])):
        print("got NaN for phones")
        break
    if torch.any(torch.isnan(t['text_int'])):
        print("got NaN for text_int")
        break
##
emg_tup, neural_tup, audio_tup, idxs = split_batch_into_emg_neural_audio(collate_gaddy_speech_or_neural([datamodule.train[i] for i in range(5)]))
(neural, length_neural, neural_phonemes, y_length_neural, y_neural) = neural_tup
neural, length_neural
##
# import matplotlib.pyplot as plt

# neural_features = train_dset[10]['neural_features'].reshape(-1)
# plt.hist(np.sqrt(neural_features), bins=50)
# plt.title("Histogram of sqrt(Neural Features)")
# plt.xlabel("Value")
# plt.ylabel("Frequency")
# plt.show()

# neural_features = train_dset[10]['neural_features'].reshape(-1)
# plt.hist(np.log2(neural_features+1), bins=50)
# plt.title("Histogram of log2(Neural Features + 1)")
# plt.xlabel("Value")
# plt.ylabel("Frequency")
# plt.show()

# neural_features = train_dset[10]['neural_features'].reshape(-1)
# plt.hist(np.log10(neural_features+1)/4, bins=50)
# plt.title("Histogram of log10(Neural Features + 1)/4")
# plt.xlabel("Value")
# plt.ylabel("Frequency")
# plt.show()

# mspec = train_dset[10]['audio_features'].reshape(-1)
# plt.hist((mspec+5)/5, bins=50)
# plt.title("Histogram of (mspec+5)/5")
# plt.xlabel("Value")
# plt.ylabel("Frequency")
# plt.show()

##
# max([x.max() for x in t12_npz["spikePow"]])
##
text_transform = TextTransform(togglePhones = togglePhones)
os.makedirs(output_directory, exist_ok=True)

# steps_per_epoch = len(datamodule.TrainBatchSampler) // grad_accum
steps_per_epoch = len(datamodule.train) // base_bz // NUM_GPUS // grad_accum
# steps_per_epoch = len(datamodule.train_dataloader()) # may crash if distributed

n_chars = len(text_transform.chars)
num_outs = n_chars + 1 # +1 for CTC blank token ( i think? )
config = MONAConfig(steps_per_epoch, lm_directory, num_outs,
    precision=precision, gradient_accumulation_steps=grad_accum,
    learning_rate=learning_rate, audio_lambda=0.,
    neural_input_features=datamodule.train.n_features,
    seqlen=seqlen, max_len=max_len,
    white_noise_sd=white_noise_sd, constant_offset_sd=constant_offset_sd)

model = MONA(config, text_transform)
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
        save_top_k=10, # TODO: try averaging weights afterwards to see if improve WER..?
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
    
# TODO: why are there only 16 * 35 = 560 sentences in validation..? shouldn't there be 1000?
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
