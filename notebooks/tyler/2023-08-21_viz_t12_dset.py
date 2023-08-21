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
import matplotlib.pyplot as plt
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
    # t12_npz_path = "/data/data/T12_data/synthetic_audio/2023-08-20_T12_dataset.npz"
    t12_npz_path = "/data/data/T12_data/synthetic_audio/2023-08-21_T12_dataset.npz"
    # t12_npz_path = "/data/data/T12_data/synthetic_audio/2023-08-19_T12_dataset.npz"
    
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
    val_bz = 8
    # max_len = 24000 # OOM
    max_len = 12000 # approx 11000 / 143 = 77 bz. 75 * 2 GPU = 150 bz. still high..?
    # max_len = 18000 # no OOM, approx 110 bz (frank used 64)
    # assert NUM_GPUS == 2
elif gpu_ram > 30:
    # V100
    # base_bz = 24
    base_bz = 12 # don't think does anything..?
    val_bz = 8
    max_len = 48000
    # assert NUM_GPUS == 4
else:
    raise ValueError("Unknown GPU")


t12_npz = np.load(t12_npz_path, allow_pickle=True)
t12_og = np.load("/data/data/T12_data/synthetic_audio/2023-08-19_T12_dataset.npz",
                 allow_pickle=True)
##
tot_trials = len(t12_npz['spikePow'])
missing_phones = np.sum(np.array([p is None for p in t12_npz['aligned_phonemes']]))
silent_trials = np.sum(np.array([p is None for p in t12_npz['mspecs']]))
missing_synth_audio = np.sum(np.array([p is None for p in t12_npz['tts_mspecs']]))

print("tot_trials:", tot_trials)
print("missing_phones:", missing_phones)
print("silent_trials:", silent_trials)
print("missing_synth_audio:", missing_synth_audio)
##
list(t12_npz.keys())
##
i = -1
tx1 = t12_npz['spikePow'][i]
fig, ax = plt.subplots(2, 1, figsize=(20, 10))
im1 = ax[0].imshow(t12_og['spikePow'][i].T)
ax[0].set_title("original")
fig.colorbar(im1, ax=ax[0])
im2 = ax[1].imshow(tx1.T)
ax[1].set_title("z-scored")
fig.colorbar(im2, ax=ax[1])
# plt.imshow(tx1)
##
maxval = 0
minval = 100000
for sp in t12_npz['spikePow']:
    maxval = max(maxval, np.max(sp))
    minval = min(minval, np.min(sp))
maxval, minval
##
maxval = 0
minval = 100000
for sp in t12_npz['tx1']:
    maxval = max(maxval, np.max(sp))
    minval = min(minval, np.min(sp))
maxval, minval
##
