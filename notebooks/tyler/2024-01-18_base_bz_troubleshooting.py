# question: why does base_bz affect the number of steps..?
# answer: because of the line `bz = base_bz * NUM_GPUS` and it changes how
# quickly we run out of samples. however the current sampling is too complicated,
# so switching to new sampling method.
##
# 2023-07-25_dtw_speech_silent_emg.py : best sEMG results
# 2023-08-24_brain_to_text_comp_split.py : most recent brain-to-text results, uses MONA name
2
##
%load_ext autoreload
%autoreload 2
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
    grad_accum = 2  # might need if run on 1 GPU
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
# lm_directory = os.path.join(gaddy_dir, "pretrained_models/librispeech_lm/")
lm_directory = "/oak/stanford/projects/babelfish/magneto/GaddyPaper/icml_lm/"
normalizers_file = os.path.join(SCRIPT_DIR, "normalizers.pkl")

if ON_SHERLOCK:
    lm_directory = ensure_folder_on_scratch(lm_directory, scratch_directory)

gpu_ram = torch.cuda.get_device_properties(0).total_memory / 1024**3
assert gpu_ram > 70, "needs A100 80GB"
##
base_bz = 10000
# base_bz = 32 
# base_bz = 32 
# base_bz = 48
val_bz = 16
# max_len = 48000 # from best perf with 4 x V100
# max_len = 128000 # OOM on A100 80GB
# max_len = 64000
max_len = 96000

##

# needed for using CachedDataset
emg_datamodule = EMGDataModule(
    data_dir,
    False,
    normalizers_file,
    max_len=max_len,
    collate_fn=collate_gaddy_or_speech,
    pin_memory=(not DEBUG),
    batch_size=val_bz,
)

##
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
steps_per_epoch
##
avg_len = []
for b in datamodule.train_dataloader():
    avg_len.append(sum(e.shape[0] for e in b["audio_features"]))
np.array(avg_len).mean()
##
