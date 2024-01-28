# given list of Neptune run-ids, save val & test predictions to disk for the
# best checkpoint of each run

##
%load_ext autoreload
%autoreload 2
##
import os, subprocess

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "backend:cudaMallocAsync" # no OOM
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
hostname = subprocess.run("hostname", capture_output=True)
ON_SHERLOCK = hostname.stdout[:2] == b"sh"

import pytorch_lightning as pl, pickle
import sys, warnings, re
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
from torchaudio.models.decoder import ctc_decoder

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
    BalancedBinPackingBatchSampler,
)
from functools import partial
from contrastive import (
    cross_contrastive_loss,
    var_length_cross_contrastive_loss,
    nobatch_cross_contrastive_loss,
    supervised_contrastive_loss,
)
import glob, scipy
from helpers import load_npz_to_memory, calc_wer, load_model, get_top_k, \
    get_best_ckpts, nep_get, get_emg_pred, get_audio_pred, get_last_ckpt, \
    load_model_from_id


# https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
# not sure if makes a difference since we use fp16
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
##
def load_dataloaders(max_len=128000, togglePhones=False):

    assert ON_SHERLOCK
    scratch_directory = os.environ["SCRATCH"]
    gaddy_dir = os.path.join(scratch_directory, "GaddyPaper")
    # t12_npz_path = os.path.join(scratch_directory, "2023-08-21_T12_dataset.npz")
    # T12_dir = os.path.join(scratch_directory, "T12_data_v4")

    data_dir = os.path.join(gaddy_dir, "processed_data/")
    normalizers_file = os.path.join(SCRIPT_DIR, "normalizers.pkl")
    # librispeech_directory = "/oak/stanford/projects/babelfish/magneto/librispeech-cache"
    librispeech_directory = os.path.join(os.environ['SCRATCH'], "librispeech-cache")
    val_bz = 8
    
    librispeech_val_cache = os.path.join(
        librispeech_directory, "2024-01-23_librispeech_noleak_val_phoneme_cache"
    )
    librispeech_test_cache = os.path.join(
        librispeech_directory, "2024-01-23_librispeech_noleak_test_phoneme_cache"
    )

    emg_datamodule = EMGDataModule(
        data_dir,
        togglePhones,
        normalizers_file,
        max_len=max_len,
        collate_fn=collate_gaddy_or_speech,
        pin_memory=True,
    )

    collate_fn = collate_gaddy_or_speech


    val_dl = torch.utils.data.DataLoader(
        emg_datamodule.val,
        collate_fn=collate_fn,
        pin_memory=True,
        num_workers=0,
        batch_size=val_bz,
    )
    test_dl = torch.utils.data.DataLoader(
        emg_datamodule.test,
        collate_fn=collate_fn,
        pin_memory=True,
        num_workers=0,
        batch_size=val_bz,
    )
    
    per_index_cache = True  # read each index from disk separately
    
    speech_val = cache_dataset(
        librispeech_val_cache,
        LibrispeechDataset,
        per_index_cache,
        remove_attrs_before_save=["dataset"],
    )()
    
    speech_test = cache_dataset(
        librispeech_test_cache,
        LibrispeechDataset,
        per_index_cache,
        remove_attrs_before_save=["dataset"],
    )()
    
    libri_val_dl = torch.utils.data.DataLoader(
        speech_val,
        collate_fn=collate_fn,
        pin_memory=True,
        num_workers=0,
        batch_size=val_bz,
    )

    libri_test_dl = torch.utils.data.DataLoader(
        speech_test,
        collate_fn=collate_fn,
        pin_memory=True,
        num_workers=0,
        batch_size=val_bz,
    )

    
    return val_dl, test_dl, libri_val_dl, libri_test_dl
##
run_ids = [
    #### crossCon + supTcon + DTW ####
    823, 822, 844, 839, 887,
    # 816 extra, drop per criteria
    #### crossCon + supTcon ####
    815, 831, 908, 867, 825,
    # 840, # skip, logging issue
    #### crossCon ####
    835, 841, 818, 868, 936,
    # 850, # skip
    #### supTcon ####
    890, 891, 904, 905, 897,
    # 896 # skip
    #### supTcon + DTW ####
    907, 906, 921, 922, 920,
    #### EMG + Audio ####
    871, 848, 861, 881, 926,
    # 837, 827 # drop per selection criteria
    #### EMG + Audio (no librispeech ####
    # 960, 961, 962, 963, 964 # not yet finished
    #### EMG ####
    888, 893,
    # 944, 943, 942 # not yet finished
    # 863, 832, 819, 852, # issues with runs

    ######## quest for the best ##########
    #### crossCon 256k ####
    937, 938, 939, 940, 941,
    
    #### crossCon no librispeech 256k ####
]
audio_only_run_ids = [
    932, 933,
    # 945, 946, 947 # not yet done
    # 929, 930 # missing last epoch
]
run_ids = [920]
run_ids = [f"GAD-{ri}" for ri in run_ids]
audio_only_run_ids = [f"GAD-{ri}" for ri in audio_only_run_ids]
# run_ids = run_ids + audio_only_run_ids
# run_ids = audio_only_run_ids

max_len = None
togglePhones = None
for ri in run_ids:
    if ri in audio_only_run_ids:
        choose = "last"
    else:
        choose = "best"
    model, config, output_directory = load_model_from_id(ri, choose=choose)
    if max_len != config.max_len or togglePhones != config.togglePhones:
        val_dl, test_dl, libri_val_dl, libri_test_dl = load_dataloaders(
            max_len=config.max_len, togglePhones=config.togglePhones)
        max_len = config.max_len
        togglePhones = config.togglePhones
    emg_val_pred = get_emg_pred(model, val_dl)
    emg_test_pred = get_emg_pred(model, test_dl)
    audio_val_pred = get_audio_pred(model, val_dl)
    audio_test_pred = get_audio_pred(model, test_dl)
    libri_val_pred = get_audio_pred(model, libri_val_dl)
    libri_test_pred = get_audio_pred(model, libri_test_dl)

    predictions = {
        "emg_val_pred": emg_val_pred,
        "emg_test_pred": emg_test_pred,
        "audio_val_pred": audio_val_pred,
        "audio_test_pred": audio_test_pred,
        "librispeech_val_pred": libri_val_pred,
        "librispeech_test_pred": libri_test_pred,
    }
    path = os.path.join(output_directory, "2024-01-27_predictions.pkl")
    with open(path, "wb") as f:
        pickle.dump(predictions, f)
    print("done with run", ri)
print("finished!")

##
