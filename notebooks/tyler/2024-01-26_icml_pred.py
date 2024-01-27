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
    get_best_ckpts, nep_get, get_emg_pred, get_audio_pred, get_last_ckpt


# https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
# not sure if makes a difference since we use fp16
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
##
def load_model_from_id(run_id, choose="best"):
    assert choose in ["best", "last"]
    
    neptune_logger = NeptuneLogger(
        run=neptune.init_run(
            with_id=run_id,
            api_token=os.environ["NEPTUNE_API_TOKEN"],
            mode='read-only',
            project="neuro/Gaddy",
        ),
        log_model_checkpoints=False,
    )
    output_directory = nep_get(neptune_logger, "output_directory")
    hparams = nep_get(neptune_logger, "training/hyperparams")
    if choose == "best":
        ckpt_paths, wers = get_best_ckpts(output_directory, n=1)
        wer = wers[0]
        ckpt_path = ckpt_paths[0]
        min_wer = nep_get(neptune_logger, "training/val/wer").value.min()
        assert np.isclose(wer, min_wer, atol=1e-3), f"wer {wer} != min_wer {min_wer}"
        print("found checkpoint with WER", wer)
    elif choose == "last":
        ckpt_path, epoch = get_last_ckpt(output_directory)
        assert epoch == hparams["max_epochs"], f"epoch {epoch} != max_epochs {hparams['max_epochs']}"
        print("found checkpoint with epoch", epoch)
    togglePhones = hparams["togglePhones"]
    assert togglePhones == False, "not implemented"
    if "use_supCon" in hparams:
        hparams["use_supTcon"] = hparams["use_supCon"]
        del hparams["use_supCon"]
    config = MONAConfig(**hparams)
    
    model = load_model(ckpt_path, config)
    return model, config, output_directory

def load_dataloaders(max_len=128000, togglePhones=False):

    assert ON_SHERLOCK
    scratch_directory = os.environ["SCRATCH"]
    gaddy_dir = os.path.join(scratch_directory, "GaddyPaper")
    # t12_npz_path = os.path.join(scratch_directory, "2023-08-21_T12_dataset.npz")
    # T12_dir = os.path.join(scratch_directory, "T12_data_v4")
    lm_directory = "/oak/stanford/projects/babelfish/magneto/GaddyPaper/icml_lm/"
    lm_directory = ensure_folder_on_scratch(lm_directory, os.environ["LOCAL_SCRATCH"])

    data_dir = os.path.join(gaddy_dir, "processed_data/")
    normalizers_file = os.path.join(SCRIPT_DIR, "normalizers.pkl")
    val_bz = 8

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
    
    return val_dl, test_dl
##
run_ids = [
#### crossCon + supTcon + DTW
823, 816, 822, 844, 839,
# 887,
#### crossCon + supTcon
815, 831,
# 840,
867, 825, 908,
#### crossCon
835, 841, 818, 868,
# 850, 936,
#### supTcon
# GAD: 890, 891, 904, 896, 905, 897,
#### supTcon + DTW
# 907, 906, 921, 920, 922,
#### EMG + Audio
871, 848, 861, 881,
# 837, 827, 926,
#### EMG
# GAD: 863, 832, 819, 852, 888, 893,
#### EMG - TAKE 2
909, 911,
# 925, 910,
912, 
#### Audio
# 931, 933, 929, 930, 932,
]

max_len = None
togglePhones = None
for ri in run_ids:
    model, config, output_directory = load_model_from_id(ri)
    if max_len != config.max_len or togglePhones != config.togglePhones:
        val_dl, test_dl = load_dataloaders(max_len=config.max_len, togglePhones=config.togglePhones)
        max_len = config.max_len
        togglePhones = config.togglePhones
    emg_val_pred = get_emg_pred(model, val_dl)
    emg_test_pred = get_emg_pred(model, test_dl)
    audio_val_pred = get_audio_pred(model, val_dl)
    audio_test_pred = get_audio_pred(model, test_dl)
    predictions = {
    "emg_val_pred": emg_val_pred,
    "emg_test_pred": emg_test_pred,
    "audio_val_pred": audio_val_pred,
    "audio_test_pred": audio_test_pred,
}
    path = os.path.join(output_directory, "2024-01-26_predictions.pkl")
    with open(path, "wb") as f:
        pickle.dump(predictions, f)
    print("done with run", ri)
print("finished!")
