##
2
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

# run_id = "GAD-739"
# run_id = "GAD-762" # best ckpts missing??
# run_id = "GAD-835"
run_id = "GAD-929"
nep_key = os.environ["NEPTUNE_API_TOKEN"]
neptune_kwargs = {
    "project": "neuro/Gaddy",
}
neptune_logger = NeptuneLogger(
    run=neptune.init_run(
        with_id=run_id,
        api_token=os.environ["NEPTUNE_API_TOKEN"],
        mode='read-only',
        **neptune_kwargs,
    ),
    log_model_checkpoints=False,
)
##
output_directory = nep_get(neptune_logger, "output_directory")
hparams = nep_get(neptune_logger, "training/hyperparams")
# ckpt_paths, wers = get_best_ckpts(output_directory, n=1)
ckpt_paths = get_last_ckpt(output_directory)
ckpt_path = ckpt_paths[0]
# wer = wers[0]
wer = 1.0
min_wer = nep_get(neptune_logger, "training/val/wer").value.min()
# assert np.isclose(wer, min_wer, atol=1e-3), f"wer {wer} != min_wer {min_wer}"
print("found checkpoint with WER", wer)
max_len = hparams["max_len"]
togglePhones = hparams["togglePhones"]
if "use_supCon" in hparams:
    hparams["use_supTcon"] = hparams["use_supCon"]
    del hparams["use_supCon"]
config = MONAConfig(**hparams)

##

if ON_SHERLOCK:
    sessions_dir = "/oak/stanford/projects/babelfish/magneto/"
    # TODO: bechmark SCRATCH vs LOCAL_SCRATCH ...?
    scratch_directory = os.environ["SCRATCH"]
    # scratch_directory = os.environ["LOCAL_SCRATCH"]
    # gaddy_dir = "/oak/stanford/projects/babelfish/magneto/GaddyPaper/"
    gaddy_dir = os.path.join(scratch_directory, "GaddyPaper")
    # scratch_lengths_pkl = os.path.join(
    #     scratch_directory, "2023-07-25_emg_speech_dset_lengths.pkl"
    # )
    # tmp_lengths_pkl = os.path.join("/tmp", "2023-07-25_emg_speech_dset_lengths.pkl")
    # if os.path.exists(scratch_lengths_pkl) and not os.path.exists(tmp_lengths_pkl):
    #     shutil.copy(scratch_lengths_pkl, tmp_lengths_pkl)
    t12_npz_path = os.path.join(scratch_directory, "2023-08-21_T12_dataset.npz")
    T12_dir = os.path.join(scratch_directory, "T12_data_v4")
    if len(os.sched_getaffinity(0)) > 16:
        print(
            "WARNING: if you are running more than one script, you may want to use `taskset -c 0-16` or similar"
        )
    lm_directory = "/oak/stanford/projects/babelfish/magneto/GaddyPaper/icml_lm/"
    lm_file = "/oak/stanford/projects/babelfish/magneto/GaddyPaper/pretrained_models/deepspeech/deepspeech-lm.binary"
    lm_directory = ensure_folder_on_scratch(lm_directory, os.environ["LOCAL_SCRATCH"])
else:
    # on my local machine
    sessions_dir = "/data/magneto/"
    scratch_directory = "/scratch"
    gaddy_dir = "/scratch/GaddyPaper/"
    # t12_npz_path = "/data/data/T12_data_v4/synthetic_audio/2023-08-21_T12_dataset_per_sentence_z-score.npz"
    t12_npz_path = "/data/data/T12_data_v4/synthetic_audio/2023-08-22_T12_dataset_gaussian-smoothing.npz"
    T12_dir = "/data/data/T12_data_v4/"

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
##
model, config = load_model(ckpt_path, config)

# predictions = get_emg_pred(model, val_dl)

if config.togglePhones:
    default_lexicon_file = os.path.join(lm_directory, "cmudict.txt")
else:
    default_lexicon_file = os.path.join(
        lm_directory, "lexicon_graphemes_noApostrophe.txt"
    )
##
predictions = get_emg_pred(model, val_dl)
topK = get_top_k(predictions,
    model.text_transform,
    # test_dl,
    # k=100,
    k=1,
    beam_size=150,
    # beam_size=500,
    togglePhones=config.togglePhones,
    use_lm=True,
    beam_threshold=100,
    lm_weight=2,
    cpus=8,
    lexicon_file=default_lexicon_file,
    lm_file=lm_file)
wer = calc_wer(topK['predictions'], topK['sentences'], model.text_transform)
print(f"EMG Validation WER: {wer * 100:.2f}%")
##
audio_val_pred = get_audio_pred(model, val_dl)
audio_val_topK = get_top_k(audio_val_pred,
    model.text_transform,
    # test_dl,
    # k=100,
    k=1,
    beam_size=150,
    # beam_size=500,
    togglePhones=config.togglePhones,
    use_lm=True,
    beam_threshold=100,
    lm_weight=2,
    cpus=8,
    lexicon_file=default_lexicon_file,
    lm_file=lm_file)
wer = calc_wer(audio_val_topK['predictions'], audio_val_topK['sentences'], model.text_transform)
print(f"Audio Validation WER: {wer * 100:.2f}%")

##
for batch in val_dl:
    pass
batch.keys()

##
predictions = get_audio_pred(model, dataloader)

##
raise ValueError("STOP HERE")
##
topK = get_top_k(model,
    test_dl,
    k=100,
    # k=1,
    # beam_size=150,
    beam_size=5000,
    togglePhones=config.togglePhones,
    use_lm=True,
    beam_threshold=100,
    lm_weight=2,
    cpus=8,
    lexicon_file=default_lexicon_file,
    lm_file=lm_file)
wer = calc_wer(topK['predictions'], topK['sentences'], model.text_transform)
print(f"Test WER: {wer * 100:.2f}%")


##
