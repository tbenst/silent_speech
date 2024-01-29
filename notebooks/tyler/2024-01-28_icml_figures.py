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
    load_model_from_id, get_neptune_run, string_to_np_array
import pandas as pd
from tqdm import tqdm
import altair as alt
alt.renderers.enable('mimetype')
alt.themes.enable('default')
# https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
# not sure if makes a difference since we use fp16
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
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
    #### crossCon + DTW ####
    #### supTcon ####
    890, 891, 904, 905, 897,
    # 896 # skip
    #### supTcon + DTW ####
    907, 906, 921, 922, 920,
    #### EMG + Audio ####
    871, 848, 861, 881, 926,
    # 837, 827 # drop per selection criteria
    #### EMG + Audio (no librispeech ####
    960, 961, 962, 963, 964,
    #### EMG ####
    888, 893, 944, 943, 942,
    # 863, 832, 819, 852, # issues with runs

    #### EMG (no librispeech ####
    965, 967, 968, 969, 966
    
    #### Audio-only ####
    932, 933, 946, 947, 945
    # 929, 930, 945 # missing last epoch


    ######## quest for the best ##########
    #### crossCon 256k ####
    937, 938, 939, 940, 941,
    
    #### crossCon + DTW 256k ####
    # 983, 984, 986, 987, 988
    # TODO: add these five runs
    
    #### crossCon no librispeech 256k ####
    972, 973, 974, 970, 971
    
]
run_ids = [f"GAD-{ri}" for ri in run_ids]
topk_files = {}
run_type = {}
run_hparams = {}
num_beams = 150
num_beams = 5000
for ri in run_ids:
    run = get_neptune_run(ri, project="neuro/gaddy")
    output_directory = nep_get(run, "output_directory")
    run_hparams[ri] = nep_get(run, "training/hyperparams")
    # path = os.path.join(output_directory, f"2024-01-28_top100_{num_beams}beams.npz")
    path = os.path.join(output_directory, f"2024-01-27_top100_{num_beams}beams.npz")
    if os.path.exists(path):
        topk_files[ri] = path
        
print(f"{len(run_ids)=}, {len(topk_files)=}, {len(run_hparams)=}")
##
def get_run_type(hparams):
    m256 = hparams["max_len"] == 256000
    d = hparams["use_dtw"]
    c = hparams["use_crossCon"]
    if "use_supCon" in hparams:
        s = hparams["use_supCon"]
    elif "use_supTcon" in hparams:
        s = hparams["use_supTcon"]
    else:
        raise ValueError("unknown run type")
    a = hparams["audio_lambda"] == 1
    if "emg_lambda" in hparams:
        e = hparams["emg_lambda"] == 1
    else:
        e = True
    b = string_to_np_array(hparams["batch_class_proportions"])
    # use librispeech
    l = b[2] > 0
    if m256 and c and l:
        return "crossCon 256k"
    elif m256 and c and not l:
        return "crossCon no librispeech 256k"
    elif d and c and s:
        return "crossCon + supTcon + DTW"
    elif c and s:
        return "crossCon + supTcon"
    elif c:
        return "crossCon"
    elif s and d:
        return "supTcon + DTW"
    elif s:
        return "supTcon"
    elif a and e and l:
        return "EMG & Audio"
    elif a and e:
        return "EMG & Audio (no Librispeech)"
    elif e and not l:
        return "EMG (no Librispeech)"
    elif e:
        return "EMG"
    elif a:
        return "Audio"
    else:
        raise ValueError(f"unknown run type for {hparams}")
    
run_type = {}
type_count = defaultdict(int)
for ri, hparams in run_hparams.items():
    try:
        run_type[ri] = get_run_type(hparams)
        type_count[run_type[ri]] += 1
    except Exception as e:
        print(f"{ri}: unknown")
        raise e
assert len(type_count) == 12, f"{len(type_count)=}"

missing_msg = False
for t, c in type_count.items():
    if c != 5:
        if not missing_msg:
            print("==== missing runs for: ====")
            missing_msg = True
        print(f"{t}: {5-c}")
        # missing EMG (no Librispeech) 966
        # Audio 
##
missing_msg = False
i = 0
for ri in run_ids:
    rt = run_type[ri]
    try:
        f = topk_files[ri]
        i += 1
    except KeyError:
        if not missing_msg:
            print("==== missing topK files for: ====")
            missing_msg = True
        print(f"{ri}: {rt}")
        continue
print(f"have {i} topK files, missing {len(run_ids) - i}")
##
# TODO: add silent EMG after jobs finish
np.unique(topk['dataset'])
##
# to_analyze = ['audio_val', 'emg_silent_val', 'emg_vocal_val', 'librispeech_val']
to_analyze = ['audio_val', 'emg_val', 'librispeech_val']
for ri, hparams in run_hparams.items():
    assert hparams["togglePhones"] == False
text_transform = TextTransform(togglePhones=hparams['togglePhones'])
# to_analyze = ['audio_test', 'emg_test', 'librispeech_test']
df_rows = []
for ri, f in tqdm(topk_files.items()):
    rt = run_type[ri]
    topk = np.load(f, allow_pickle=True)
    dset = topk['dataset'][:]
    for task in to_analyze:
        idx = np.where(dset == task)[0]
        preds = topk["predictions"][idx]
        labels = topk["sentences"][idx]
        non_zero = np.where(labels != "")[0]
        preds = np.array([p[0] for p in preds[non_zero]])
        labels = labels[non_zero]
        wer = calc_wer(preds, labels, text_transform)
        df_rows.append({
            "run_id": ri,
            "run_type": rt,
            "task": task,
            "wer": wer
        })
df_final_wer = pd.DataFrame.from_records(df_rows)
df_final_wer
##
row_to_remove = (df_final_wer['run_type'] == "Audio") & (df_final_wer['task'].isin(["emg_val"]))
df_final_wer = df_final_wer[~row_to_remove]
row_to_remove = (df_final_wer['run_type'].isin(["EMG", "EMG (no Librispeech)"])) & (df_final_wer['task'].isin(["audio_val", "librispeech_val"]))
df_final_wer = df_final_wer[~row_to_remove]
row_to_remove = (df_final_wer['run_type'].isin(["EMG & Audio (no Librispeech)"])) & (df_final_wer['task'].isin(["librispeech_val"]))
df_final_wer = df_final_wer[~row_to_remove]


##
category_order = [
    'Audio', 
    'EMG (no Librispeech)', 
    'EMG', 
    'EMG & Audio (no Librispeech)', 
    'EMG & Audio', 
    'supTcon', 
    'supTcon + DTW', 
    'crossCon', 
    'crossCon + supTcon', 
    'crossCon + supTcon + DTW', 
    'crossCon 256k', 
    'crossCon no librispeech 256k'
]
task_order = [
    'librispeech_val',
    'audio_val', 
    'emg_val', 
]

task_labels = {
    "audio_val": "Gaddy Audio",
    "librispeech_val": "Librispeech",
    "emg_val": "Gaddy Silent EMG"
}

def create_chart(task, df):
    df_task = df[df['task'] == task]
    xlabel = "model" if task == "emg_val" else None
    return alt.Chart(df_task).mark_circle(size=20).encode(
        x=alt.X('run_type:N', axis=alt.Axis(labelAngle=-45),
            sort=category_order, title=xlabel),
        y=alt.Y('wer:Q', title='word error rate (WER)'),
        xOffset="jitter:Q",
        color=alt.Color('run_type:N', legend=None),
        tooltip=['run_id', 'run_type', 'task', 'wer']
    ).transform_calculate(
        # jitter='random()'
        jitter="sqrt(-2*log(random()))*cos(2*PI*random())"
    ).properties(
        width=600,
        height=200,
        title=task_labels[task],
    )

# Concatenate charts vertically
chart = alt.vconcat(*[create_chart(task, df_final_wer) for task in task_order])
chart.save("../../plots/val-wer_5000beams.png")
chart.save("../../plots/val-wer_5000beams.svg")
chart
##
wer_run_ids = [
    871, 848, 861, 881, 926, # EMG & Audio
    835, 841, 818, 868, 936, # crossCon
    965, 966, 967, 968, 969 # EMG (no librispeech)
]
wer_run_ids = [f"GAD-{ri}" for ri in wer_run_ids]
df_wer = []
for run_id in wer_run_ids:
    run = get_neptune_run(run_id, project="neuro/gaddy")
    wer = nep_get(run, "training/val/wer").value
    epoch = nep_get(run, "training/epoch").value
    hparams = nep_get(run, "training/hyperparams")
    df = wer.to_frame(name="wer")
    df["run_id"] = run_id
    df["type"] = run_type[run_id]
    df_wer.append(df)
df_wer = pd.concat(df_wer)
df_wer

##
