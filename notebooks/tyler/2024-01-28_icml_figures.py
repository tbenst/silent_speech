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
    load_model_from_id, get_neptune_run, string_to_np_array, get_run_type
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

    #### EMG (no librispeech) ####
    965, 967, 968, 969, 966, # TODO: consider dropping
    
    #### Audio-only ####
    932, 933, 946, 947, 945,
    # 929, 930, 945 
    # # missing last epoch
    # TODO: can use finished-training_epoch=200.ckpt
    # need to fall back to `finished-training_epoch=200.ckpt`
    # (or default to) if choosing `last`


    ######## quest for the best ##########
    #### crossCon 256k ####
    937, 938, 939, 940, 941,
    
    #### crossCon + DTW 256k ####
    983, 984, 986, 987, 988,
    
    #### crossCon no librispeech 256k ####
    972, 973, 974, 970, 971,
    
    #### crossCon balanced 256k ####
    957, 958, 989, 990, 991,
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
    path = os.path.join(output_directory, f"2024-01-28_top100_{num_beams}beams.npz")
    # path = os.path.join(output_directory, f"2024-01-27_top100_{num_beams}beams.npz")
    if os.path.exists(path):
        topk_files[ri] = path
        
print(f"{len(run_ids)=}, {len(topk_files)=}, {len(run_hparams)=}")
##
   
run_type = {}
type_count = defaultdict(int)
for ri, hparams in run_hparams.items():
    try:
        run_type[ri] = get_run_type(hparams)
        type_count[run_type[ri]] += 1
    except Exception as e:
        print(f"{ri}: unknown")
        raise e
assert len(type_count) == 14, f"{len(type_count)=}"

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
# order used for plotting order too
to_analyze = [
    'librispeech_val',
    'audio_val',
    'emg_vocal_val',
    'emg_silent_val',
]
# to_analyze = ['audio_val', 'emg_val', 'librispeech_val']
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
            "model": rt,
            "task": task,
            "wer": wer
        })
df_final_wer = pd.DataFrame.from_records(df_rows)
df_final_wer
##
row_to_remove = (df_final_wer['model'] == "Audio") & \
    (df_final_wer['task'].isin(["emg_vocal_val", "emg_silent_val"]))
df_final_wer = df_final_wer[~row_to_remove]
row_to_remove = (df_final_wer['model'].isin(["EMG (256k)",
                                                "EMG (no Librispeech)"])) & \
    (df_final_wer['task'].isin(["audio_val", "librispeech_val"]))
df_final_wer = df_final_wer[~row_to_remove]
row_to_remove =(df_final_wer['model'].isin(["EMG & Audio (no Librispeech)",
                                       'crossCon (no Librispeech) 256k'])) & \
    (df_final_wer['task'].isin(["librispeech_val"]))
df_final_wer = df_final_wer[~row_to_remove]

# no interesting difference from EMG
row_to_remove = (df_final_wer['model'] == "EMG (no Librispeech)")
df_final_wer = df_final_wer[~row_to_remove]

##
category_order = [
    'Audio', 
    # 'EMG (no Librispeech)', 
    'EMG', 
    'EMG & Audio (no Librispeech)', 
    'EMG & Audio', 
    'supTcon', 
    'supTcon + DTW', 
    'crossCon', 
    'crossCon + supTcon', 
    'crossCon + supTcon + DTW', 
    'crossCon + DTW 256k',
    'crossCon 256k',
    'crossCon (balanced) 256k',
    'crossCon (no Librispeech) 256k'
]

cat_labels = {
    'EMG & Audio (no Librispeech)': \
        'EMG & Audio (-Libri)',
    'crossCon (no Librispeech) 256k': \
        'crossCon (-Libri) 256k',
    'crossCon (balanced) 256k': \
        'crossCon (bal) 256k',
    'crossCon + supTcon + DTW': \
        'crossCon+supTcon+DTW',
}

category_order = [cat_labels.get(c, c) for c in category_order]

task_labels = {
    "audio_val": "Gaddy Audio",
    "librispeech_val": "Librispeech",
    "emg_silent_val": "Gaddy Silent EMG",
    "emg_vocal_val": "Gaddy Vocal EMG"
}

# Calculate global min and max wer for consistent y-axis
global_min_wer = 0.
global_max_wer = 0.35

def create_chart(task, df):
    df_task = df[df['task'] == task]
    df_task['short_model'] = df_task['model'].apply(lambda x: cat_labels.get(x, x))
    return alt.Chart(df_task).mark_circle(size=50).encode(
        x=alt.X('short_model:N', axis=alt.Axis(labelAngle=-20, labelFontSize=16),
            sort=category_order, title=None,  scale=alt.Scale(domain=category_order)),
        y=alt.Y('wer:Q', title='word error rate (%)',
                scale=alt.Scale(domain=[global_min_wer, global_max_wer]),
                axis=alt.Axis(format='%',
                labelFontSize=16, titleFontSize=16)),
        xOffset="jitter:Q",
        color=alt.Color('short_model:N', scale = alt.Scale(scheme='category20'),
                        legend=None),
        tooltip=['run_id', 'short_model', 'task', 'wer']
    ).transform_calculate(
        # jitter='random()'
        jitter="sqrt(-2*log(random()))*cos(2*PI*random())"
    ).properties(
        width=1100,
        height=250,
        title=alt.Title(task_labels[task],
            fontSize=20)
    )

# audio_chart = alt.hconcat(*[create_chart(task, df_final_wer) for task in to_analyze[:2]])
# emg_chart = alt.hconcat(*[create_chart(task, df_final_wer) for task in to_analyze[2:]])
# chart = alt.vconcat(audio_chart, emg_chart)
# Concatenate charts vertically
chart = alt.vconcat(*[create_chart(task, df_final_wer) for task in to_analyze])
chart.save(f"../../plots/val-wer_{num_beams}beams.png", scale_factor=2.0)
chart.save(f"../../plots/val-wer_{num_beams}beams.svg")
chart
##
wer_run_ids = [
    871, 848, 861, 881, 926, # EMG & Audio
    # 835, 841, 818, 868, 936, # crossCon
    937, 938, 939, 940, 941,
    # 965, 967, 968, 969 # EMG (no librispeech)
    888, 893, 944, 943, 942, # EMG
    # 966 # TODO add when finished
]
wer_run_ids = [f"GAD-{ri}" for ri in wer_run_ids]
df_wer = []
for run_id in wer_run_ids:
    run = get_neptune_run(run_id, project="neuro/gaddy")
    wer = nep_get(run, "training/val/wer").value
    hparams = nep_get(run, "training/hyperparams")
    df = wer.to_frame(name="wer")
    df["epoch"] = df.index
    df["run_id"] = run_id
    df["model"] = run_type[run_id]
    df_wer.append(df)
df_wer = pd.concat(df_wer)
df_wer
df_wer.loc[df_wer["model"] == "EMG (no Librispeech)", "model"] = "EMG (256k)" 
df_wer = df_wer[df_wer['epoch'] <= 199]
##
# Creating mean and standard deviation dataframes
mean_df = df_wer.groupby(['epoch', 'model']).median().reset_index()
std_df = df_wer.groupby(['epoch', 'model']).std().reset_index()

# Define common encoding for the x-axis
xaxis = alt.X('epoch:Q', axis=alt.Axis(title='epoch', titleFontSize=16, labelFontSize=16))

# Define common encoding for the y-axis
yaxis = alt.Y('wer:Q', title='word error rate', axis=alt.Axis(titleFontSize=16, labelFontSize=16))

# Define common color encoding
color_scale = alt.Color('model:N',
                        scale=alt.Scale(domain=["EMG 256k", "EMG & Audio", "crossCon 256k"]),
                        legend=alt.Legend(title='model', titleFontSize=16, labelFontSize=16))

# Create a line chart for the mean
val_wer_chart = alt.Chart(mean_df).mark_line().encode(
    x=xaxis,
    y=yaxis,
    color=color_scale,
    tooltip=['epoch', 'model', 'wer']
)

# Save and display the chart
val_wer_chart.save(f"../../plots/mean-wer-by-epoch.png", scale_factor=2.0)
val_wer_chart.save(f"../../plots/mean-wer-by-epoch.svg")
val_wer_chart
##
########## CTC loss charts ##########
ctc_run_ids = [
    871, 848, 861, 881, 926, # EMG & Audio
    # 835, 841, 818, 868, 936, # crossCon 128k
    937, 938, 939, 940, 941,
    # 965, 967, 968, 969 # EMG (no librispeech)
    888, 893, 944, 943, 942, # EMG
    # 966 # TODO add when finished
]
ctc_run_ids = [f"GAD-{ri}" for ri in ctc_run_ids]
df_ctc = []
for run_id in ctc_run_ids:
    run = get_neptune_run(run_id, project="neuro/gaddy")
    ctc = nep_get(run, "training/val/emg_ctc_loss").value
    hparams = nep_get(run, "training/hyperparams")
    df = ctc.to_frame(name="ctc")
    df["epoch"] = df.index
    df["run_id"] = run_id
    df["model"] = run_type[run_id]
    df_ctc.append(df)
df_ctc = pd.concat(df_ctc)
df_ctc
df_ctc.loc[df_ctc["model"] == "EMG (no Librispeech)", "model"] = "EMG (256k)" 
df_ctc = df_ctc[df_ctc['epoch'] <= 199]
##
# Creating mean and standard deviation dataframes
mean_df = df_ctc.groupby(['epoch', 'model']).median().reset_index()
std_df = df_ctc.groupby(['epoch', 'model']).std().reset_index()

# Define common encoding for the x-axis
xaxis = alt.X('epoch:Q', axis=alt.Axis(title='epoch', titleFontSize=16, labelFontSize=16))

# Define common encoding for the y-axis
yaxis = alt.Y('ctc:Q', title='CTC loss', axis=alt.Axis(titleFontSize=16, labelFontSize=16))

# Define common color encoding
color_scale = alt.Color('model:N', legend=alt.Legend(title='model', titleFontSize=16, labelFontSize=16))

# Create a line chart for the mean
ctc_line = alt.Chart(mean_df).mark_line().encode(
    x=xaxis,
    y=yaxis,
    color=color_scale,
    tooltip=['epoch', 'model', 'ctc']
)

# Save and display the chart
ctc_line.save(f"../../plots/mean-ctc-by-epoch.png", scale_factor=2.0)
ctc_line.save(f"../../plots/mean-ctc-by-epoch.svg")
ctc_line


##
wer_ctc_chart = alt.hconcat(val_wer_chart, ctc_line)

# Save and display the chart
wer_ctc_chart.save(f"../../plots/wer_ctc-by-epoch.png", scale_factor=2.0)
wer_ctc_chart.save(f"../../plots/wer_ctc-by-epoch.svg")
wer_ctc_chart
##
df_final_wer[np.logical_and(df_final_wer.model == "EMG 256k",
                            df_final_wer.task == "emg_silent_val") ].wer.mean()
##
df_final_wer[np.logical_and(df_final_wer.model == "EMG 256k",
                            df_final_wer.task == "emg_vocal_val") ].wer.mean()
##
df_final_wer[np.logical_and(df_final_wer.model == "EMG & Audio (no Librispeech)",
                            df_final_wer.task == "emg_silent_val") ].wer.mean()

##
df_final_wer[np.logical_and(df_final_wer.model == "EMG & Audio (no Librispeech)",
                            df_final_wer.task == "emg_silent_val") ].wer.mean()
##
df_final_wer[np.logical_and(df_final_wer.model == "EMG & Audio",
                            df_final_wer.task == "emg_silent_val") ].wer.mean()

##
df_final_wer[np.logical_and(df_final_wer.model == "crossCon",
                            df_final_wer.task == "emg_silent_val") ].wer.mean()
##
df_final_wer[np.logical_and(df_final_wer.model == "EMG & Audio",
                            df_final_wer.task == "emg_vocal_val") ].wer.mean()
##
df_final_wer[np.logical_and(df_final_wer.model == "crossCon",
                            df_final_wer.task == "emg_vocal_val") ].wer.mean()
##
df_final_wer[np.logical_and(df_final_wer.model == "crossCon 256k",
                            df_final_wer.task == "emg_silent_val") ].wer.mean()
##
df_final_wer[np.logical_and(df_final_wer.model == "crossCon + DTW 256k",
                            df_final_wer.task == "emg_silent_val") ].wer.mean()

##
# we sort by val silent EMG WER.
ensembleB = np.array([
    'GAD-984', # crossCon + DTW (256k)
    'GAD-992', # crossCon + DTW (256k)
    'GAD-986', # crossCon + DTW (256k)
    'GAD-996', # crossCon + DTW (256k)
    'GAD-993', # crossCon + DTW (256k)
    'GAD-987', # crossCon + DTW (256k)
    'GAD-988', # crossCon + DTW (256k)
    'GAD-940', # crossCon (256k)
    'GAD-995', # crossCon + DTW (256k)
    'GAD-983', # crossCon + DTW (256k) # 2024-01-31 ensemble10
])

# 2024-01-30 ensemble10
# includes 5 crossCon and 5 crossCon + DTW runs
ensembleA = np.array([
    "GAD-984",
    "GAD-986",
    "GAD-987",
    "GAD-988",
    "GAD-983",
    "GAD-940",
    "GAD-938",
    "GAD-941",
    "GAD-937",
    "GAD-939",
])

df_final_wer[np.logical_and(
    df_final_wer.run_id.isin(ensembleA),
    df_final_wer.task == "emg_silent_val"
)].mean()
##
df_final_wer[np.logical_and(
    df_final_wer.run_id.isin(ensembleB),
    df_final_wer.task == "emg_silent_val"
)].mean()
##
