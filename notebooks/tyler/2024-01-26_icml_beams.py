# given a Neptune run-ids, run beam search over val & test predictions
# requires that 2024-01-26_icml_pred.py was already run
##
2
##
# %load_ext autoreload
# %autoreload 2
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
    load_model_from_id, get_neptune_run, nep_get, run_beam_search
from functional import ParallelTqdm
from joblib import delayed

def flatten(l):
    return [item for sublist in l for item in sublist]

def getTopK(
        predictions, text_transform, lm_directory,
        k: int = 100,
        beam_size: int = 500,
        togglePhones: bool = False,
        use_lm: bool = True,
        beam_threshold: int = 100,
        lm_weight: float = 2,
        cpus: int = 8,
        lexicon_file: str = None,
        lm_file: str = None):

    # Define the function to be used with concurrent.futures
    func = partial(run_beam_search, text_transform=text_transform,
                   k=k, lm_weight=lm_weight, beam_size=beam_size,
                   beam_threshold=beam_threshold, use_lm=use_lm, togglePhones=togglePhones,
                   lexicon_file=lexicon_file, lm_file=lm_file)

    # If cpus=0, run without multiprocessing
    if cpus == 0:
        beam_results = [func(pred) for pred in tqdm(predictions)]
    else:
        beam_results = ParallelTqdm(n_jobs=cpus,total_tasks=len(predictions))(
            delayed(func)(pred) for pred in predictions)

    # flatten batched tuples of (all_trl_top_k, all_trl_beam_scores, all_sentences)
    # Separate and flatten the results
    all_trl_top_k, all_trl_beam_scores, all_sentences = [], [], []
    for trl_top_k, trl_beam_scores, sentences in beam_results:
        all_trl_top_k.extend(trl_top_k)
        all_trl_beam_scores.extend(trl_beam_scores)
        all_sentences.extend(sentences)

    # Collecting results
    topk_dict = {
        'k': k,
        'beam_size': beam_size,
        'beam_threshold': beam_threshold,
        'sentences': np.array(all_sentences),
        'predictions': np.array(all_trl_top_k, dtype=object), # ragged array
        'beam_scores': np.array(all_trl_beam_scores, dtype=object),
    }

    return topk_dict
##
app = typer.Typer(pretty_exceptions_show_locals=False)

@app.command()
def main(
        run_id: str = typer.Option(..., help='run evaluation on given model run id'),
        k: int = typer.Option(100, help='max beams to return'),
        beam_size: int = typer.Option(5000, help='maximum number of beams to search'),
        lm_weight: float = typer.Option(2., help='language model weight'),
        beam_threshold: int = typer.Option(75, help='prune beam search if more than this away from best score'),
        use_lm: bool = typer.Option(True, help='whether to use a language model'),
        cpus: int = typer.Option(32, help='Number of CPUs to use for beam search'),
):
    assert ON_SHERLOCK
    run = get_neptune_run(run_id, project="neuro/Gaddy")
    output_directory = nep_get(run, "output_directory")
    hparams = nep_get(run, "training/hyperparams")
    togglePhones = hparams["togglePhones"]
    text_transform = TextTransform(togglePhones = togglePhones)

    lm_directory = "/oak/stanford/projects/babelfish/magneto/GaddyPaper/icml_lm/"
    lm_directory = ensure_folder_on_scratch(lm_directory, os.environ["LOCAL_SCRATCH"])

    if togglePhones:
        lexicon_file = os.path.join(lm_directory, 'cmudict.txt')
    else:
        lexicon_file = os.path.join(lm_directory, 'lexicon_graphemes_noApostrophe.txt')
        
    lm_file = os.path.join(lm_directory, "lm.binary")
    assert os.path.exists(lm_file)
    assert os.path.exists(lexicon_file)

    path = os.path.join(output_directory, "2024-01-26_predictions.pkl")
    with open(path, "rb") as f:
        predictions = pickle.load(f)
    emg_val_pred = predictions["emg_val_pred"]
    emg_test_pred = predictions["emg_test_pred"]
    audio_val_pred = predictions["audio_val_pred"]
    audio_test_pred = predictions["audio_test_pred"]

    N_emg_val = sum([len(x[1]) for x in emg_val_pred])
    N_emg_test = sum([len(x[1]) for x in emg_test_pred])
    N_audio_val = sum([len(x[1]) for x in audio_val_pred])
    N_audio_test = sum([len(x[1]) for x in audio_test_pred])
    N = N_emg_val + N_emg_test + N_audio_val + N_audio_test

    all_pred = emg_val_pred + emg_test_pred + audio_val_pred + audio_test_pred
    dataset = ["emg_val"] * N_emg_val + ["emg_test"] * N_emg_test + \
            ["audio_val"] * N_audio_val + ["audio_test"] * N_audio_test
    topk_dict  = getTopK(all_pred, text_transform, lm_directory,
        k=k, beam_size=beam_size, beam_threshold=beam_threshold, use_lm=use_lm,
        # cpus=cpus,
        cpus=8,
        lexicon_file=lexicon_file, lm_file=lm_file, lm_weight=lm_weight) 
    topk_dict['dataset'] = np.array(dataset)
    save_fname = os.path.join(output_directory, 'topK_beams.npz')
    np.savez(save_fname, **topk_dict)
    print(f'Predictions saved to:\n{save_fname}')

if __name__ == '__main__':
    app()
##
exit(0)
##
run_id = "GAD-823"
k = 3
beam_size = 50
lm_weight = 2.
beam_threshold = 75
use_lm = True
cpus = 8

##
assert ON_SHERLOCK
run = get_neptune_run(run_id, project="neuro/Gaddy")
output_directory = nep_get(run, "output_directory")
hparams = nep_get(run, "training/hyperparams")
togglePhones = hparams["togglePhones"]
text_transform = TextTransform(togglePhones = togglePhones)

lm_directory = "/oak/stanford/projects/babelfish/magneto/GaddyPaper/icml_lm/"
lm_directory = ensure_folder_on_scratch(lm_directory, os.environ["LOCAL_SCRATCH"])

if togglePhones:
    lexicon_file = os.path.join(lm_directory, 'cmudict.txt')
else:
    lexicon_file = os.path.join(lm_directory, 'lexicon_graphemes_noApostrophe.txt')
    
lm_file = os.path.join(lm_directory, "lm.binary")
assert os.path.exists(lm_file)
assert os.path.exists(lexicon_file)

path = os.path.join(output_directory, "2024-01-26_predictions.pkl")
with open(path, "rb") as f:
    predictions = pickle.load(f)
emg_val_pred = predictions["emg_val_pred"]
emg_test_pred = predictions["emg_test_pred"]
audio_val_pred = predictions["audio_val_pred"]
audio_test_pred = predictions["audio_test_pred"]

N_emg_val = sum([len(x[1]) for x in emg_val_pred])
N_emg_test = sum([len(x[1]) for x in emg_test_pred])
N_audio_val = sum([len(x[1]) for x in audio_val_pred])
N_audio_test = sum([len(x[1]) for x in audio_test_pred])
N = N_emg_val + N_emg_test + N_audio_val + N_audio_test

all_pred = emg_val_pred + emg_test_pred + audio_val_pred + audio_test_pred
dataset = ["emg_val"] * N_emg_val + ["emg_test"] * N_emg_test + \
        ["audio_val"] * N_audio_val + ["audio_test"] * N_audio_test
topk_dict  = getTopK(all_pred, text_transform, lm_directory,
    k=k, beam_size=beam_size, beam_threshold=beam_threshold, use_lm=use_lm,
    # cpus=cpus,
    cpus=8,
    lexicon_file=lexicon_file, lm_file=lm_file, lm_weight=lm_weight) 
topk_dict['dataset'] = np.array(dataset)
save_fname = os.path.join(output_directory, 'topK_beams.npz')
np.savez(save_fname, **topk_dict)
print(f'Predictions saved to:\n{save_fname}')
##
