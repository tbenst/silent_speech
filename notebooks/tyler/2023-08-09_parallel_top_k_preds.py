'''
Using pretrained model and a torch-compatible language model, generate top k candidates and save data.

Example:
python 2023-08-09_parallel_top_k_preds.py --checkpoint /scratch/users/tbenst/2023-08-01T06:54:28.359594_gaddy/SpeechOrEMGToText-epoch=199-val/wer=0.264.ckpt --beam-size 5000 --no-use-lm
'''
import os
import sys
# horrible hack to get around this repo not being a proper python package
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.getcwd()))
# SCRIPT_DIR = "/home/tyler/code/silent_speech/"``
sys.path.append(SCRIPT_DIR)

import numpy as np
import logging
import subprocess
import jiwer
import random

import torch
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
from torchaudio.models.decoder import ctc_decoder

from read_emg import EMGDataModule
from architecture import SpeechOrEMGToText, SpeechOrEMGToTextConfig
from dataloaders import EMGAndSpeechModule, collate_gaddy_or_speech
from data_utils import combine_fixed_length, decollate_tensor, TextTransform
from transformer import TransformerEncoderLayer
import neptune.new as neptune
import multiprocessing
from functools import partial
from concurrent.futures import ProcessPoolExecutor

import typer
import os
import sys

hostname = subprocess.run("hostname", capture_output=True)
ON_SHERLOCK = hostname.stdout[:2] == b"sh"
if ON_SHERLOCK:
    # TODO: bechmark SCRATCH vs LOCAL_SCRATCH ...?
    # scratch_directory = os.environ["LOCAL_SCRATCH"]
    gaddy_dir = '/oak/stanford/projects/babelfish/magneto/GaddyPaper/'
else:
    # on my local machine
    gaddy_dir = '/scratch/GaddyPaper/'

data_dir = os.path.join(gaddy_dir, 'processed_data/')
lm_directory = os.path.join(gaddy_dir, 'pretrained_models/librispeech_lm/')

togglePhones = False

if togglePhones:
    default_lexicon_file = os.path.join(lm_directory, 'cmudict.txt')
else:
    default_lexicon_file = os.path.join(lm_directory, 'lexicon_graphemes_noApostrophe.txt')

app = typer.Typer(pretty_exceptions_show_locals=False)

# Function to get the predictions
def get_pred(model, dataloader):
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            X = nn.utils.rnn.pad_sequence(batch['raw_emg'], batch_first=True)
            X = X.cuda()
            pred = model.emg_forward(X)[0].cpu()
            predictions.append((batch, pred))

    return predictions

# Function to run the beam search
def run_beam_search(batch_pred, text_transform, lm_directory, k, lm_weight, beam_size, beam_threshold, use_lm, togglePhones, lexicon_file, lm_file):
    batch, pred = batch_pred
    
    if use_lm:
        lm = lm_file
    else:
        lm = None

    decoder = ctc_decoder(
        lexicon=lexicon_file,
        tokens=text_transform.chars + ['_'],
        lm=lm,
        blank_token='_',
        sil_token='|',
        nbest=k,
        lm_weight=lm_weight,
        beam_size=beam_size,
        beam_threshold=beam_threshold
    )

    beam_results = decoder(pred)
    all_trl_top_k = []
    all_trl_beam_scores = []
    all_sentences = []
    for i, (example, beam_result) in enumerate(zip(batch, beam_results)):
        # Filter out silences
        target_sentence = text_transform.clean_2(batch['text'][i])
        if len(target_sentence) > 0:
            trl_top_k = []
            trl_beam_scores = []
            for beam in beam_result:
                transcript = " ".join(beam.words).strip().lower()
                score = beam.score
                trl_top_k.append(transcript)
                trl_beam_scores.append(score)
                
            all_trl_top_k.append(np.array(trl_top_k))
            all_trl_beam_scores.append(np.array(trl_beam_scores))
            all_sentences.append(target_sentence)      

    return all_trl_top_k, all_trl_beam_scores, all_sentences


def getTopK(
        model, dataloader, text_transform, lm_directory,
        k: int = 100,
        beam_size: int = 500,
        togglePhones: bool = False,
        use_lm: bool = True,
        beam_threshold: int = 100,
        lm_weight: float = 2,
        cpus: int = 8,
        lexicon_file: str = None,
        lm_file: str = None):
    predictions = get_pred(model, dataloader)

    # Define the function to be used with concurrent.futures
    func = partial(run_beam_search, text_transform=text_transform, lm_directory=lm_directory,
                   k=k, lm_weight=lm_weight, beam_size=beam_size,
                   beam_threshold=beam_threshold, use_lm=use_lm, togglePhones=togglePhones,
                   lexicon_file=lexicon_file, lm_file=lm_file)

    # If cpus=0, run without multiprocessing
    if cpus == 0:
        beam_results = [func(pred) for pred in tqdm(predictions)]
    else:
        # Use concurrent.futures for running the beam search with a progress bar
        with ProcessPoolExecutor(max_workers=cpus) as executor:
            beam_results = list(tqdm(executor.map(func, predictions), total=len(predictions)))

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
        'predictions': np.array(all_trl_top_k),
        'beam_scores': np.array(all_trl_beam_scores),
    }

    return topk_dict

def evaluate_saved(
        k: int,
        beam_size: int,
        lm_weight: float,
        beam_threshold: int,
        checkpoint_file: str,
        use_lm: bool,
        cpus: int,
        lexicon_file: str,
        lm_file: str,
):
    normalizers_file = os.path.join(SCRIPT_DIR, "normalizers.pkl")
    
    togglePhones = False
    text_transform = TextTransform(togglePhones = togglePhones)
    val_bz = 12
    max_len = 48000
    emg_datamodule = EMGDataModule(data_dir, togglePhones, normalizers_file, max_len=max_len,
        collate_fn=collate_gaddy_or_speech,
        pin_memory=True, batch_size=val_bz)
    
    testset = emg_datamodule.val_dataloader()
    
    checkpoint = torch.load(checkpoint_file)
    steps_per_epoch = 200 
    n_chars = len(text_transform.chars)
    num_outs = n_chars + 1 # +1 for CTC blank token ( i think? )
    precision = "16-mixed"
    config = SpeechOrEMGToTextConfig(steps_per_epoch, lm_directory, num_outs, precision=precision)
    model = SpeechOrEMGToText(config, text_transform)
    model.load_state_dict(checkpoint["state_dict"])
    model.cuda()
    topk_dict  = getTopK(model, testset, text_transform, lm_directory,
        k=k, beam_size=beam_size, beam_threshold=beam_threshold, use_lm=use_lm, cpus=cpus,
        lexicon_file=lexicon_file, lm_file=lm_file) 
    name = f"top{k}_{beam_size}beams_thresh{beam_threshold}_lmweight{lm_weight}"
    if use_lm:
        lm_name = os.path.splitext(os.path.basename(lm_file))[0]  # Extract the LM name
        name += f"_LM-{lm_name}"
    else:
        name += "_noLM"
    save_fname = os.path.join(os.path.split(checkpoint_file)[0], f'{name}.npz')
    
    np.savez(save_fname, **topk_dict)
    print(f'Predictions saved to:\n{save_fname}')

@app.command()
def main(
        k: int = typer.Option(100, help='max beams to return'),
        beam_size: int = typer.Option(500, help='maximum number of beams to search'),
        lm_weight: float = typer.Option(2., help='language model weight'),
        beam_threshold: int = typer.Option(75, help='prune beam search if more than this away from best score'),
        checkpoint: str = typer.Option(..., help='run evaluation on given model file'),
        use_lm: bool = typer.Option(True, help='whether to use a language model'),
        cpus: int = typer.Option(8, help='Number of CPUs to use for beam search'),
        lexicon_file: str = typer.Option(default_lexicon_file, help='Path to the lexicon file'),
        lm: str = typer.Option(os.path.join(lm_directory, '4gram_lm.bin'), help='Path to the language model file'),
):
    evaluate_saved(k, beam_size, lm_weight, beam_threshold, checkpoint, use_lm, cpus, lexicon_file, lm)


if __name__ == '__main__':
    app()
