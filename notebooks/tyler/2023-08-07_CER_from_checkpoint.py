'''
Using pretrained model and a torch-compatible language model, generate top k candidates and save data.
'''
##
import os
import sys
# horrible hack to get around this repo not being a proper python package
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.getcwd()))
# SCRIPT_DIR = "/home/tyler/code/silent_speech/"
sys.path.append(SCRIPT_DIR)

import numpy as np
import logging
import subprocess
import jiwer, evaluate
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
from scipy.io import savemat
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_boolean('debug', False, 'debug')
flags.DEFINE_string('output_directory', 'output', 'where to save models and outputs')
flags.DEFINE_integer('S4', 0, 'Toggle S4 model in place of transformer')
flags.DEFINE_integer('batch_size', 32, 'training batch size')
flags.DEFINE_float('learning_rate', 3e-4, 'learning rate')
flags.DEFINE_integer('epochs', 200, 'training epochs')
flags.DEFINE_integer('learning_rate_warmup', 1000, 'steps of linear warmup')
flags.DEFINE_integer('learning_rate_patience', 5, 'learning rate decay patience')
flags.DEFINE_string('start_training_from', None, 'start training from this model')
flags.DEFINE_float('l2', 0, 'weight decay')
flags.DEFINE_string('checkpoint', None, 'run evaluation on given model file')
# flags.DEFINE_string('checkpoint', '/scratch/2023-07-31T21:00:13.113729_gaddy/SpeechOrEMGToText-epoch=87-val/wer=0.273.ckpt', 'run evaluation on given model file')
flags.DEFINE_string('lm_directory', '/oak/stanford/projects/babelfish/magneto/GaddyPaper/pretrained_models/librispeech_lm/', 
                    'Path to KenLM language model')
flags.DEFINE_string('base_dir', '/oak/stanford/projects/babelfish/magneto/GaddyPaper/processed_data/',
                    'path to processed EMG dataset')

# checkpoint = '/scratch/2023-07-31T21:00:13.113729_gaddy/SpeechOrEMGToText-epoch=87-val/wer=0.273.ckpt'

togglePhones = False

from evaluate import load
cer = load("cer")
##
def get_char_preds(model, dataloader, text_transform,
            device='cuda', k = 100, togglePhones=False):
    model.eval()
    # prune beam search if more than this away from best score
    thresh = 50
    
    decoder = ctc_decoder(
       lexicon = None,
       lm = None,
       tokens  = text_transform.chars + ['_'],
       blank_token = '_',
       sil_token   = '|',
       nbest       = k,
       #sil_score   = -2,
    #    beam_size   = k+50,
    #    beam_size   = k*50,
       beam_size   = int(k*2),
    #    beam_size   = int(k*1.5),
       beam_threshold = thresh # defaults to 50
    )

    predictions = []
    references = []
        
    with torch.no_grad():
        for example in tqdm(dataloader):
            X = nn.utils.rnn.pad_sequence(example['raw_emg'], batch_first=True)
            Y = example['text']
            bz = X.shape[0]
            X = X.cuda()
            # X = batch['raw_emg'][0].unsqueeze(0) 

            logging.debug(f"calling emg_forward with {X.shape=}")
            pred  = model.emg_forward(X)[0].cpu()


            # print(f"{pred.shape=}")
            # 1:-1 to drop first and last token (beam search adds silence token)
            # chars_pred = torch.argmax(pred, dim=-1)[0][1:-1]
            # tokens = text_transform.chars + ['_']
            # chars_pred = ''.join(tokens[i] for i in chars_pred)
            # print(f"{chars_pred=}")
            # print(f"{Y=}")

            assert len(pred) == bz
            
            beam_results = decoder(pred)
            # return beam_results
            
            # use top hypothesis from beam search
            # beam search adds silence token. we rid with `.strip`
            pred_text = [text_transform.int_to_text(b[0].tokens).strip() for b in beam_results]
        
            target_text  = [text_transform.clean_2(b) for b in example['text']]
            for p,t in zip(pred_text, target_text):
                if len(t) > 0:
                    predictions.append(p)
                    references.append(t)
    # return cer.compute(predictions=predictions, references=references)
    return predictions, references

def evaluate_saved():
    hostname = subprocess.run("hostname", capture_output=True)
    ON_SHERLOCK = hostname.stdout[:2] == b"sh"
    if ON_SHERLOCK:
        sessions_dir = '/oak/stanford/projects/babelfish/magneto/'
        # TODO: bechmark SCRATCH vs LOCAL_SCRATCH ...?
        scratch_directory = os.environ["SCRATCH"]
        # scratch_directory = os.environ["LOCAL_SCRATCH"]
        gaddy_dir = '/oak/stanford/projects/babelfish/magneto/GaddyPaper/'
        scratch_lengths_pkl = os.path.join(scratch_directory, "2023-07-25_emg_speech_dset_lengths.pkl")
        tmp_lengths_pkl = os.path.join("/tmp", "2023-07-25_emg_speech_dset_lengths.pkl")
    else:
        # on my local machine
        sessions_dir = '/data/magneto/'
        scratch_directory = "/scratch"
        gaddy_dir = '/scratch/GaddyPaper/'
        
    data_dir = os.path.join(gaddy_dir, 'processed_data/')
    lm_directory = os.path.join(gaddy_dir, 'pretrained_models/librispeech_lm/')
    normalizers_file = os.path.join(SCRIPT_DIR, "normalizers.pkl")

    togglePhones = False
    text_transform = TextTransform(togglePhones = togglePhones)
    val_bz = 12
    max_len = 48000
    emg_datamodule = EMGDataModule(data_dir, togglePhones, normalizers_file, max_len=max_len,
        collate_fn=collate_gaddy_or_speech,
        pin_memory=True, batch_size=val_bz)

    # testset = datamodule.val_dataloader()
    testset = emg_datamodule.val_dataloader()

    # model = SpeechOrEMGToText.load_from_checkpoint(FLAGS.checkpoint)
    checkpoint = torch.load(FLAGS.checkpoint)
    steps_per_epoch = 200 
    n_chars = len(text_transform.chars)
    num_outs = n_chars + 1 # +1 for CTC blank token ( i think? )
    precision = "16-mixed"
    config = SpeechOrEMGToTextConfig(steps_per_epoch, lm_directory, num_outs, precision=precision)
    model = SpeechOrEMGToText(config, text_transform)
    model.load_state_dict(checkpoint["state_dict"])
    model.cuda()
    k = 100
    pred, ref = get_char_preds(model, testset, text_transform, lm_directory, k=k)
    cer_whitespace = cer.compute(predictions=pred, references=ref)
    pred_no_whitespace = [p.replace(" ", "") for p in pred]
    ref_no_whitespace = [r.replace(" ", "") for r in ref]
    cer_no_whitespace = cer.compute(predictions=pred_no_whitespace, references=ref_no_whitespace)
    print(FLAGS.checkpoint)
    print(f"CER: {cer_whitespace*100:.2f}%\nCER (no whitespace): {cer_no_whitespace*100:.2f}%")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python get_top_k_preds.py --checkpoint=PATH_TO_MODEL")
    FLAGS(sys.argv)
    evaluate_saved()