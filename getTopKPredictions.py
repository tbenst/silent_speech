'''
Using pretrained model and a torch-compatible language model, generate top k candidates and save data.
'''


import os
import sys
import numpy as np
import logging
import subprocess
import jiwer
import random

import torch
from torch import nn
import torch.nn.functional as F
from torchaudio.models.decoder import ctc_decoder

from read_emg import EMGDataset, SizeAwareSampler, PreprocessedEMGDataset, PreprocessedSizeAwareSampler
from architecture import Model, S4Model, H3Model
from data_utils import combine_fixed_length, decollate_tensor
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
flags.DEFINE_string('evaluate_saved', None, 'run evaluation on given model file')
flags.DEFINE_string('lm_directory', '/oak/stanford/projects/babelfish/magneto/GaddyPaper/pretrained_models/librispeech_lm/', 
                    'Path to KenLM language model')
flags.DEFINE_string('base_dir', '/oak/stanford/projects/babelfish/magneto/GaddyPaper/processed_data/',
                    'path to processed EMG dataset')


togglePhones = False


def getTopK(model, testset, device, k = 100):
    model.eval()

    blank_id = len(testset.text_transform.chars)
    
    if testset.togglePhones:
        lexicon_file = 'cmudict.txt'
    else:
        lexicon_file = os.path.join(FLAGS.lm_directory, 'lexicon_graphemes_noApostrophe.txt')

    decoder = ctc_decoder(
       lexicon = lexicon_file,
       tokens  = testset.text_transform.chars + ['_'],
       lm      = os.path.join(FLAGS.lm_directory, '4gram_lm.bin'),
       blank_token = '_',
       sil_token   = '|',
       nbest       = k,
       lm_weight   = 2, # default is 2; Gaddy sets to 1.85
       #word_score  = -3,
       #sil_score   = -2,
       beam_size   = 150  # SET TO 150 during inference
    )

    dataloader  = torch.utils.data.DataLoader(testset, batch_size=1, collate_fn=testset.collate_raw)
    references  = []
    predictions = []
    topk_dict = {
        'k'          : k,
        'sentences'  : [],
        'predictions': [],
        'beam_scores': [],
    }
    
    with torch.no_grad():
        for example in dataloader:
            X     = example['emg'][0].unsqueeze(0).to(device)
            X_raw = example['raw_emg'][0].unsqueeze(0).to(device)
            sess  = example['session_ids'][0].to(device)

            pred  = F.log_softmax(model(X, X_raw, sess), -1).cpu()

            beam_results = decoder(pred)
            #pred_int     = beam_results[0][0].tokens
            #pred_text    = ' '.join(beam_results[0][0].words).strip().lower()
            
            trl_top_k       = list()
            trl_beam_scores = list()
            for i in range(len(beam_results[0])):
                transcript = " ".join(beam_results[0][i].words).strip().lower()
                score      = beam_results[0][i].score
                
                trl_top_k.append(transcript)
                trl_beam_scores.append(score)
               
            # Filter out silences
            target_sentence = testset.text_transform.clean_2(example['text'][0][0])
            if len(target_sentence) > 0:
                topk_dict['predictions'].append(trl_top_k)
                topk_dict['beam_scores'].append(trl_beam_scores)
                topk_dict['sentences'].append(target_sentence)       

    return topk_dict



def evaluate_saved():
    device  = 'cuda' if torch.cuda.is_available() and not FLAGS.debug else 'cpu'    
    testset = PreprocessedEMGDataset(base_dir = FLAGS.base_dir, train = False, dev = True, test = False,
                                    togglePhones = togglePhones)
    n_chars = len(testset.text_transform.chars)
    
    if FLAGS.S4:
        model = S4Model(testset.num_features, n_chars+1).to(device)
    else:
        model = Model(testset.num_features, n_chars+1).to(device)
    
    model.load_state_dict(torch.load(FLAGS.evaluate_saved))
    topk_dict  = getTopK(model, testset, device)
    save_fname = os.path.join(FLAGS.evaluate_saved.split('model.pt')[0], 'topk_data.mat')
    
    savemat(save_fname, topk_dict)
    print('Predictions saved to:',  save_fname)

if __name__ == '__main__':
    FLAGS(sys.argv)
    evaluate_saved()

