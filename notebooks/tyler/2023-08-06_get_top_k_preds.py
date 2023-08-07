'''
Using pretrained model and a torch-compatible language model, generate top k candidates and save data.
'''
import os
import sys
# horrible hack to get around this repo not being a proper python package
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.getcwd()))
SCRIPT_DIR = "/home/tyler/code/silent_speech/"
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
# flags.DEFINE_string('checkpoint', None, 'run evaluation on given model file')
flags.DEFINE_string('checkpoint', '/scratch/2023-07-31T21:00:13.113729_gaddy/SpeechOrEMGToText-epoch=87-val/wer=0.273.ckpt', 'run evaluation on given model file')
flags.DEFINE_string('lm_directory', '/oak/stanford/projects/babelfish/magneto/GaddyPaper/pretrained_models/librispeech_lm/', 
                    'Path to KenLM language model')
flags.DEFINE_string('base_dir', '/oak/stanford/projects/babelfish/magneto/GaddyPaper/processed_data/',
                    'path to processed EMG dataset')


togglePhones = False

##### k=100, beam_size=5000 (best so far!)

# len(ctc_hypotheses)=100
# len(ctc_hypotheses)=65
# len(ctc_hypotheses)=100
# len(ctc_hypotheses)=18
# len(ctc_hypotheses)=100
# len(ctc_hypotheses)=86
# len(ctc_hypotheses)=68
# len(ctc_hypotheses)=5
# len(ctc_hypotheses)=12
# len(ctc_hypotheses)=33
# len(ctc_hypotheses)=100
# len(ctc_hypotheses)=100

# "keep back said ofer"
# "keep bag said ofer"
# "keep back said over"
# "keep bag said over"
# "keep back said offer"

##### k=100, beam_size=500
# "keep back said over"
# "keep bag said over"
# "keep back sad over"

# len(ctc_hypotheses)=89
# len(ctc_hypotheses)=5
# len(ctc_hypotheses)=100
# len(ctc_hypotheses)=100
# len(ctc_hypotheses)=35
# len(ctc_hypotheses)=31
# len(ctc_hypotheses)=13
# len(ctc_hypotheses)=5
# len(ctc_hypotheses)=12
# len(ctc_hypotheses)=3
# len(ctc_hypotheses)=67
# len(ctc_hypotheses)=46


# k=100, beam_size=200
# returns one hypothesis:
# "keep back said over"


# k=100, beam_size=150
# returns 100 hypotheses that repeat same two
# "keep back said"
# "keep bag said"
# "keep bag said"
# "keep back said"
# "keep bag said"

def getTopK(model, dataloader, text_transform, lm_directory,
            device='cuda', k = 100, togglePhones=False):
    model.eval()
    
    if togglePhones:
        lexicon_file = os.path.join(lm_directory, 'cmudict.txt')
    else:
        lexicon_file = os.path.join(lm_directory, 'lexicon_graphemes_noApostrophe.txt')
    # prune beam search if more than this away from best score
    thresh = 50
    
    # len(ctc_hypotheses)=22
    # len(ctc_hypotheses)=3
    # len(ctc_hypotheses)=22
    # len(ctc_hypotheses)=90
    # len(ctc_hypotheses)=8
    # len(ctc_hypotheses)=5
    # len(ctc_hypotheses)=4
    # len(ctc_hypotheses)=5
    # len(ctc_hypotheses)=19
    # len(ctc_hypotheses)=92
    # len(ctc_hypotheses)=21
    # len(ctc_hypotheses)=16
    decoder = ctc_decoder(
       lexicon = lexicon_file,
       tokens  = text_transform.chars + ['_'],
       lm      = os.path.join(lm_directory, '4gram_lm.bin'),
       blank_token = '_',
       sil_token   = '|',
       nbest       = k,
       lm_weight   = 2, # default is 2; Gaddy sets to 1.85
       #word_score  = -3,
       #sil_score   = -2,
    #    beam_size   = k+50,
       beam_size   = k*50,
    #    beam_size   = int(k*1.5),
       beam_threshold = thresh # defaults to 50
    )

    references  = []
    predictions = []
    topk_dict = {
        'k'          : k,
        'sentences'  : [],
        'predictions': [],
        'beam_scores': [],
    }
    
    with torch.no_grad():
        for example in tqdm(dataloader):
            X = nn.utils.rnn.pad_sequence(example['raw_emg'], batch_first=True)
            bz = X.shape[0]
            X = X.cuda()
            # X = batch['raw_emg'][0].unsqueeze(0) 

            logging.debug(f"calling emg_forward with {X.shape=}")
            pred  = model.emg_forward(X)[0].cpu()
            assert len(pred) == bz
            
            # INFO: beam search does not always return k results!
            # TODO: try integrating LLaMA directly...
            # https://pytorch.org/audio/stable/tutorials/asr_inference_with_ctc_decoder_tutorial.html#custom-language-model
            # or use NEMO?
            # https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/asr_language_modeling.html
            # https://github.com/NVIDIA/NeMo/blob/stable/scripts/asr_language_modeling/neural_rescorer/eval_neural_rescorer.py
            beam_results = decoder(pred)
            
            for ctc_hypotheses in beam_results:
                print(f"{len(ctc_hypotheses)=}")

            trl_top_k       = list()
            trl_beam_scores = list()
            for i in range(len(beam_results[0])):
                transcript = " ".join(beam_results[0][i].words).strip().lower()
                score      = beam_results[0][i].score
                
                trl_top_k.append(transcript)
                trl_beam_scores.append(score)
               
            # Filter out silences
            target_sentence = text_transform.clean_2(example['text'][0][0])
            if len(target_sentence) > 0:
                topk_dict['predictions'].extend(trl_top_k)
                topk_dict['beam_scores'].extend(trl_beam_scores)
                topk_dict['sentences'].extend(target_sentence)       

    return topk_dict



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
    
    # datamodule =  EMGAndSpeechModule(emg_datamodule.train,
    #     emg_datamodule.val, emg_datamodule.test,
    #     speech_train, speech_val, speech_test,
    #     bz=bz, val_bz=val_bz, num_replicas=NUM_GPUS, pin_memory=(not DEBUG),
    #     num_workers=num_workers,
    #     TrainBatchSampler=TrainBatchSampler,
    #     ValSampler=ValSampler,
    #     TestSampler=TestSampler
    # )
    
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
    topk_dict  = getTopK(model, testset, text_transform, lm_directory, k=k)
    save_fname = os.path.join(os.path.split(FLAGS.checkpoint)[0],
                              f'top{k}_data.mat')
    
    savemat(save_fname, topk_dict)
    print('Predictions saved to:',  save_fname)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python get_top_k_preds.py --checkpoint=PATH_TO_MODEL")
    FLAGS(sys.argv)
    evaluate_saved()

