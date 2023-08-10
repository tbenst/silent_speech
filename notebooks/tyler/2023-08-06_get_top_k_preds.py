'''
Using pretrained model and a torch-compatible language model, generate top k candidates and save data.
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

from absl import flags

hostname = subprocess.run("hostname", capture_output=True)
ON_SHERLOCK = hostname.stdout[:2] == b"sh"
if ON_SHERLOCK:
    # TODO: bechmark SCRATCH vs LOCAL_SCRATCH ...?
    # scratch_directory = os.environ["LOCAL_SCRATCH"]
    gaddy_dir = '/oak/stanford/projects/babelfish/magneto/GaddyPaper/'
else:
    # on my local machine
    gaddy_dir = '/scratch/GaddyPaper/'

FLAGS = flags.FLAGS
flags.DEFINE_integer('k', 100, 'max beams to return')
flags.DEFINE_integer('beam_size', 500, 'maximum number of beams to search')
flags.DEFINE_float('lm_weight', 2., 'language model weight')
flags.DEFINE_integer('beam_threshold', 75, 'prune beam search if more than this away from best score')
flags.DEFINE_string('checkpoint', None, 'run evaluation on given model file')
flags.DEFINE_boolean('no_lm', None, "don't use a language model (pass as --no_lm)")
flags.DEFINE_string('gaddy_dir', gaddy_dir, 
                    'Path to GaddyPaper directory')

togglePhones = False

def getTopK(model, dataloader, text_transform, lm_directory,
        k = 100, beam_size=500, togglePhones=False, use_lm=True,
        beam_threshold=100, lm_weight=2):
    model.eval()
    
    if togglePhones:
        lexicon_file = os.path.join(lm_directory, 'cmudict.txt')
    else:
        lexicon_file = os.path.join(lm_directory, 'lexicon_graphemes_noApostrophe.txt')
    # prune beam search if more than this away from best score
    
    if use_lm:
        lm = os.path.join(lm_directory, '4gram_lm.bin')
    else:
        lm = None
 
    decoder = ctc_decoder(
       lexicon = lexicon_file,
       tokens  = text_transform.chars + ['_'],
       lm      = lm,
       blank_token = '_',
       sil_token   = '|',
       nbest       = k,
       # default is 2; Gaddy sets to 1.85
       # makes seemingly no difference with those two values
       lm_weight   = lm_weight,
       #word_score  = -3,
       #sil_score   = -2,
    #    beam_size   = k+50,
       beam_size   = beam_size,
    #    beam_size   = int(k*1.5),
       beam_threshold = beam_threshold # defaults to 50
    )

    topk_dict = {
        'k'          : k,
        'beam_size'  : beam_size,
        'beam_threshold': beam_threshold,
        'sentences'  : [],
        'predictions': [],
        'beam_scores': [],
    }
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            X = nn.utils.rnn.pad_sequence(batch['raw_emg'], batch_first=True)
            bz = X.shape[0]
            X = X.cuda()

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
            
            ctc_str = "Number of CTC hypotheses per example: "
            for ctc_hypotheses in beam_results:
                ctc_str += f"{len(ctc_hypotheses)},"
            print(ctc_str)

            for i, (example, beam_result) in enumerate(zip(batch, beam_results)):
                trl_top_k = []
                trl_beam_scores = []
                for beam in beam_result:
                    transcript = " ".join(beam.words).strip().lower()
                    score      = beam.score
                    trl_top_k.append(transcript)
                    trl_beam_scores.append(score)
               
                # Filter out silences
                target_sentence = text_transform.clean_2(batch['text'][i])
                if len(target_sentence) > 0:
                    topk_dict['predictions'].append(np.array(trl_top_k))
                    topk_dict['beam_scores'].append(np.array(trl_beam_scores))
                    topk_dict['sentences'].append(target_sentence)       
                
    topk_dict['predictions'] = np.array(topk_dict['predictions'])
    topk_dict['beam_scores'] = np.array(topk_dict['beam_scores'])
    topk_dict['sentences'] = np.array(topk_dict['sentences'])

    return topk_dict


def evaluate_saved():     
    data_dir = os.path.join(FLAGS.gaddy_dir, 'processed_data/')
    lm_directory = os.path.join(FLAGS.gaddy_dir, 'pretrained_models/librispeech_lm/')
    normalizers_file = os.path.join(SCRIPT_DIR, "normalizers.pkl")
    
    togglePhones = False
    text_transform = TextTransform(togglePhones = togglePhones)
    val_bz = 12
    max_len = 48000
    emg_datamodule = EMGDataModule(data_dir, togglePhones, normalizers_file, max_len=max_len,
        collate_fn=collate_gaddy_or_speech,
        pin_memory=True, batch_size=val_bz)
    
    testset = emg_datamodule.val_dataloader()
    
    checkpoint = torch.load(FLAGS.checkpoint)
    steps_per_epoch = 200 
    n_chars = len(text_transform.chars)
    num_outs = n_chars + 1 # +1 for CTC blank token ( i think? )
    precision = "16-mixed"
    config = SpeechOrEMGToTextConfig(steps_per_epoch, lm_directory, num_outs, precision=precision)
    model = SpeechOrEMGToText(config, text_transform)
    model.load_state_dict(checkpoint["state_dict"])
    model.cuda()
    k = FLAGS.k
    beam_size = FLAGS.beam_size
    beam_threshold = FLAGS.beam_threshold
    lm_weight = FLAGS.lm_weight
    use_lm = FLAGS.no_lm is None
    print(use_lm)
    exit(0)
    topk_dict  = getTopK(model, testset, text_transform, lm_directory,
        k=k, beam_size=beam_size, beam_threshold=beam_threshold, use_lm=use_lm)
    name = f"top{k}_{beam_size}beams_thresh{beam_threshold}_lmweight{lm_weight}"
    if not use_lm:
        name += "_noLM"
    save_fname = os.path.join(os.path.split(FLAGS.checkpoint)[0],
        f'{name}.npz')
    
    np.savez(save_fname, **topk_dict)
    print('Predictions saved to:',  save_fname)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python get_top_k_preds.py --checkpoint=PATH_TO_MODEL")
        print("Example: CUDA_VISIBLE_DEVICES=1 python 2023-08-06_get_top_k_preds.py --checkpoint /scratch/users/tbenst/2023-08-01T06:54:28.359594_gaddy/SpeechOrEMGToText-epoch=199-val/wer=0.264.ckpt --beam_size 150 --beam_threshold 50")
    FLAGS(sys.argv)
    evaluate_saved()

