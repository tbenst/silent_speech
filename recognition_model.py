'''
This script has been modified to use torchaudio in place of ctcdecode. On standard benchmarks it provides a ~10-100x speed improvement.

Download the Relevant LM files and then point script toward a directory holding the files via --lm_directory flag. Files can be obtained through:

wget -c https://download.pytorch.org/torchaudio/download-assets/librispeech-3-gram/{lexicon.txt, tokens.txt, lm.bin}

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

from absl import flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean('debug', False, 'debug')
flags.DEFINE_boolean('log_neptune', True, 'log training to Neptune.ai')
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


seqlen       = 600
togglePhones = False

# horrible hack to get around this repo not being a proper python package
SCRIPT_DIR = os.getcwd()
normalizers_file = os.path.join(SCRIPT_DIR, "normalizers.pkl")
print(f"{normalizers_file=}")

def test(model, testset, device):
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
       nbest       = 1,
       lm_weight   = 2, # default is 2; Gaddy sets to 1.85
       #word_score  = -3,
       #sil_score   = -2,
       beam_size   = 150  # SET TO 150 during inference
    )

    dataloader  = torch.utils.data.DataLoader(testset, batch_size=1, collate_fn=testset.collate_raw)
    references  = []
    predictions = []
    with torch.no_grad():
        for example in dataloader:
            X     = example['emg'][0].unsqueeze(0).to(device)
            X_raw = example['raw_emg'][0].unsqueeze(0).to(device)
            sess  = example['session_ids'][0].to(device)

            pred  = F.log_softmax(model(X, X_raw, sess), -1).cpu()

            beam_results = decoder(pred)
            pred_int     = beam_results[0][0].tokens
            pred_text    = ' '.join(beam_results[0][0].words).strip().lower()
            #pred_text    = testset.text_transform.int_to_text(pred_int)
            #target_text  = testset.text_transform.int_to_text(example['text_int'][0])
            target_text  = testset.text_transform.clean_2(example['text'][0][0])
            
            if len(target_text) > 0:
                references.append(target_text)
                predictions.append(pred_text)
            #print('Prediction: ', pred_text)
            #print('Target: ', target_text)

    model.train()
    return jiwer.wer(references, predictions)


def train_model(trainset, devset, device, n_epochs):
    
    dataloader = torch.utils.data.DataLoader(trainset, pin_memory=(device=='cuda'), 
                                         collate_fn=devset.collate_raw, num_workers=0,
                                         batch_sampler = PreprocessedSizeAwareSampler(trainset, 128000))


    n_chars = len(devset.text_transform.chars)
    
    if FLAGS.S4:
        #model = H3Model(devset.num_features, n_chars+1).to(device)
        model = S4Model(devset.num_features, n_chars+1).to(device)
    else:
        model = Model(devset.num_features, n_chars+1).to(device)

    if FLAGS.log_neptune:
        run = neptune.init_run(
        project="neuro/Gaddy",    
        api_token=os.environ["NEPTUNE_API_TOKEN"])  

    logging.info(model)
    model_parameters  = filter(lambda p: p.requires_grad, model.parameters())
    params            = sum([np.prod(p.size()) for p in model_parameters])
    if FLAGS.log_neptune:
        run['num_parameters'] = params
    logging.info(f'Number of parameters: {params}')

    if FLAGS.start_training_from is not None:
        state_dict = torch.load(FLAGS.start_training_from)
        model.load_state_dict(state_dict, strict=False)

    optim    = torch.optim.AdamW(model.parameters(), lr=FLAGS.learning_rate, weight_decay=FLAGS.l2)
    lr_sched = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[125,150,175], gamma=.5)

    #lr_sched = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[150,175,200], gamma=.5)
    #lr_sched = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr = FLAGS.learning_rate, epochs = FLAGS.epochs,
    #                                              steps_per_epoch = int(8055 / FLAGS.batch_size) + 1)
    
    
    params = {}
    for flag in FLAGS:
        params[flag] = getattr(FLAGS, flag) 
    
    def set_lr(new_lr):
        for param_group in optim.param_groups:
            param_group['lr'] = new_lr

    target_lr = FLAGS.learning_rate
    def schedule_lr(iteration):
        iteration = iteration + 1
        if iteration <= FLAGS.learning_rate_warmup:
            set_lr(iteration*target_lr/FLAGS.learning_rate_warmup)
            
    if FLAGS.log_neptune:
        run['hyperparameters'] = params
        run["sys/tags"].add("MultiStepLR")
        run["sys/tags"].add("800Hz")
        run["sys/tags"].add("8xDownsampling")
        run["sys/tags"].add("FCN_embedding")

    batch_idx = 0
    optim.zero_grad()
    for epoch_idx in range(n_epochs):
        losses = []
        for example in dataloader:
            schedule_lr(batch_idx)

            X     = combine_fixed_length(example['emg'], seqlen).to(device)
            X_raw = combine_fixed_length(example['raw_emg'], seqlen*8).to(device)
            sess  = combine_fixed_length(example['session_ids'], seqlen).to(device)

            pred = model(X, X_raw, sess)
            pred = F.log_softmax(pred, 2)

            # seq first, as required by ctc
            pred = nn.utils.rnn.pad_sequence(decollate_tensor(pred, example['lengths']), batch_first=False) 
            y    = nn.utils.rnn.pad_sequence(example['text_int'], batch_first=True).to(device)
            loss = F.ctc_loss(pred, y, example['lengths'], example['text_int_lengths'], blank=n_chars)
            
            if torch.isnan(loss) or torch.isinf(loss):
                print('batch:', batch_idx)
                print('Isnan output:',torch.any(torch.isnan(pred)))
                print('Isinf output:',torch.any(torch.isinf(pred)))
                raise ValueError("NaN/Inf detected in loss")
                
            losses.append(loss.item())
            
            loss.backward()
            if (batch_idx+1) % 2 == 0:
                nn.utils.clip_grad_norm_(model.parameters(), 10)
                optim.step()
                optim.zero_grad(set_to_none=True)
                #lr_sched.step() # EXPERIMENTAL
                
            del example, pred, loss, y, sess, X, X_raw
            torch.cuda.empty_cache()

            batch_idx += 1
        train_loss = np.mean(losses)
        val        = test(model, devset, device)
        lr_sched.step() # EXPERIMENTAL
        logging.info(f'finished epoch {epoch_idx+1} - training loss: {train_loss:.4f} validation WER: {val*100:.2f}')
        
        if FLAGS.log_neptune:
            run["train/loss"].log(train_loss)
            run["val/WER"].log(val * 100)
        
        
        torch.save(model.state_dict(), os.path.join(FLAGS.output_directory,'model.pt'))
        
    run.stop()
    model.load_state_dict(torch.load(os.path.join(FLAGS.output_directory,'model.pt'))) # re-load best parameters
    
    return model


def evaluate_saved():
    device  = 'cuda' if torch.cuda.is_available() and not FLAGS.debug else 'cpu'
    #testset = PreprocessedEMGDataset(base_dir = FLAGS.base_dir, train = False, dev = False, test = True)
    
    testset = PreprocessedEMGDataset(base_dir = FLAGS.base_dir, train = False, dev = True, test = False,
                                    togglePhones = togglePhones, normalizers_file = normalizers_file)
    n_chars = len(testset.text_transform.chars)
    
    if FLAGS.S4:
        model = S4Model(testset.num_features, n_chars+1).to(device)
    else:
        model = Model(testset.num_features, n_chars+1).to(device)
    
    model.load_state_dict(torch.load(FLAGS.evaluate_saved))
    print('WER:', test(model, testset, device))

def main():
    os.makedirs(FLAGS.output_directory, exist_ok=True)
    logging.basicConfig(handlers=[
            logging.FileHandler(os.path.join(FLAGS.output_directory, 'log.txt'), 'w'),
            logging.StreamHandler()
            ], level=logging.INFO, format="%(message)s")

    logging.info(sys.argv)
    
    trainset = PreprocessedEMGDataset(base_dir = FLAGS.base_dir, train = True, dev = False, test = False,
                                     togglePhones = togglePhones, normalizers_file = normalizers_file)
    #trainset = trainset.subset(0.01) # FOR DEBUGGING - REMOVE WHEN RUNNING
    devset   = PreprocessedEMGDataset(base_dir = FLAGS.base_dir, train = False, dev = True, test = False,
                                     togglePhones = togglePhones, normalizers_file = normalizers_file)
    
    logging.info('output example: %s', devset.example_indices[0])
    logging.info('train / dev split: %d %d',len(trainset),len(devset))

    device = 'cuda' if torch.cuda.is_available() and not FLAGS.debug else 'cpu'
    model  = train_model(trainset, devset, device, n_epochs = FLAGS.epochs)

if __name__ == '__main__':
    FLAGS(sys.argv)
    if FLAGS.evaluate_saved is not None:
        evaluate_saved()
    else:
        main()
