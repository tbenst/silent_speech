'''
Some things to improve on:
    - used torchaudio as an exception but worth just removing soundfile entirely from here and using torchaudio in place
'''

import sys
import os
import shutil
import numpy as np

import soundfile as sf
import librosa

import torch
from torch import nn
import torchaudio

from architecture import Model, S4Model
from transduction_model import get_aligned_prediction
from read_emg import EMGDataset, PreprocessedEMGDataset
from data_utils import phoneme_inventory

from absl import flags
FLAGS = flags.FLAGS
flags.DEFINE_string('model', None, 'checkpoint of model to run')

def main():
    trainset = PreprocessedEMGDataset(base_dir = FLAGS.base_dir, train = True, dev = False, test = False,
                                     togglePhones = False)
    #trainset = trainset.subset(0.01) # FOR DEBUGGING - REMOVE WHEN RUNNING
    devset   = PreprocessedEMGDataset(base_dir = FLAGS.base_dir, train = False, dev = True, test = False,
                                     togglePhones = False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if FLAGS.S4:
        model = S4Model(devset.num_features, devset.num_speech_features, len(phoneme_inventory)).to(device)
    else:
        model = Model(devset.num_features, devset.num_speech_features, len(phoneme_inventory)).to(device)
        
    n_phones   = len(phoneme_inventory)
    state_dict = torch.load(FLAGS.model)
    model.load_state_dict(state_dict)

    os.makedirs(os.path.join(FLAGS.output_directory, 'mels'), exist_ok=True)
    os.makedirs(os.path.join(FLAGS.output_directory, 'wavs'), exist_ok=True)

    for dataset, name_prefix in [(trainset, 'train'), (devset, 'dev')]:
        with open(os.path.join(FLAGS.output_directory, f'{name_prefix}_filelist.txt'), 'w') as filelist:
            for i, datapoint in enumerate(dataset):
                spec = get_aligned_prediction(model, datapoint, device, dataset.mfcc_norm)
                spec = spec.T[np.newaxis,:,:].detach().cpu().numpy()
                np.save(os.path.join(FLAGS.output_directory, 'mels', f'{name_prefix}_output_{i}.npy'), spec)
                try:
                    audio, r = sf.read(datapoint['audio_file'][0])
                except:
                    audio, r = torchaudio.load(datapoint['audio_file'][0])
                    audio    = audio.numpy()
                if r != 22050:
                    audio = librosa.resample(audio, orig_sr = r, target_sr = 22050, res_type='kaiser_fast')
                audio = np.clip(audio, -1, 1) # because resampling sometimes pushes things out of range
                try:
                    sf.write(os.path.join(FLAGS.output_directory, 'wavs', f'{name_prefix}_output_{i}.wav'), audio, 22050)
                except:
                    torchaudio.save(os.path.join(FLAGS.output_directory, 'wavs', f'{name_prefix}_output_{i}.wav'), torch.tensor(audio), 22050)
                filelist.write(f'{name_prefix}_output_{i}\n')
        

if __name__ == "__main__":
    FLAGS(sys.argv)
    main()
