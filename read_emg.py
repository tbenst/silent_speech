import re
import os
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import defaultdict
import scipy.signal
import json
import copy
import glob
import sys
import pickle
import string
import logging
import pytorch_lightning as pl
from functools import lru_cache
from copy import copy

import soundfile as sf
import librosa
from scipy.io import loadmat

import torch
from data_utils import load_audio, get_emg_features, FeatureNormalizer, phoneme_inventory, read_phonemes, TextTransform
from torch.utils.data import DataLoader

DATA_FOLDER    = '/oak/stanford/projects/babelfish/magneto/GaddyPaper/'
project_folder = '/home/users/ghwilson/projects/silent_speech/'

REMOVE_CHANNELS = []
SILENT_DATA_DIRECTORIES = [f'{DATA_FOLDER}/emg_data/silent_parallel_data']
VOICED_DATA_DIRECTORIES = [f'{DATA_FOLDER}/emg_data/voiced_parallel_data'
                                              f'{DATA_FOLDER}/emg_data/nonparallel_data']
TESTSET_FILE = f'{project_folder}/testset_largedev.json'
TEXT_ALIGN_DIRECTORY = f'{DATA_FOLDER}/text_alignments'

def remove_drift(signal, fs):
    b, a = scipy.signal.butter(3, 2, 'highpass', fs=fs)
    return scipy.signal.filtfilt(b, a, signal)

def notch(signal, freq, sample_frequency):
    b, a = scipy.signal.iirnotch(freq, 30, sample_frequency)
    return scipy.signal.filtfilt(b, a, signal)

def notch_harmonics(signal, freq, sample_frequency):
    for harmonic in range(1,8):
        signal = notch(signal, freq*harmonic, sample_frequency)
    return signal

def subsample(signal, new_freq, old_freq):
    times = np.arange(len(signal))/old_freq
    sample_times = np.arange(0, times[-1], 1/new_freq)
    result = np.interp(sample_times, times, signal)
    return result

def apply_to_all(function, signal_array, *args, **kwargs):
    results = []
    for i in range(signal_array.shape[1]):
        results.append(function(signal_array[:,i], *args, **kwargs))
    return np.stack(results, 1)

def load_utterance(base_dir, index, limit_length=False, debug=False, text_align_directory=None, returnRaw= False):
    index   = int(index)
    raw_emg = np.load(os.path.join(base_dir, f'{index}_emg.npy'))
    before  = os.path.join(base_dir, f'{index-1}_emg.npy')
    after   = os.path.join(base_dir, f'{index+1}_emg.npy')
    if os.path.exists(before):
        raw_emg_before = np.load(before)
    else:
        raw_emg_before = np.zeros([0,raw_emg.shape[1]])
    if os.path.exists(after):
        raw_emg_after = np.load(after)
    else:
        raw_emg_after = np.zeros([0,raw_emg.shape[1]])

    x = np.concatenate([raw_emg_before, raw_emg, raw_emg_after], 0)
    x = apply_to_all(notch_harmonics, x, 60, 1000)
    x = apply_to_all(remove_drift, x, 1000)
    x = x[raw_emg_before.shape[0]:x.shape[0]-raw_emg_after.shape[0],:]

    if returnRaw:
        emg_orig = apply_to_all(subsample, x, 689.06, 1000)
    else:
        emg_orig = x.copy()
    x   = apply_to_all(subsample, x, 516.79, 1000)
    emg = x
    
    for c in REMOVE_CHANNELS:
        emg[:,int(c)] = 0
        emg_orig[:,int(c)] = 0

    emg_features = get_emg_features(emg)

    mfccs = load_audio(os.path.join(base_dir, f'{index}_audio_clean.flac'),
            max_frames=min(emg_features.shape[0], 800 if limit_length else float('inf')))

    if emg_features.shape[0] > mfccs.shape[0]:
        emg_features = emg_features[:mfccs.shape[0],:]
    assert emg_features.shape[0] == mfccs.shape[0]
    emg      = emg[6:6+6*emg_features.shape[0],:]
    emg_orig = emg_orig[8:8+8*emg_features.shape[0],:]
    assert emg.shape[0] == emg_features.shape[0]*6

    with open(os.path.join(base_dir, f'{index}_info.json')) as f:
        info = json.load(f)

    sess = os.path.basename(base_dir)
    tg_fname = f'{text_align_directory}/{sess}/{sess}_{index}_audio.TextGrid'
    if os.path.exists(tg_fname):
        phonemes = read_phonemes(tg_fname, mfccs.shape[0])
    else:
        phonemes = np.zeros(mfccs.shape[0], dtype=np.int64)+phoneme_inventory.index('sil')

    return mfccs, emg_features, info['text'], (info['book'],info['sentence_index']), phonemes, emg_orig.astype(np.float32)

class EMGDirectory(object):
    def __init__(self, session_index, directory, silent, exclude_from_testset=False):
        self.session_index = session_index
        self.directory = directory
        self.silent = silent
        self.exclude_from_testset = exclude_from_testset

    def __lt__(self, other):
        return self.session_index < other.session_index

    def __repr__(self):
        return self.directory

class SizeAwareSampler(torch.utils.data.Sampler):
    def __init__(self, emg_dataset, max_len):
        self.dataset = emg_dataset
        self.max_len = max_len

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)
        batch = []
        batch_length = 0
        for idx in indices:
            directory_info, file_idx = self.dataset.example_indices[idx]
            with open(os.path.join(directory_info.directory, f'{file_idx}_info.json')) as f:
                info = json.load(f)
            if not np.any([l in string.ascii_letters for l in info['text']]):
                continue
            length = sum([emg_len for emg_len, _, _ in info['chunks']])
            if length > self.max_len:
                logging.warning(f'Warning: example {idx} cannot fit within desired batch length')
            if length + batch_length > self.max_len:
                yield batch
                batch = []
                batch_length = 0
            batch.append(idx)
            batch_length += length
        # dropping last incomplete batch
        
def lookup_emg_length(example):
    x = loadmat(example)['audio_file'][0]
    json_file = x.split('_audio_clean')[0] + '_info.json'

    with open(json_file) as f:
        info = json.load(f)
        
        
    if not np.any([l in string.ascii_letters for l in info['text']]):
        return False

    length = sum([emg_len for emg_len, _, _ in info['chunks']])
    return length
        
class PreprocessedSizeAwareSampler(torch.utils.data.Sampler):
    def __init__(self, emg_dataset, max_len):
        self.dataset = emg_dataset
        self.max_len = max_len
        
        self.lengths = [lookup_emg_length(ex) for ex in self.dataset.example_indices]
        self.approx_len = int(np.ceil(np.array(self.lengths)).sum() / max_len)

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)
        batch = []
        batch_length = 0
        for idx in indices:
            length = self.lengths[idx]
            if length > self.max_len:
                logging.warning(f'Warning: example {idx} cannot fit within desired batch length')
            if length + batch_length > self.max_len:
                yield batch
                batch = []
                batch_length = 0
            batch.append(idx)
            batch_length += length
        # dropping last incomplete batch
        
    def __len__(self):
        return self.approx_len

class EMGDataset(torch.utils.data.Dataset):
    def __init__(self, base_dir=None, normalizers_file=None, limit_length=False, dev=False, test=False, no_testset=False, 
                 no_normalizers=False, returnRaw = False, togglePhones = False):

        self.text_align_directory = TEXT_ALIGN_DIRECTORY

        if no_testset:
            devset = []
            testset = []
        else:
            with open(TESTSET_FILE) as f:
                testset_json = json.load(f)
                devset = testset_json['dev']
                testset = testset_json['test']

        directories = []
        if base_dir is not None:
            directories.append(EMGDirectory(0, base_dir, False))
        else:
            for sd in SILENT_DATA_DIRECTORIES:
                for session_dir in sorted(os.listdir(sd)):
                    directories.append(EMGDirectory(len(directories), os.path.join(sd, session_dir), True))

            has_silent = len(SILENT_DATA_DIRECTORIES) > 0
            for vd in VOICED_DATA_DIRECTORIES:
                for session_dir in sorted(os.listdir(vd)):
                    directories.append(EMGDirectory(len(directories), os.path.join(vd, session_dir), False, exclude_from_testset=has_silent))

        self.example_indices = []
        self.voiced_data_locations = {} # map from book/sentence_index to directory_info/index
        for directory_info in directories:
            for fname in os.listdir(directory_info.directory):
                m = re.match(r'(\d+)_info.json', fname)
                if m is not None:
                    idx_str = m.group(1)
                    with open(os.path.join(directory_info.directory, fname)) as f:
                        info = json.load(f)
                        if info['sentence_index'] >= 0: # boundary clips of silence are marked -1
                            location_in_testset = [info['book'], info['sentence_index']] in testset
                            location_in_devset = [info['book'], info['sentence_index']] in devset
                            if (test and location_in_testset and not directory_info.exclude_from_testset) \
                                    or (dev and location_in_devset and not directory_info.exclude_from_testset) \
                                    or (not test and not dev and not location_in_testset and not location_in_devset):
                                self.example_indices.append((directory_info,int(idx_str)))

                            if not directory_info.silent:
                                location = (info['book'], info['sentence_index'])
                                self.voiced_data_locations[location] = (directory_info,int(idx_str))

        self.example_indices.sort()
        random.seed(0)
        random.shuffle(self.example_indices)

        self.no_normalizers = no_normalizers
        if not self.no_normalizers:
            self.mfcc_norm, self.emg_norm = pickle.load(open(normalizers_file,'rb'))

        sample_mfccs, sample_emg, _, _, _, _ = load_utterance(self.example_indices[0][0].directory, self.example_indices[0][1])
        self.num_speech_features = sample_mfccs.shape[1]
        self.num_features = sample_emg.shape[1]
        self.limit_length = limit_length
        self.num_sessions = len(directories)
        self.returnRaw    = returnRaw

        self.text_transform = TextTransform(togglePhones = togglePhones)

    def silent_subset(self):
        result = copy(self)
        silent_indices = []
        for example in self.example_indices:
            if example[0].silent:
                silent_indices.append(example)
        result.example_indices = silent_indices
        return result

    def subset(self, fraction):
        result = copy(self)
        result.example_indices = self.example_indices[:int(fraction*len(self.example_indices))]
        return result

    def __len__(self):
        return len(self.example_indices)

    @lru_cache(maxsize=None)
    def __getitem__(self, i):
        directory_info, idx = self.example_indices[i]
        mfccs, emg, text, book_location, phonemes, raw_emg = load_utterance(directory_info.directory, idx, self.limit_length, text_align_directory=self.text_align_directory, returnRaw = self.returnRaw)
        raw_emg = raw_emg / 20
        raw_emg = 50*np.tanh(raw_emg/50.)

        if not self.no_normalizers:
            mfccs = self.mfcc_norm.normalize(mfccs)
            emg = self.emg_norm.normalize(emg)
            emg = 8*np.tanh(emg/8.)
            

        session_ids = np.full(emg.shape[0], directory_info.session_index, dtype=np.int64)
        audio_file = f'{directory_info.directory}/{idx}_audio_clean.flac'

        text_int = np.array(self.text_transform.text_to_int(text), dtype=np.int64)

        result = {'audio_features':torch.from_numpy(mfccs).pin_memory(), 'emg':torch.from_numpy(emg).pin_memory(), 'text':text, 'text_int': torch.from_numpy(text_int).pin_memory(), 'file_label':idx, 'session_ids':torch.from_numpy(session_ids).pin_memory(), 'book_location':book_location, 'silent':directory_info.silent, 'raw_emg':torch.from_numpy(raw_emg).pin_memory()}

        if directory_info.silent:
            voiced_directory, voiced_idx = self.voiced_data_locations[book_location]
            voiced_mfccs, voiced_emg, _, _, phonemes, _ = load_utterance(voiced_directory.directory, voiced_idx, False, text_align_directory=self.text_align_directory)

            if not self.no_normalizers:
                #voiced_mfccs = self.mfcc_norm.normalize(voiced_mfccs)  # HACKY WORKAROUND - AVOID MAKING MFCCS
                voiced_emg = self.emg_norm.normalize(voiced_emg)
                voiced_emg = 8*np.tanh(voiced_emg/8.)

            result['parallel_voiced_audio_features'] = torch.from_numpy(voiced_mfccs).pin_memory()
            result['parallel_voiced_emg'] = torch.from_numpy(voiced_emg).pin_memory()

            audio_file = f'{voiced_directory.directory}/{voiced_idx}_audio_clean.flac'

        result['phonemes'] = torch.from_numpy(phonemes).pin_memory() # either from this example if vocalized or aligned example if silent
        result['audio_file'] = audio_file

        return result

    @staticmethod
    def collate_raw(batch):
        batch_size = len(batch)
        audio_features = []
        audio_feature_lengths = []
        parallel_emg = []
        for ex in batch:
            if ex['silent']:
                audio_features.append(ex['parallel_voiced_audio_features'])
                audio_feature_lengths.append(ex['parallel_voiced_audio_features'].shape[0])
                parallel_emg.append(ex['parallel_voiced_emg'])
            else:
                audio_features.append(ex['audio_features'])
                audio_feature_lengths.append(ex['audio_features'].shape[0])
                parallel_emg.append(np.zeros(1))
        phonemes = [ex['phonemes'] for ex in batch]
        emg = [ex['emg'] for ex in batch]
        raw_emg = [ex['raw_emg'] for ex in batch]
        session_ids = [ex['session_ids'] for ex in batch]
        lengths = [ex['emg'].shape[0] for ex in batch]
        silent = [ex['silent'] for ex in batch]
        text_ints = [ex['text_int'] for ex in batch]
        text_lengths = [ex['text_int'].shape[0] for ex in batch]

        result = {'audio_features':audio_features,
                  'audio_feature_lengths':audio_feature_lengths,
                  'emg':emg,
                  'raw_emg':raw_emg,
                  'parallel_voiced_emg':parallel_emg,
                  'phonemes':phonemes,
                  'session_ids':session_ids,
                  'lengths':lengths,
                  'silent':silent,
                  'text_int':text_ints,
                  'text_int_lengths':text_lengths}
        return result
    
    
class PreprocessedEMGDataset(torch.utils.data.Dataset):
    def __init__(self, base_dir=None, train = False, dev=False, test=False, limit_length = False,
                 pin_memory = True, no_normalizers = False, togglePhones = False, device = None,
                 normalizers_file=None):
        
        self.togglePhones = togglePhones

        files = list()
        if train:
            partition_files = glob.glob(os.path.join(base_dir, 'train/') + '*/*.mat')
            #print(f'Adding {len(partition_files)} to dataset.')
            files.extend(partition_files)
            
        if dev:
            partition_files = glob.glob(os.path.join(base_dir, 'dev/') + '*/*.mat')
            #print(f'Adding {len(partition_files)} to dataset.')
            files.extend(partition_files)
            
        if test:
            partition_files = glob.glob(os.path.join(base_dir, 'test/') + '*/*.mat')
            #print(f'Adding {len(partition_files)} to dataset.')
            files.extend(partition_files)
            
        self.example_indices = files
        self.train = train
        self.dev = dev
        self.test = test
        self.pin_memory = pin_memory 
        self.device = device

        self.example_indices.sort()
        np.random.seed(0)
        np.random.shuffle(self.example_indices)

    
        self.num_speech_features = loadmat(self.example_indices[0])['audio_features'].shape[1]
        self.num_features        = loadmat(self.example_indices[0])['emg'].shape[1]
        self.limit_length = limit_length
        self.num_sessions = len(glob.glob(os.path.join(base_dir, 'train/') + '*session_*/'))
        
        self.text_transform = TextTransform(togglePhones = self.togglePhones)
        
        self.no_normalizers = no_normalizers
        if not self.no_normalizers:
            self.mfcc_norm, self.emg_norm = pickle.load(open(normalizers_file,'rb'))

    def silent_subset(self):
        result = copy(self)
        silent_indices = []
        for example in self.example_indices:
            if example[0].silent:
                silent_indices.append(example)
        result.example_indices = silent_indices
        return result

    def subset(self, fraction):
        result = copy(self)
        result.example_indices = self.example_indices[:int(fraction*len(self.example_indices))]
        return result

    def __len__(self):
        return len(self.example_indices)

    @lru_cache(maxsize=None)
    def __getitem__(self, i):
        
        result = loadmat(self.example_indices[i])
                
        if self.pin_memory:
            keys = ['audio_features',  'emg', 'text', 'session_ids', 'raw_emg', 'phonemes', 
                    'parallel_voiced_emg', 'parallel_voiced_audio_features']
            for key in keys:
                try:
                    result[key] = torch.tensor(result[key].squeeze()).pin_memory()
                except:
                    continue
                    
        result['text_int'] = torch.tensor(np.array(self.text_transform.text_to_int(result['text'][0]), dtype=np.int64)).pin_memory()

        return result
    

    @staticmethod
    def collate_raw(batch):
        
        audio_features        = []
        audio_feature_lengths = []
        parallel_emg          = []
        for ex in batch:
            if ex['silent']:
                audio_features.append(ex['parallel_voiced_audio_features'])
                audio_feature_lengths.append(ex['parallel_voiced_audio_features'].shape[0])
                parallel_emg.append(ex['parallel_voiced_emg'])
            else:
                audio_features.append(ex['audio_features'])
                audio_feature_lengths.append(ex['audio_features'].shape[0])
                parallel_emg.append(np.zeros(1))
                
        phonemes    = [ex['phonemes'] for ex in batch]
        emg         = [ex['emg'] for ex in batch]
        raw_emg     = [ex['raw_emg'] for ex in batch]
        session_ids = [ex['session_ids'] for ex in batch]
        lengths     = [ex['emg'].shape[0] for ex in batch]
        silent      = [ex['silent'] for ex in batch]
        text        = [ex['text'] for ex in batch]
        text_int    = [ex['text_int'] for ex in batch]
        int_lengths = [ex['text_int'].shape[0] for ex in batch]

        result = {'audio_features':audio_features,
                  'audio_feature_lengths':audio_feature_lengths,
                  'emg':emg,
                  'raw_emg':raw_emg,
                  'parallel_voiced_emg':parallel_emg,
                  'phonemes':phonemes,
                  'session_ids':session_ids,
                  'lengths':lengths,
                  'silent':silent,
                  'text':text,
                  'text_int': text_int,
                  'text_int_lengths':int_lengths}
        
        return result
    
    
class EMGDataModule(pl.LightningDataModule):
    def __init__(self, base_dir, togglePhones, normalizers_file,
                 batch_size=32, max_len=128000, num_workers=0) -> None:
        super().__init__()
        self.train = PreprocessedEMGDataset(base_dir = base_dir, train = True, dev = False, test = False,
                                        togglePhones = togglePhones, normalizers_file = normalizers_file)
        self.val   = PreprocessedEMGDataset(base_dir = base_dir, train = False, dev = True, test = False,
                                        togglePhones = togglePhones, normalizers_file = normalizers_file)
        
        self.test = PreprocessedEMGDataset(base_dir = base_dir, train = False, dev = False, test = True,
                                    togglePhones = togglePhones, normalizers_file = normalizers_file)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_len = max_len
        
    def train_dataloader(self):
        loader = DataLoader(
            self.train,
            # batch_size=self.batch_size,
            collate_fn=self.train.collate_raw,
            # shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            batch_sampler = PreprocessedSizeAwareSampler(self.train, self.max_len)
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.val,
            batch_size=1, # gaddy uses bz=1 for val/test, does this matter for WER..?
            collate_fn=self.val.collate_raw,
            num_workers=self.num_workers,
            pin_memory=True
        )
        return loader

    def test_dataloader(self):
        loader = DataLoader(
            self.test,
            batch_size=1,
            collate_fn=self.val.collate_raw,
            num_workers=self.num_workers,
            pin_memory=True
        )
        return loader


def make_normalizers(normalizers_file):
    dataset = EMGDataset(no_normalizers=True)
    mfcc_samples = []
    emg_samples = []
    for d in dataset:
        mfcc_samples.append(d['audio_features'])
        emg_samples.append(d['emg'])
        if len(emg_samples) > 50:
            break
    mfcc_norm = FeatureNormalizer(mfcc_samples, share_scale=True)
    emg_norm = FeatureNormalizer(emg_samples, share_scale=False)
    pickle.dump((mfcc_norm, emg_norm), open(normalizers_file, 'wb'))
