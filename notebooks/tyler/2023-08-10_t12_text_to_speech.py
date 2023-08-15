##
# Create .wav audio files for T12 dataset, and .lab text files of utterances.
# then see README.md for how to create textgrid files (phoneme alignments).
import os, sys, glob
import torch
from tqdm import tqdm
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchmetrics

import jiwer
from unidecode import unidecode
import re, sys
import string
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(SCRIPT_DIR)
from dataloaders import persist_to_file
##
from functools import lru_cache
from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav
from IPython.display import Audio

# download and load all models
preload_models()

datadir        = '/data/data/T12_data/competitionData'
sentences_dir = '/data/data/T12_data/sentences'
TTS_dir = os.path.join(os.path.dirname(datadir), 'TTS')

# format like `t12.2022.04.28.mat`
train_files = glob.glob(datadir + '*/train/*')
test_files  = glob.glob(datadir + '*/test/*')

# format like `t12.2022.04.28_sentences.mat`
sentences_files = glob.glob(sentences_dir + '/*')

train_file_mapping = {}
days = []
for train_file in train_files + test_files:
    day = train_file.split('.mat')[0].split('t12.')[1]
    sentences_file = [sf for sf in sentences_files if day in sf][0]
    train_file_mapping[train_file] = sentences_file
    days.append(day)

unique = np.unique(days)
session_mapping = dict(zip(unique, np.arange(len(unique))))
print('Unique days:', len(session_mapping.keys()))
##
@persist_to_file(os.path.join(os.path.dirname(datadir), "sentence_mapping_per_file.pkl"))
def get_sentence_mapping_per_file(train_files, test_files):
    sentence_mapping_per_file = {}
    for tf in tqdm((train_files + test_files)):
        mat = scipy.io.loadmat(tf)
        mat2 = scipy.io.loadmat(train_file_mapping[tf])
        matching_indices = []
        for i, sentence in enumerate(mat['sentenceText']):
            sentence = sentence.rstrip()  # strip whitespace at end
            matching_sentence = None
            last_match_idx = matching_indices[-1] if len(matching_indices) > 0 else 0
            # start at last matching index to avoid matching the same sentence twice
            for j, sentence2 in enumerate(mat2['sentences'][last_match_idx:], start=last_match_idx):
                sentence2 = str(sentence2[0][0]).rstrip()  # strip whitespace at end
                if sentence == sentence2:
                    matching_sentence = sentence2
                    matching_indices.append(j)
                    break
            if matching_sentence is None:
                raise Exception(f"No matching sentence found for sentence {i} in mat.\n{sentence}\n{sentence2}")
        if len(matching_indices) != len(set(matching_indices)):
            print(f"Error: There are {len(matching_indices) - len(set(matching_indices))} identical indices in matching_indices.") 
            # find identical index
            for i in range(len(matching_indices)):
                for j in range(len(matching_indices)):
                    if i != j and matching_indices[i] == matching_indices[j]:
                        print(f"Identical index ({matching_indices[i]}) at: {i} and {j}")
            assert len(matching_indices) == len(set(matching_indices)), "There are identical indices in matching_indices."
        sentence_mapping_per_file[tf] = matching_indices
    return sentence_mapping_per_file

sentence_mapping_per_file = get_sentence_mapping_per_file(train_files, test_files)
##
# read all sentences into a list
@persist_to_file(os.path.join(os.path.dirname(datadir), "all_sentences.pkl"))
def read_all_sentences(sentences_files):
    sentences = []
    for sentences_file in tqdm(sentences_files):
        mat = scipy.io.loadmat(sentences_file)
        for sentence in mat['sentences']:
            sentences.append(str(sentence[0][0]).rstrip())  # strip whitespace at end
    return sentences
all_sentences = read_all_sentences(sentences_files)
print('Number of sentences:', len(all_sentences))
print('Unique sentences:', len(np.unique(all_sentences)))
##
def sentence_to_fn(sentence, directory=TTS_dir, ext=".wav"):
    fn = re.sub(r'[^\w\s]', '', sentence)  # remove punctuation
    fn = fn.lower().replace(' ', '_')  # lowercase with underscores
    return os.path.join(directory, fn+ext)

# number of unique filenames should be equal to number of unique sentences. Print sentence pair if not.
from collections import defaultdict
uniq_sentences = np.unique(all_sentences)
filenames = defaultdict(list)
# halfway = len(uniq_sentences) // 2
halfway = 6291
for sentence in tqdm(uniq_sentences[1245:halfway]):
    fn = sentence_to_fn(sentence)
    filenames[fn].append(sentence)
    audio_array = generate_audio(sentence, history_prompt="v2/en_speaker_9", silent=True)
    write_wav(fn, SAMPLE_RATE, audio_array)
    
for fn, sentences in filenames.items():
    if len(sentences) > 1:
        print(fn, sentences)
        # There is one...
        # ['I hope you guys enjoy it!', 'I hope you guys enjoy it.']
        raise Exception("There are multiple sentences with the same filename.")

print('done with first half starting until', halfway)
##
# Save transcript for each file in Prosodylab-aligner format

for sentence in tqdm(uniq_sentences):
    fn = sentence_to_fn(sentence, ext=".lab")
    with open(fn, 'w') as f:
        f.write(sentence)

##
from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav
from IPython.display import Audio

# download and load all models
preload_models()
##
# generate audio from text
import re

text_prompt = 'Most of them were twenty or so.'
audio_array = generate_audio(text_prompt, history_prompt="v2/en_speaker_9")
fn = sentence_to_fn(text_prompt)
write_wav(fn, SAMPLE_RATE, audio_array)
print(fn)
Audio(audio_array, rate=SAMPLE_RATE)
