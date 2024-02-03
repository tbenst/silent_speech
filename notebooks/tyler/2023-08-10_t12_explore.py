##
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

class SpeechDataset(torch.utils.data.Dataset):   
    def __init__(self, X, y, z=None, toggle_phones = False):
        '''Inputs are:
            X (list of 2D arrays) - entries are time x channels of neural activity
            y (list of 1D arrays) - entries are integer-encoded target strings (character or phoneme)
            z (list of ints)      - entries are session IDs'''
        
        assert len(X) == len(y), 'Target and predictor lists must be same length'
        if z is not None:
            assert len(X) == len(z), 'Target and session ID lists must be same length'
            
        self.X = X
        self.y = y
        self.z = z
        self.num_features   = self.X[0].shape[1]
        self.toggle_phones  = toggle_phones
        self.text_transform = TextTransform(toggle_phones = self.toggle_phones)
        self.smoothed       = False
        
    def __len__(self):
        return len(self.X)

    #@lru_cache
    def __getitem__(self, index):        
        if self.z is not None:
            return (self.X[index], self.y[index], self.z[index])
        else:
            return (self.X[index], self.y[index])
        
    def smooth_data(self, sigma):
        if not self.smoothed and sigma > 0:
            for idx in range(len(self.X)):
                self.X[idx] = gaussian_filter1d(self.X[idx], sigma = sigma, causal=True, axis = 0)
            self.smoothed = True
        else:
            print('Warning: data already smoothed. Skipping...')
    
    
    def collate_fn(self, batch):
        '''
        Padds batch of variable length
        '''
        # get sequence lengths
        lengths = torch.tensor([t[0].shape[0] for t in batch])
        # pad
        batch_x = [torch.tensor(t[0]) for t in batch]
        batch_x = torch.nn.utils.rnn.pad_sequence(batch_x, batch_first=True)
        
        datas = dict()
        datas['neural']           = batch_x
        datas['text_int']         = [torch.tensor(t[1]) for t in batch]
        datas['text_int_lengths'] = torch.tensor([ex[ex != 0].shape[0] for ex in datas['text_int']])
        datas['lengths']          = lengths
        if self.z is not None:
            datas['session_ids'] = torch.tensor([t[2] for t in batch])
        else:
            datas['session_ids'] = None
        
        return datas


def convertNumbersToStrings(sentence):
    
    output_sentence = []
    for word in sentence.split():
        if word.isdigit():
            output_sentence.append(numToWords(word))
        else:
            output_sentence.append(word)
    output_sentence = ' '.join(output_sentence)

    return output_sentence


class TextTransform(object):
    def __init__(self, toggle_phones = False):
        self.togglePhones     = toggle_phones
        
        self.transformation   = jiwer.Compose([jiwer.RemovePunctuation(), jiwer.ToLowerCase()])
        self.replacement_dict = {}
        
        if self.togglePhones:
            self.chars = [
                    'AA', 'AE', 'AH', 'AO', 'AW',
                    'AY', 'B',  'CH', 'D', 'DH',
                    'EH', 'ER', 'EY', 'F', 'G',
                    'HH', 'IH', 'IY', 'JH', 'K',
                    'L', 'M', 'N', 'NG', 'OW',
                    'OY', 'P', 'R', 'S', 'SH',
                    'T', 'TH', 'UH', 'UW', 'V',
                    'W', 'Y', 'Z', 'ZH', '|'
                ]

        else:
            self.chars = [x for x in string.ascii_lowercase + '|']         

    def clean_text(self, text):
        if self.togglePhones:
            raise NotImplementedError("Use preprocessed integer encoding for phone data")
        else:
            #text = unidecode(text)
            #text = text.replace('-', ' ')
            #text = text.replace(':', ' ')
            text = self.transformation(text)
            text = convertNumbersToStrings(text)
        
        return text

    def text_to_int(self, text):
        text = self.clean_text(text)
        if self.togglePhones:
            text = [x.replace(' ', '|') for x in text]
        else:
            text = text.replace(' ', '|')
        return [self.chars.index(c) for c in text]

    def int_to_text(self, ints):
        text = ''.join(self.chars[i] for i in ints)
        text = text.replace('|', ' ').lower()
        return text
##
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
##
import scipy

toggle_phones = True

# loading data into CPU, should work but takes ~ 6-10 minutes:
datasets  = list()
transform = TextTransform(toggle_phones=toggle_phones)

for files in [train_files, test_files]:
    mat = [scipy.io.loadmat(x) for x in files]
    X   = [trl['tx1'] for trl in mat]
    
    if toggle_phones:
        y = [trl['phoneme_ints'][trl['phoneme_ints'] != 0] for trl in mat]
    else:
        y = [transform.text_to_int(trl['text'][0]) for trl in mat]
        
    z = [session_mapping[f.split('sessions/')[1].split('/')[0]] for f in files]
    datasets.append(SpeechDataset(X, y, z, toggle_phones=toggle_phones))
    del mat, X, y
    

trainset = datasets[0]
testset  = datasets[1]


print(len(trainset))
print(len(testset))
##
from torchaudio.functional import edit_distance


def addRandomWalk(timeseries, strength):
    '''Apply mean drift to timeseries data using autoregressive noise. Inputs are:
    
        timeseries (batch x time x features) - data input
        strength (float)                     - noise standard deviation '''
    
    nBatch, nTime, nChannels = timeseries.shape
    noise                    = torch.zeros(timeseries.shape)
    noise[:, 0, :]           = torch.normal(torch.zeros((nChannels)), strength) 
    
    for t in range(1, nTime):
        noise[:, t, :] = noise[:, t-1, :] + torch.normal(torch.zeros((nBatch, nChannels)), strength)  
        
    return timeseries + noise

def addWhiteNoise(timeseries, strength):
    '''Apply IID gaussian noise to timeseries data. Inputs are:

    timeseries (batch x time x features) - data input
    strength (float)                     - noise standard deviation '''

    noise = torch.normal(torch.zeros((timeseries.shape)), strength)    
    return timeseries + noise

def addOffset(timeseries, strength):
    '''Add constant offsets to timeseries data. Inputs are: 
    
        timeseries (batch x time x features) - data input
        strength (float)                     - offset standard deviation '''

    nBatch, nTime, nChannels = timeseries.shape
    offset = torch.normal(torch.zeros((nBatch, nChannels)), strength)  
    return timeseries + offset[:, None, :]

def addNoise(timeseries, offset_strength = 0, whitenoise_strength = 0, randomwalk_strength = 0):  
    '''Interface function for adding various noise types to data.'''
    
    if offset_strength > 0:
        timeseries = addOffset(timeseries, strength = offset_strength)
    if whitenoise_strength > 0:
        timeseries = addWhiteNoise(timeseries, strength = whitenoise_strength)
    if randomwalk_strength > 0:
        timeseries = addRandomWalk(timeseries, strength = randomwalk_strength) 
        
    return timeseries


def stripLeadingTrailing(arr, val):
    '''Remove leading and trailing values from array. E.g.:
        
        > x = [0, 0, 1, 2, -1, 5, 's', 0]
        > stripLeadingTrailing(x) 
        > [1, 2, -1, 5, 's']
    '''
    
    # if arr is just a sequence of <val>, return a singleton
    unique_vals = torch.unique(arr)
    if len(unique_vals) == 1 and unique_vals[0] == val:
        return torch.tensor([val])
    
    else:
        idx = 0
        while arr[idx] == val:
            idx += 1

        arr = arr[idx:]

        # trailing values
        idx = -1
        while arr[idx] == val:
            idx -= 1

        arr = arr[:idx]
    
        return arr
    
    
def buildCTCDecoder(testset, use_lm):
    
    if testset.toggle_phones:
        #lexicon_file = 'cmudict.txt'
        #lexicon_file = '/oak/stanford/groups/shenoy/fwillett/speech/cmudict-0.7b.txt'
        lexicon_file = None
    else:
        lexicon_file = os.path.join(FLAGS.lm_directory, 'lexicon_graphemes_noApostrophe.txt')
        
    if use_lm:
        lm = os.path.join(FLAGS.lm_directory, '4gram_lm.bin')
    else:
        lm = None

    decoder = ctc_decoder(
       lexicon     = lexicon_file,
       tokens      = ['_'] + [x.lower() for x in testset.text_transform.chars],
       lm          = lm, 
       blank_token = '_',
       sil_token   = '|',
       nbest       = 1,
       lm_weight   = 2, 
       #word_score  = -3,
       #sil_score   = -2,
       beam_size   = 50  # SET TO 150 during inference
    )
    
    return decoder



def test(model, testset, device, use_lm = False):
    model.eval()
    
    decoder = buildCTCDecoder(testset, use_lm)
    
    dataloader  = torch.utils.data.DataLoader(testset, batch_size=1, collate_fn=testset.collate_fn)
    charTargets = []
    charPreds   = []
    wordTargets = []
    wordPreds   = []
    with torch.no_grad():
        for i, example in enumerate(dataloader):
            X_raw   = example['neural'][0].unsqueeze(0).to(device)
            session = example['session_ids'][0].unsqueeze(0).to(device)
            pred    = F.log_softmax(model(X_raw, session), -1).cpu()
            
            beam_results = decoder(pred)
            pred_int     = beam_results[0][0].tokens
            target_int   = example['text_int'][0][:example['text_int_lengths'][0]]
            
            pred_int   = stripLeadingTrailing(pred_int, 40)
            target_int = stripLeadingTrailing(target_int, 40)

            
            charPreds.append(pred_int)
            charTargets.append(target_int)
            
            if not testset.toggle_phones:
                pred_text    = ' '.join(beam_results[0][0].words).strip().lower()
                target_text  = testset.text_transform.int_to_text(example['text_int'][0])
                
                wordPreds.append(pred_text)
                wordTargets.append(target_text)

            #pred_text    = testset.text_transform.int_to_text(pred_int)
            #target_text  = testset.text_transform.clean_2(example['text'][0][0])
            
            if i < 0:
                print('Prediction: ', pred_int)
                print('Target: ', target_int)
            if i > 100:
                # for now only measure 100 sentence for validation
                break
                
    if testset.toggle_phones:
        # phone/character error rate computed across all examples
        cer  = 0
        lens = 0
        for x, y in zip(charTargets, charPreds):
            cer  += edit_distance(x, y)
            lens += len(x)
            
        cer /= lens
        wer = 0
        
    else:
        cer =  0 #jiwer.cer(charTargets, charPreds)  
        wer = jiwer.wer(wordTargets, wordPreds)
        
    return cer, wer