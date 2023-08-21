##
# Dynamic time warping of TTS audio & phoneme textgrids onto T12 audio envelopes.

##
import os, sys, glob
import torch
from tqdm import tqdm
import numpy as np
from textgrids import TextGrid
import torch.nn as nn, scipy
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchmetrics
import datetime
import zarr
from scipy.spatial.distance import cdist
import jiwer
import sklearn
from unidecode import unidecode
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import librosa
import re, sys, pickle
import string
from collections import defaultdict
import scipy
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(SCRIPT_DIR)
from dataloaders import persist_to_file
from functools import lru_cache
from scipy.io.wavfile import write as write_wav
from IPython.display import Audio
from align import align_from_distances
from helpers import sentence_to_fn
from data_utils import read_phonemes, mel_spectrogram

import bottleneck

from bark import SAMPLE_RATE as TTS_SAMPLE_RATE
SCRIPT_DIR = "/home/tyler/code/silent_speech"
normalizers_file = os.path.join(SCRIPT_DIR, "normalizers.pkl")
mfcc_norm, emg_norm = pickle.load(open(normalizers_file,'rb'))

##
T12_dir = '/data/data/T12_data'
datadir        = os.path.join(T12_dir, 'competitionData')
sentences_dir = os.path.join(T12_dir, 'sentences')
TTS_dir = os.path.join(os.path.dirname(datadir), 'synthetic_audio', 'TTS')

# format like `t12.2022.04.28.mat`
train_files = glob.glob(datadir + '*/train/*')
test_files  = glob.glob(datadir + '*/test/*')

# format like `t12.2022.04.28_sentences.mat`
sentences_files = glob.glob(sentences_dir + '/*')

# map each competitionData mat file to its corresponding sentences mat file
competition_file_mapping = {}
days = []
for train_file in train_files + test_files:
    day = train_file.split('.mat')[0].split('t12.')[1]
    sentences_file = [sf for sf in sentences_files if day in sf][0]
    competition_file_mapping[train_file] = sentences_file
    days.append(day)

unique = np.unique(days)
session_mapping = dict(zip(unique, np.arange(len(unique))))
print('Unique days:', len(session_mapping.keys()))
##
@persist_to_file(os.path.join(os.path.dirname(datadir), "sentence_mapping_per_file.pkl"))
def get_competition_to_sentence_mapping_per_file(train_files, test_files, competition_file_mapping):
    "Map competitionData files to sentences files."
    sentence_mapping_per_file = {}
    for tf in tqdm((train_files + test_files)):
        mat = scipy.io.loadmat(tf)
        mat2 = scipy.io.loadmat(competition_file_mapping[tf])
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

# map indices in competitionData mat files to indices in sentences mat files
# e.g.
# mat = '/data/data/T12_data/competitionData/test/t12.2022.07.21.mat'
# competition_file_mapping[mat] == '/data/data/T12_data/sentences/t12.2022.07.21_sentences.mat'
# sentence_mapping_per_file[mat][0] == 400
# means that the first sentence in the competitionData mat file corresponds to the 400th sentence in the sentences mat file
competition_to_sentence_mapping_per_file = get_competition_to_sentence_mapping_per_file(train_files, test_files, competition_file_mapping)

##
# create dictionary with sentence as key, and list of tuples (file, index) as value
# only use competitionData mat files
# e.g.
# sentence_mapping['hello world'] == [('/data/data/T12_data/competitionData/test/t12.2022.07.21.mat', 400)]
@persist_to_file(os.path.join(os.path.dirname(datadir), "sentence_mapping.pkl"))
def create_sentence_mapping(competition_to_sentence_mapping_per_file, competition_file_mapping):
    sentence_mapping = {}
    for k, v in tqdm(competition_to_sentence_mapping_per_file.items()):
        mat = scipy.io.loadmat(k)
        mat2 = scipy.io.loadmat(competition_file_mapping[k])
        for i, idx in enumerate(v):
            sentence = mat['sentenceText'][i]
            sentence = sentence.rstrip()  # strip whitespace at end
            # sentence = unidecode(sentence)  # remove accents
            # sentence = re.sub(r'[^a-zA-Z0-9\s]', '', sentence)  # remove punctuation
            # sentence = sentence.lower()
            if sentence not in sentence_mapping:
                sentence_mapping[sentence] = []
            sentence_mapping[sentence].append((k, idx))
    return sentence_mapping

sentence_mapping = create_sentence_mapping(competition_to_sentence_mapping_per_file, competition_file_mapping)
##
repeated_utterances = {k: v for k, v in sentence_mapping.items() if len(v) > 1}

##
# load all mat files into dict of filename: mat
mat_files = {}
for f in tqdm(sorted(train_files + test_files + sentences_files)):
    mat_files[f] = scipy.io.loadmat(f)
##
# speaking mode of each mat file
speaking_modes = {}
for f, mat in tqdm(mat_files.items()):
    if not 'competitionData' in f:
        mode = mat['speakingMode'][0]
        if mode == 'attempted nonvocal speaking':
            speaking_modes[f] = "silent"
        elif mode == 'attempted speaking':
            speaking_modes[f] = "vocalized"
        else:
            raise Exception(f"Unknown speakingMode: {mode} from file {f}")
        
for comp_file, sentence_file in tqdm(competition_file_mapping.items()):
    speaking_modes[comp_file] = speaking_modes[sentence_file]
    
speaking_modes
##
# what fraction of silent sentences are also vocalized?
silent_count = 0
vocalized_count = 0
parallel_count = 0
for sentence, files in tqdm(sentence_mapping.items()):
    is_silent = False
    is_vocalized = False
    for f, idx in files:
        if speaking_modes[f] == "silent":
            is_silent = True
        elif speaking_modes[f] == "vocalized":
            is_vocalized = True
        else:
            raise Exception(f"Unknown speakingMode: {speaking_modes[f]} from file {f}")
    if is_silent and is_vocalized:
        parallel_count += 1
    elif is_silent:
        silent_count += 1
    elif is_vocalized:
        vocalized_count += 1
print(f"{silent_count=}, {vocalized_count=}, {parallel_count=}")
print(f"Percent of silent sentences that are also vocalized: {parallel_count / silent_count* 100:.2f}%")
print(f"Number of repeated utterances: {len(repeated_utterances)}")
##
# number of sentences in train and test
print(f"Number of utterances in train: {sum([len(v) for k, v in competition_to_sentence_mapping_per_file.items() if 'competitionData/train' in k])}")
print(f"Number of utterances in test: {sum([len(v) for k, v in competition_to_sentence_mapping_per_file.items() if 'competitionData/test' in k])}")
num_utterances_in_both = 0
for k,v in repeated_utterances.items():
    has_train = False
    has_test = False
    for f, idx in v:
        if 'competitionData/train' in f:
            has_train = True
        elif 'competitionData/test' in f:
            has_test = True
    if has_train and has_test:
        num_utterances_in_both += 1
    
print(f"Number of sentences in both train & test: {num_utterances_in_both}")
##
# align competitionData neural data to sentences neural data
def window_middle_signal(signal):
    """Extract the middle 50% of a signal."""
    length = len(signal)
    start_idx = int(0.25 * length)
    end_idx = int(0.75 * length)
    return signal[start_idx:end_idx]

def compute_offset_1d(signal1, reference_signal):
    # Window the signal to get the middle 50%
    # windowed_signal1 = window_middle_signal(signal1)

    # Compute 1D cross-correlation
    # cross_corr_1d = np.correlate(reference_signal, windowed_signal1, mode='full')
    cross_corr_1d = np.correlate(reference_signal, signal1,  mode='full')
    
    # Get the index of the peak of the 1D cross-correlation
    idx_peak = np.argmax(cross_corr_1d)
    print(f"{idx_peak}")
    
    # Compute the offset. The offset represents how much to shift signal1
    # to align it with the reference signal.
    offset = (len(signal1) - 1) - idx_peak

    return offset
# for mat_file, sentenceIdxs in tqdm(list(competition_to_sentence_mapping_per_file.items())[10:]):
for mat_file, sentenceIdxs in tqdm(list(competition_to_sentence_mapping_per_file.items())[0:]):
    mat = mat_files[mat_file]
    sentence_file = competition_file_mapping[mat_file]
    sentence_mat = mat_files[sentence_file]
    comp_mat = mat_files[mat_file]
    go_cues = sentence_mat['goTrialEpochs']
    for compIdx, sentenceIdx in enumerate(sentenceIdxs):
        # TODO: why is this failing..?
        try:
            sentence_dat = sentence_mat['spikePow'][go_cues[sentenceIdx,0]:go_cues[sentenceIdx,1]]
            comp_dat = comp_mat['spikePow'][0,compIdx]
            assert np.all(np.isclose(sentence_dat, comp_dat))
        except Exception as e:
            print(f"Neural data does not match for sentence {sentenceIdx} in file {sentence_file} and compIdx {compIdx} in file {mat_file}")
            offset = compute_offset_1d(sentence_dat.mean(axis=1), comp_dat.mean(axis=1))
            # raise e
            sentence_dat = sentence_dat[offset:]
            # known issue sadly, two sentences can't be fixed like this
            # assert np.all(np.isclose(sentence_dat, comp_dat)), f"still doesn't match with offset {offset}"
            # print(f"fixed alignment to {mat_file} with offset {offset}")

    # break
    #     go_cue = go_cues[idx]
    #     neural_data = mat['spikePow']
# offset = 23
##
# show the bad alignment / skipped samples
plt.plot(comp_dat.mean(axis=1)[0:])
# plt.plot(sentence_dat.mean(axis=1)[offset:])
plt.plot(sentence_dat.mean(axis=1)[0:])
plt.legend(["competitionData", "sentences"])
plt.title(f"Neural data for sentence {sentenceIdx} in sentences and sentence {compIdx} in file\n{mat_file}")
plt.ylabel("mean spike power")
plt.xlabel("timestep (20ms)")
##

def compute_audio_envelope(audio, sample_rate=44100, frame_size_ms=20, hop_size_ms=20):
    """
    Compute the audio envelope using the FFT magnitude spectrum.
    
    Parameters:
    - audio: 1D numpy array of audio samples.
    - sample_rate: Sampling rate of the audio (default is 44.1kHz).
    - frame_size_ms: Size of the frames in milliseconds for FFT computation (default is 20ms).
    - hop_size_ms: Step size between frames in milliseconds (default is 10ms).
    
    Returns:
    - envelope: Audio envelope as a numpy array.
    """

    # Calculate frame size and hop size in samples
    frame_size_samples = int(sample_rate * frame_size_ms / 1000)
    hop_size_samples = int(sample_rate * hop_size_ms / 1000)

    # Apply a window function (Hann window)
    window = np.hanning(frame_size_samples)

    # Compute the number of frames
    num_frames = 1 + (len(audio) - frame_size_samples) // hop_size_samples

    # Compute the envelope
    envelope = np.zeros(num_frames)
    for i in range(num_frames):
        start_idx = i * hop_size_samples
        end_idx = start_idx + frame_size_samples
        frame = audio[start_idx:end_idx] * window

        # Compute magnitude spectrum of the frame
        magnitude_spectrum = np.abs(np.fft.rfft(frame))

        envelope[i] = np.mean(magnitude_spectrum)

    return envelope

i = 0
for mat_file, sentenceIdxs in tqdm(competition_to_sentence_mapping_per_file.items()):
    mat = mat_files[mat_file]
    sentence_file = competition_file_mapping[mat_file]
    print(sentence_file)
    sentence_mat = mat_files[sentence_file]
    i += 1
    if i == 21:
        break


blockIdx = 0
aud = np.squeeze(sentence_mat['audio'][0,blockIdx])
aud = compute_audio_envelope(aud, sample_rate=30000, frame_size_ms=20, hop_size_ms=20)

fig, axs = plt.subplots(2, 1, figsize=(10, 10))
axs[0].plot(sentence_mat['audioEnvelope'][:10000])
axs[0].set_title('Reference Audio Envelope')
axs[1].plot(aud[:10000])
axs[1].set_title('Computed Audio Envelope')
fig.suptitle(sentence_file)
plt.show()
##
S = 300
N = 450
fig, axs = plt.subplots(2, 1, figsize=(10, 10))
axs[0].plot(sentence_mat['audioEnvelope'][S:N])
axs[0].set_title('Reference Audio Envelope')
axs[1].plot(aud[S:N])
axs[1].set_title('Computed Audio Envelope')
plt.show()
##
S = 0
S = 220
N = 400
# N = 10000
# num_mels = 42
num_mels = 80
mspec = mel_spectrogram(torch.tensor(sentence_mat['audio'][0,blockIdx], dtype=torch.float32),
                              # n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax
                                2048, num_mels, 30000, 30000//50, 30000//25, 0, 8000, center=False)

mspec = mspec.squeeze().log2().numpy()
# mspec = mfcc_norm.normalize(mspec.T).T # kinda weird

fig, axs = plt.subplots(2, 1, figsize=(10, 10))
axs[0].imshow(sentence_mat['audioFeatures'][S:N].T, aspect='auto', origin='lower')
axs[0].set_title('Audio Features')
axs[0].set_ylabel("Mel Frequency Cepstral Coefficients")
axs[1].imshow(mspec[:,S:N], aspect='auto', origin='lower')
axs[1].set_title('Computed mel Spectrogram')
axs[1].set_ylabel("Mel Frequency Cepstral Coefficients")
axs[1].set_xlabel("# frames (20ms each)")
plt.show()

##
# concatenate all audio arrays
# audio = np.concatenate([np.squeeze(sentence_mat['audio'][0,blockIdx]) for blockIdx in range(len(sentence_mat['audio'][0]))])
# audio_envelope = compute_audio_envelope(audio, sample_rate=30000, frame_size_ms=20)
# mspec = mel_spectrogram(torch.tensor(audio[None], dtype=torch.float32),
#                               # n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax
#                                 2048, 42, 30000, 30000//50, 30000//50, 0, 8000, center=False).squeeze()

def load_TTS_data(sentence, directory=TTS_dir, ms_per_frame=10):
    """Load corresponding TTS audio and phoneme labels for a given sentence."""
    tts_audio_path = sentence_to_fn(sentence, directory, ext=".wav")
    textgrid = sentence_to_fn(sentence, directory, ext=".TextGrid")
    # even though neural data at 20ms bins, run alignment at 10ms bins
    phonemes = read_phonemes(textgrid, ms_per_frame=ms_per_frame)
    audio, sample_rate = librosa.load(tts_audio_path)
    return audio, phonemes, sample_rate

# TODO: get phoneme labels for each timestep T of competitionData

# Main loop for processing sentences
# for each vocalized sentence, get audio spectrogram
# read parallel TTS audio + phoneme labels
# Run DTW on spectrograms of each vocalized sentence (audio) to TTS audio
# accumulate phoneme labels for each timestep of neural data

def go_cue_to_block_and_rel_index(go_cue_idx, block_start_idxs):
    "Given a go_cue index, return the corresponding block index and relative index."
    block_idx = np.where(block_start_idxs <= go_cue_idx)[0][-1]
    rel_idx = go_cue_idx - block_start_idxs[block_idx]
    return block_idx, rel_idx

# compare to audio envelope to check if correct
idx = 28
block_start_idxs = np.concatenate([[0], 1 + np.where(np.diff(sentence_mat['blockNum'][:,0]))[0]])
go_cue_idx = sentence_mat['goTrialEpochs'][idx,0]
block_idx, relIdx = go_cue_to_block_and_rel_index(go_cue_idx, block_start_idxs)

plt.plot(sentence_mat['audioEnvelope'][go_cue_idx:go_cue_idx+150])
aud = sentence_mat['audio'][0,block_idx][0]
aud = compute_audio_envelope(aud, sample_rate=30000, frame_size_ms=20)
aud = aud[relIdx:relIdx+150]
print(aud.shape)
plt.plot(aud)
# not matching anymore, prob just deleted some things..?
plt.legend(["audioEnvelope reference", "audio check"])
##
# check audio length
for mat_file in sentences_files:
    sentence_mat = mat_files[mat_file]
    T = sentence_mat['spikePow'].shape[0]
    neural_seconds = T * 20 / 1000
    nAudio = np.sum([m[0].shape[0] for m in sentence_mat['audio'][0]])
    audio_seconds = nAudio / 30000

    if np.abs(neural_seconds - audio_seconds) > 0.5:
        print(f"==== {mat_file} ====")
        print(f"{T=}\n{neural_seconds=}")
        print(f"{nAudio=}\n{audio_seconds=}")
        break
# sentences/t12.2022.06.28_sentences.mat audio block 5 has length of zero
##
plt.plot()

##
########################## THIS IS THE KEY FUNCTION ########################################
# save a npz file

# iterate sentences mat files
ms_per_frame = 20
nframes_per_sec = 1000 // ms_per_frame
mat_sentences = {}
mat_block = {}
mat_mspecs = {}
mat_tts_mspecs = {}
mat_tts_phonemes = {}
mat_aligned_mspecs = {}
mat_aligned_phonemes = {}
mat_spikePow = {}
mat_tx1 = {}
mat_tx2 = {}
mat_tx3 = {}
mat_tx4 = {}
mat_speakingMode = {}
mat_audioEnvelope = {}
mat_dataset_partition = {}


total_T = 0
n_sentences = 0
for mat_file in sentences_files:
    sentence_mat = mat_files[mat_file]
    total_T += sentence_mat['spikePow'].shape[0]
    n_sentences += len(sentence_mat['sentences'])
# npz = np.load('/data/data/T12_data/synthetic_audio/2023-08-20_T12_dataset.npz',
#               allow_pickle=True)
# np.sum([n.shape[0] for n in npz['spikePow']]) / len(npz['spikePow'])
# 311.06
# rolling window of 20 sentences.
# Avg sentence is: 312 during goCue and 555.8 in total on average
# 20 sentences rolling z-score idea from Willett et al. 2023
# my implementation a bit different as constant in time steps
window_size = int(np.ceil(total_T/n_sentences)) * 20
##

def moving_mean(x, window):
    "For T x N matrix, compute the rolling mean over window timesteps."
    x_mean = x.unfold(0,window,1).mean(dim=2)
    # use first mean for first window-1 timesteps
    # technically acausal for first window-1 timesteps
    x[:window-1] = x_mean[0]
    x[window-1:] = x_mean
    return x

def moving_std(x, window):
    "For T x N matrix, compute the rolling mean over window timesteps."
    x_mean = x.unfold(0,window,1).std(dim=2)
    # use first mean for first window-1 timesteps
    # technically acausal for first window-1 timesteps
    x[:window-1] = x_mean[0]
    x[window-1:] = x_mean
    return x

def moving_zscore(x, window, eps=1e-6):
    "For T x N matrix, compute the rolling z-score over window timesteps."
    x_mean = moving_mean(x, window)
    x_std = moving_std(x, window)
    zscored = (x - x_mean) / (x_std + eps)
    return zscored
    
# movet = moving_mean(sentence_mat['spikePow'], window_size)

saw_bad_audio = False
# for mat_file in tqdm(sentences_files):
for mat_file in sentences_files:
    mat_name = os.path.split(mat_file)[-1]
    if not speaking_modes[mat_file] == "vocalized":
        mat_speakingMode[mat_name] = "silent"
    else:
        mat_speakingMode[mat_name] = "vocalized"
        sentence_mat = mat_files[mat_file]
        block_start_idxs = np.concatenate([
            [0],
            1 + np.where(np.diff(sentence_mat['blockNum'].squeeze()))[0]
        ])
        block_end_idxs = np.concatenate([
            block_start_idxs[1:],
            [len(sentence_mat['blockNum'].squeeze())]
        ])

        # last_block_idx = sentence_mat['blockNum'][:,0][-1] # test set (doesn't always start at 1 / skips numbers)
        last_block_idx = len(sentence_mat['blockList']) - 1 # last block is test set
        audio_block = []
        for i in range(len(sentence_mat['audio'][0])):
            aud = sentence_mat['audio'][0,i][0]
            if aud.shape[0] == 0:
                assert not saw_bad_audio, "there should only be one..."
                saw_bad_audio = True
                audio_block.append(None)
            else:
                aud = librosa.util.buf_to_float(aud)
                aud = np.clip(aud, -1, 1)
                audio_block.append(aud)
        # audio_block = [librosa.util.buf_to_float(sentence_mat['audio'][0,i][0]) for i in range(len(sentence_mat['audio'][0]))]
        # volume_block = [compute_audio_envelope(aud, sample_rate=30000, frame_size_ms=20) for aud in audio_block]
        
        volume_block = []
        for aud in audio_block:
            if aud is None:
                volume_block.append(None)
            else:
                volume_block.append(compute_audio_envelope(aud, sample_rate=30000, frame_size_ms=20))
                
        # for a in audio_block:
        #     print(f"AB min: {np.min(a)}, max: {np.max(a)}")
        mspec_block = []
        for aud in audio_block:
            if aud is None:
                mspec_block.append(None)
            else:
                au = torch.tensor(aud[None], dtype=torch.float32).cuda().clip(-1,1)
                mspec_block.append(mel_spectrogram(au,
                    # n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax
                    2048, 80, 30000, 30000//nframes_per_sec, 30000//(nframes_per_sec//2), 0, 8000, center=False).squeeze().T)

        # mspec_block = [mel_spectrogram(torch.tensor(aud[None], dtype=torch.float32).cuda(),
        #     # n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax
        #     2048, 80, 30000, 30000//nframes_per_sec, 30000//(nframes_per_sec//2), 0, 8000, center=False).squeeze()
        #                for aud in audio_block]
    
    assert len(audio_block) < 100
    # always append
    sentences = []
    spikePow = []
    tx1 = []
    tx2 = []
    tx3 = []
    tx4 = []
    dataset_partition = []
    block = []
    
    # try to append
    mspecs = []
    tts_mspecs = []
    tts_phonemes = []
    aligned_mspecs = []
    aligned_phonemes = []
    audioEnvelope = []


    session_spikePow = torch.from_numpy(sentence_mat['spikePow']).float().cuda()
    session_tx1 = torch.from_numpy(sentence_mat['tx1']).float().cuda()
    session_tx2 = torch.from_numpy(sentence_mat['tx2']).float().cuda()
    session_tx3 = torch.from_numpy(sentence_mat['tx3']).float().cuda()
    session_tx4 = torch.from_numpy(sentence_mat['tx4']).float().cuda()
    
    spikePow_mean = moving_mean(session_spikePow, window_size)
    tx1_mean = moving_mean(session_tx1, window_size)
    tx2_mean = moving_mean(session_tx2, window_size)
    tx3_mean = moving_mean(session_tx3, window_size)
    tx4_mean = moving_mean(session_tx4, window_size)
    spikePow_std = moving_std(session_spikePow, window_size) + 1
    tx1_std = moving_std(session_tx1, window_size) + 1
    tx2_std = moving_std(session_tx2, window_size) + 1
    tx3_std = moving_std(session_tx3, window_size) + 1
    tx4_std = moving_std(session_tx4, window_size) + 1
    
    # session_spikePow = session_spikePow.cpu().numpy()
    # session_tx1 = session_tx1.cpu().numpy()
    # session_tx2 = session_tx2.cpu().numpy()
    # session_tx3 = session_tx3.cpu().numpy()
    # session_tx4 = session_tx4.cpu().numpy()

    # TODO: try taking sqrt before zscore
    
    for sentenceIdx in tqdm(range(len(sentence_mat['sentences']))):
        sentence = sentence_mat['sentences'][sentenceIdx][0][0]
        sentence = sentence.rstrip()
        go_cue_idx = sentence_mat['goTrialEpochs'][sentenceIdx]
        
        sentence_spikePow = session_spikePow[go_cue_idx[0]:go_cue_idx[1]]
        sentence_tx1 = session_tx1[go_cue_idx[0]:go_cue_idx[1]]
        sentence_tx2 = session_tx2[go_cue_idx[0]:go_cue_idx[1]]
        sentence_tx3 = session_tx3[go_cue_idx[0]:go_cue_idx[1]]
        sentence_tx4 = session_tx4[go_cue_idx[0]:go_cue_idx[1]]
        
        wi = go_cue_idx[1] - 1
        sentence_spikePow = (sentence_spikePow - spikePow_mean[wi]) / spikePow_std[wi]
        sentence_tx1 = (sentence_tx1 - tx1_mean[wi]) / tx1_std[wi]
        sentence_tx2 = (sentence_tx2 - tx2_mean[wi]) / tx2_std[wi]
        sentence_tx3 = (sentence_tx3 - tx3_mean[wi]) / tx3_std[wi]
        sentence_tx4 = (sentence_tx4 - tx4_mean[wi]) / tx4_std[wi]
        
        sentence_spikePow = sentence_spikePow.cpu().numpy()
        sentence_tx1 = sentence_tx1.cpu().numpy()
        sentence_tx2 = sentence_tx2.cpu().numpy()
        sentence_tx3 = sentence_tx3.cpu().numpy()
        sentence_tx4 = sentence_tx4.cpu().numpy()
        
        
        sentences.append(sentence)
        spikePow.append(sentence_spikePow)
        tx1.append(sentence_tx1)
        tx2.append(sentence_tx2)
        tx3.append(sentence_tx3)
        tx4.append(sentence_tx4)
        
        block.append(sentence_mat['blockNum'][go_cue_idx[0]])
        
        # note block_idx here starts at 0, but doesn't in sentence_mat['blockList']
        block_idx, startIdx = go_cue_to_block_and_rel_index(go_cue_idx[0], block_start_idxs)
        block_idx2, stopIdx = go_cue_to_block_and_rel_index(go_cue_idx[1], block_start_idxs)
        if stopIdx == 0:
            block_idx2, stopIdx = go_cue_to_block_and_rel_index(go_cue_idx[1]-1, block_start_idxs)
            stopIdx += 1
        assert block_idx == block_idx2
        
        if block_idx == last_block_idx:
            dataset_partition.append("test")
        else:
            dataset_partition.append("train")
        
        try:
            tts_audio, tts_phones, sample_rate = load_TTS_data(sentence, ms_per_frame=ms_per_frame)
        except FileNotFoundError:
            print("Skipping as could not read file (prob TextGrid) for sentence: ", sentence)
            mspecs.append(None)
            tts_mspecs.append(None)
            tts_phonemes.append(None)
            aligned_mspecs.append(None)
            aligned_phonemes.append(None)
            audioEnvelope.append(None)
            continue
        
        # print(f"TTS min audio: {np.min(tts_audio)}, max audio: {np.max(tts_audio)}")
        tts_volume = compute_audio_envelope(tts_audio, sample_rate=sample_rate, frame_size_ms=20)
        tts_au = torch.tensor(tts_audio, dtype=torch.float32).cuda()[None].clip(-1,1)
        tts_mspec = mel_spectrogram(tts_au,
            2048, 80, sample_rate, sample_rate//nframes_per_sec, sample_rate//(nframes_per_sec//2), 0, 8000, center=False).squeeze().T
        
        tts_mspecs.append(tts_mspec.cpu().numpy())
        tts_phonemes.append(tts_phones)
        
        if mspec_block[block_idx] is None:
            # we're missing audio data
            audioEnvelope.append(None)
            mspecs.append(None)
            aligned_mspecs.append(None)
            aligned_phonemes.append(None)
            continue
        
        t12_mspec = mspec_block[block_idx][startIdx:stopIdx]
        t12_volume = volume_block[block_idx][startIdx:stopIdx]
        
        if speaking_modes[mat_file] == "vocalized":
            # finally, run dynamic time warping between t12_mspec and tts_mspec
            
            # good!
            # dists = cdist(t12_mspec.T, tts_mspec.T)
            dists = torch.cdist(t12_mspec, tts_mspec)
            
            # bad...
            # dists = 1 - torchmetrics.functional.pairwise_cosine_similarity(t12_mspec.T, tts_mspec.T).cpu().numpy()

            # okay...
            # dists = cdist(t12_volume[None].T, tts_volume[None].T)
            
            alignment = align_from_distances(dists.cpu().numpy())
            audioEnvelope.append(t12_volume)
            mspecs.append(t12_mspec.cpu().numpy())
            aligned_mspecs.append(tts_mspec[alignment].cpu().numpy())
            aligned_phonemes.append(tts_phones[alignment])
        else:
            audioEnvelope.append(None)
            mspecs.append(None)
            aligned_mspecs.append(None)
            aligned_phonemes.append(None)
        # raise Exception("stop here")
    
    mat_sentences[mat_name] = sentences
    mat_block[mat_name] = block
    mat_mspecs[mat_name] = mspecs
    mat_tts_mspecs[mat_name] = tts_mspecs
    mat_tts_phonemes[mat_name] = tts_phonemes
    mat_aligned_mspecs[mat_name] = aligned_mspecs
    mat_aligned_phonemes[mat_name] = aligned_phonemes
    mat_spikePow[mat_name] = spikePow
    mat_tx1[mat_name] = tx1
    mat_tx2[mat_name] = tx2
    mat_tx3[mat_name] = tx3
    mat_tx4[mat_name] = tx4
    mat_audioEnvelope[mat_name] = audioEnvelope
    mat_dataset_partition[mat_name] = dataset_partition

##
# save to Zarr
# flatten to 1D arrays of length num_sentences
num_sentences_per_mat = []
flat_block = []
flat_session = []
flat_dataset_partition = []
flat_sentences = []
flat_mspecs = []
flat_tts_mspecs = []
flat_tts_phonemes = []
flat_aligned_mspecs = []
flat_aligned_phonemes = []
flat_spikePow = []
flat_tx1 = []
flat_tx2 = []
flat_tx3 = []
flat_tx4 = []

for mat_file, v in mat_mspecs.items():
    nsentences = len(v)
    num_sentences_per_mat.append(nsentences)
    flat_session.extend([mat_file] * nsentences)
    
    flat_mspecs.extend(mat_mspecs[mat_file])
    
    assert len(mat_sentences[mat_file]) == nsentences
    flat_sentences.extend(mat_sentences[mat_file])
    assert(len(mat_block[mat_file]) == nsentences)
    flat_block.extend(mat_block[mat_file])
    assert len(mat_dataset_partition[mat_file]) == nsentences
    flat_dataset_partition.extend(mat_dataset_partition[mat_file])
    assert len(mat_tts_mspecs[mat_file]) == nsentences
    flat_tts_mspecs.extend(mat_tts_mspecs[mat_file])
    assert len(mat_tts_phonemes[mat_file]) == nsentences
    flat_tts_phonemes.extend(mat_tts_phonemes[mat_file])
    assert len(mat_aligned_mspecs[mat_file]) == nsentences
    flat_aligned_mspecs.extend(mat_aligned_mspecs[mat_file])
    assert len(mat_aligned_phonemes[mat_file]) == nsentences
    flat_aligned_phonemes.extend(mat_aligned_phonemes[mat_file])
    assert len(mat_spikePow[mat_file]) == nsentences
    flat_spikePow.extend(mat_spikePow[mat_file])
    assert len(mat_tx1[mat_file]) == nsentences
    flat_tx1.extend(mat_tx1[mat_file])
    assert len(mat_tx2[mat_file]) == nsentences
    flat_tx2.extend(mat_tx2[mat_file])
    assert len(mat_tx3[mat_file]) == nsentences
    flat_tx3.extend(mat_tx3[mat_file])
    assert len(mat_tx4[mat_file]) == nsentences
    flat_tx4.extend(mat_tx4[mat_file])

##
cur_date = datetime.datetime.now().strftime("%Y-%m-%d")
path = os.path.join(os.path.dirname(datadir), "synthetic_audio", f"{cur_date}_T12_dataset.npz")
# mdict = {
#     "session": session, "sentences": mat_sentences,
#     "mspecs": flat_mspecs, "aligned_mspecs": flat_aligned_mspecs, "aligned_phonemes": flat_aligned_phonemes,
#     "spikePow": flat_spikePow, "tx1": flat_tx1, "tx2": flat_tx2, "tx3": flat_tx3, "tx4": flat_tx4,
# }
mdict = {
    "session": flat_session, "dataset_partition": flat_dataset_partition, "sentences": flat_sentences,
    "block": flat_block,
    "mspecs": flat_mspecs, "tts_mspecs": flat_tts_mspecs, "tts_phonemes": flat_tts_phonemes,
    "aligned_tts_mspecs": flat_aligned_mspecs, "aligned_phonemes": flat_aligned_phonemes,
    "spikePow": flat_spikePow, "tx1": flat_tx1, "tx2": flat_tx2, "tx3": flat_tx3, "tx4": flat_tx4,
}

mdict_arr = {}
for k,v in mdict.items():
    try:
        mdict_arr[k] = np.array(v)
    except:
        # support ragged array
        mdict_arr[k] = np.array(v, dtype=object)
np.savez(path, **mdict_arr)

# may not work
# zarr.save_group(path, **mdict)

# Prob don't need to run script below here unless exploring data
print(f"Saved T12 dataset to {path}")
exit(0)
##
# spot check 6/28 since missing audio block 5
# not sure if okay or not
# DTW alignment is not great overall...
idx = -24
idx = -4
fig, axs = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
mat = mat_files['/data/data/T12_data/sentences/t12.2022.06.28_sentences.mat']
ae = mat['audioEnvelope']
start, stop = mat['goTrialEpochs'][idx]
axs[0].plot(ae[start:stop])
axs[0].set_title('audio volume')
axs[1].imshow(mat['audioFeatures'][start:stop].T, aspect='auto', origin='lower')
axs[1].set_title('audioFeatures')
axs[2].imshow(mat_mspecs['t12.2022.06.28_sentences.mat'][idx], aspect='auto', origin='lower')
axs[2].set_title('T12 mspec')
axs[3].imshow(mat_aligned_mspecs['t12.2022.06.28_sentences.mat'][idx], aspect='auto', origin='lower')
axs[3].set_title('aligned TTS mspec')
##
fig, axs = plt.subplots(3, 1, figsize=(10, 6), sharey=True)
axs[0].imshow(t12_mspec.cpu().numpy(), aspect='auto', origin='lower')
axs[0].set_title('t12 mspec')
axs[1].imshow(tts_mspec[alignment].cpu().numpy(), aspect='auto', origin='lower')
axs[1].set_title('aligned TTS mspec')
axs[2].imshow(tts_mspec.cpu().numpy(), aspect='auto', origin='lower')
axs[2].set_title('TTS mspec')
plt.tight_layout()
plt.show()
##
plt.plot(alignment)
plt.title("DTW alignment")
plt.xlabel("T12 index")
plt.ylabel("TTS index")
##
fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
axs[0].plot(tts_phonemes)
axs[0].set_title("TTS phonemes")
axs[0].set_ylabel("phoneme")
axs[1].imshow(tts_mspec, aspect='auto', origin='lower')
axs[1].set_ylabel("MFCC")
axs[1].set_xlabel("time (20ms)")
plt.show()
##
fig, axs = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
axs[0].imshow(t12_mspec, aspect='auto', origin='lower')
axs[0].set_ylabel("MFCC")
axs[0].set_title("T12 mspec")
axs[1].plot(tts_phonemes[alignment])
axs[1].set_title("T12 (aligned) phonemes")
axs[1].set_ylabel("phoneme")
axs[2].imshow(tts_mspec[alignment], aspect='auto', origin='lower')
axs[2].set_title('aligned TTS mspec')
axs[2].set_ylabel("MFCC")
axs[2].set_xlabel("time (20ms)")
plt.tight_layout()
plt.show()


##
fig, axs = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
axs[0].imshow(t12_mspec.cpu().numpy(), aspect='auto', origin='lower')
axs[0].set_ylabel("MFCC")
axs[0].set_title("T12 mspec")
axs[1].plot(mat_aligned_phonemes['/data/data/T12_data/sentences/t12.2022.05.05_sentences.mat'][-1])
axs[1].set_title("T12 (aligned) phonemes")
axs[1].set_ylabel("phoneme")
axs[2].imshow(mat_aligned_mspecs['/data/data/T12_data/sentences/t12.2022.05.05_sentences.mat'][-1], aspect='auto', origin='lower')
axs[2].set_title('aligned TTS mspec')
axs[2].set_ylabel("MFCC")
axs[2].set_xlabel("time (20ms)")
plt.tight_layout()
plt.show()


##
def resample_idx(idx, orig_sr, target_sr):
    return int(idx * target_sr / orig_sr)
s = resample_idx(startIdx, 1000/20, 30000)
e = resample_idx(stopIdx, 1000/20, 30000)
t12_audio = audio_block[block_idx][s:e]
ex_mspec = mel_spectrogram(torch.tensor(t12_audio[None], dtype=torch.float32),
                                2048, 80, 30000, 30000//50, 30000//25, 0, 8000, center=False).squeeze()
fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharey=True)
axs[0].imshow(t12_mspec, aspect='auto', origin='lower')
axs[0].set_title('t12_mspec')
axs[1].imshow(ex_mspec, aspect='auto', origin='lower')
axs[1].set_title('again from audio (should match)')
Audio(t12_audio, rate=30000)
##
Audio(tts_audio, rate=sample_rate)
##
# for sentence, utterances in tqdm(sentence_mapping.items()):
#     for mat_file, idx in utterances:
#         sentence_file = competition_file_mapping[mat_file]
#         sentence_mat = mat_files[sentence_file]
#         # map goCue index to block index & relative audio index
        
#         t12_audio = sentence_mat['audio'][0,blockIdx][0]

#         if speaking_modes[mat_file] == "vocalized":
#             tts_audio, tts_phonemes, sample_rate = load_TTS_data(sentence)
            
#             print(f"{t12_audio.shape=}, {tts_audio.shape=}")
#             break
#             dtw_path = DTW_between_audio_files(t12_audio, tts_audio)
            
#             # This is a placeholder for how you might align phoneme labels.
#             # You might need a more specific way to accumulate phoneme labels based on your data.
#             aligned_phonemes = [tts_phonemes[i] for i in dtw_path]
#             # TODO: accumulate the aligned_phonemes for each timestep of your neural data.

print("Processing complete!")
##


##
# use torch.cdist and align_from_distances for DTW






##








##
# look at some audio envelops for speakingMode 'speak' or 'mouthing'
mouthing_mat = None
speak_mat = None
for f in sentences_files:
    # mat = scipy.io.loadmat(f)
    mat = mat_files[f]
    assert len(mat['speakingMode']) == 1, f"More than one speakingMode in file {f}"
    if mat['speakingMode'][0] == 'attempted nonvocal speaking':
        mouthing_mat = mat
    elif mat['speakingMode'][0] == 'attempted speaking':
        sp
        eak_mat = mat
    else:
        raise Exception(f"Unknown speakingMode: {mat['speakingMode'][0]} fro file {f}")
    # if mouthing_mat is not None and speak_mat is not None:
    #     break
    
##
# plot audio envelopes for mouthing and speaking
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
mouthed_audio = librosa.util.buf_to_float(mouthing_mat['audio'][0,0][0])
spoken_audio = librosa.util.buf_to_float(speak_mat['audio'][0,0][0])
ax1.plot(librosa.resample(mouthed_audio, orig_sr=30000, target_sr=1000))
ax2.plot(librosa.resample(spoken_audio, orig_sr=30000, target_sr=1000))
ax1.set_title('mouthing')
ax2.set_title('speaking')
plt.show()

##
# mouthed Audio
# write_wav(os.path.join(T12_dir, "mouthing.wav"), 30000, mouthed_audio)
Audio(mouthed_audio, rate=30000)
##
# vocalized Audio
# write_wav(os.path.join(T12_dir, "mouthing.wav"), 30000, mouthed_audio)
Audio(spoken_audio, rate=30000)
    
    
##
# ============== Explore the data format ==============
# blockList starts at 0 or 5 or ...?
print(f"{mouthing_mat['blockList']=}\n{mouthing_mat['blockTypes']=}")
num_blocks = len(mouthing_mat['blockTypes'])
T = mouthing_mat['spikePow'].shape[0]
neural_seconds = T * 20 / 1000
print(f"{T=}\n{neural_seconds=}")
# audio: B x 1 vector of raw audio snippets (B=number of blocks)
# Audio data was recorded at 30 kHz and is aligned to the neural data
# (it begins at the first time step of neural data for that block).
nAudio = np.sum([m[0].shape[0] for m in mouthing_mat['audio'][0]])
audio_seconds = nAudio / 30000

# audioFeatures: T x 42 matrix of MFCC features (T=number of 20 ms time steps).
# Can be used as a control to attempt to decode speech from audio features. 
print(f"{nAudio=}\n{audio_seconds=}")
audio_seconds = 20 * nAudio / T / 1000
audio_seconds
##
mouthing_mat['xpcClock']
##
mouthing_mat['nsp1Clock']
##
mouthing_mat['nsp2Clock']
##
mouthing_mat['redisClock']
##
block_start_idx = np.concatenate([[0], 1 + np.where(np.diff(mouthing_mat['blockNum'][:,0]))[0]])
block_start_idx

##
mouthing_mat['delayTrialEpochs'] # eg [1, 152]
mouthing_mat['goTrialEpochs'] # eg [153, 345]
assert np.all(mouthing_mat['delayTrialEpochs'][:,1] + 1 == mouthing_mat['goTrialEpochs'][:,0])
assert len(mouthing_mat['sentences']) == len(mouthing_mat['goTrialEpochs'])
##
trial_start = mouthing_mat['delayTrialEpochs'][:,0]
trial_end = mouthing_mat['goTrialEpochs'][:,1]
trial_duration = trial_end - trial_start
tot_time = trial_duration.sum()
tot_time / 50 # seconds
##
sentenceDurations
##
len(mouthing_mat['sentences']), mouthing_mat['audioFeatures'].shape, mouthing_mat['audioEnvelope'].shape, mouthing_mat['audio'].shape
##
mouthing_mat['spikePow'].shape
##
##
# For each sentence, read the audio envelope and run DTW.















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
uniq_sentences = np.unique(all_sentences)
filenames = defaultdict(list)
# halfway = len(uniq_sentences) // 2
halfway = 6291
for sentence in tqdm(uniq_sentences[1245:halfway]):
    fn = sentence_to_fn(sentence)
    filenames[fn].append(sentence)
   
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
