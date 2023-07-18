##
2
##
# %load_ext autoreload
# %autoreload 2
##
from datasets import load_dataset
import os, torch, sys, pytorch_lightning as pl, numpy as np, librosa, pickle
import subprocess
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch.nn.functional as F

# horrible hack to get around this repo not being a proper python package
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(SCRIPT_DIR)
from data_utils import mel_spectrogram
from read_emg import load_audio, EMGDataModule, ensure_folder_on_scratch

from dataloaders import LibrispeechDataset, EMGAndSpeechModule

hostname = subprocess.run("hostname", capture_output=True)
ON_SHERLOCK = hostname.stdout[:2] == b"sh"

# load our data file paths and metadata:
if ON_SHERLOCK:
    sessions_dir = '/oak/stanford/projects/babelfish/magneto/'
    scratch_directory = os.environ["LOCAL_SCRATCH"]
else:
    sessions_dir = '/data/magneto/'
    scratch_directory = "/scratch"

# load our data file paths and metadata:
if ON_SHERLOCK:
    sessions_dir = '/oak/stanford/projects/babelfish/magneto/'
    scratch_directory = os.environ["LOCAL_SCRATCH"]
    gaddy_dir = '/oak/stanford/projects/babelfish/magneto/GaddyPaper/'
else:
    sessions_dir = '/data/magneto/'
    scratch_directory = "/scratch"
    gaddy_dir = '/scratch/GaddyPaper/'


##
# This approach is massively slow, at least 1+ hours.
# we should figure out how to better cache...
# (may need to expand the $LOCAL_SCRATCH variable first)
# ln -s $LOCAL_SCRATCH/huggingface ~/.cache/huggingface
# rsync -avP /oak/stanford/projects/babelfish/magneto/huggingface $LOCAL_SCRATCH/
librispeech_datasets = load_dataset("librispeech_asr")
librispeech_clean_train = torch.utils.data.ConcatDataset([librispeech_datasets['train.clean.100'],
                                                    librispeech_datasets['train.clean.360']])
                                                    # librispeech_datasets['train.other.500']])
librispeech_clean_val = librispeech_datasets['validation.clean']
librispeech_clean_test = librispeech_datasets['test.clean']
##
# TODO: running this cell is sometimes very slow even when should be cached...
max_len = 128000 * 2
data_dir = os.path.join(gaddy_dir, 'processed_data/')
emg_dir = os.path.join(gaddy_dir, 'emg_data/')
normalizers_file = os.path.join(SCRIPT_DIR, "normalizers.pkl")
togglePhones = False


# copy_metadata_command = f"rsync -am --include='*.json' --include='*/' --exclude='*' {emg_dir} {scratch_directory}/"
scratch_emg = os.path.join(scratch_directory,"emg_data")
if ON_SHERLOCK:
    if not os.path.exists(scratch_emg):
        os.symlink(emg_dir, scratch_emg)
    data_dir = ensure_folder_on_scratch(data_dir, scratch_directory)


emg_datamodule = EMGDataModule(data_dir, togglePhones, normalizers_file, max_len=max_len)
emg_train = emg_datamodule.train

mfcc_norm, emg_norm = pickle.load(open(normalizers_file,'rb'))
##
    
speech_train = LibrispeechDataset(librispeech_clean_train, emg_train.text_transform, mfcc_norm)
speech_val = LibrispeechDataset(librispeech_clean_val, emg_train.text_transform, mfcc_norm)
speech_test = LibrispeechDataset(librispeech_clean_test, emg_train.text_transform, mfcc_norm)
num_emg_train = len(emg_train)
num_speech_train = len(speech_train)

num_emg_train, num_speech_train
emg_speech_train = torch.utils.data.ConcatDataset([
    emg_train, speech_train
])
len(emg_speech_train)

emg_speech_train[num_emg_train-1]
emg_speech_train[num_emg_train]

datamodule =  EMGAndSpeechModule(emg_datamodule, speech_train, speech_val, speech_test)

for bat in datamodule.train_dataloader():
    print(bat.keys())
    break
##
# [ex.shape for ex in bat['audio_features']]

# gaddy dataloader gives array([[0]], dtype=uint8), librispeech gives False
# does this matter?
# [ex for ex in bat['silent']]

[ex for ex in bat['text']]
# [ex.shape for ex in bat['text']]
# [type(ex) for ex in bat['text']]
# [emg_train.text_transform.int_to_text(emg_train.text_transform.text_to_int(ex[0])) for ex in bat['text']]
##
import matplotlib.pyplot as plt

import torchaudio, pickle
from data_utils import normalize_volume, mel_spectrogram
from sklearn.preprocessing import normalize, minmax_scale, power_transform, scale, robust_scale
from matplotlib.gridspec import GridSpec

ref_audio_features = emg_speech_train[num_emg_train-1]['audio_features']

filename = emg_speech_train[num_emg_train-1]['audio_file'][0]
audio, r = torchaudio.load(filename)
audio    = audio.numpy().T

if len(audio.shape) > 1:
    audio = audio[:,0] # select first channel of stero audio

# audio = normalize_volume(audio)

if r == 16000:
    audio = librosa.resample(audio, orig_sr=16000, target_sr=22050)
else:
    assert r == 22050
#print(audio, audio.shape)

audio = np.clip(audio, -1, 1) # because resampling sometimes pushes things out of range
pytorch_mspec = mel_spectrogram(torch.tensor(audio, dtype=torch.float32).unsqueeze(0), 1024, 80, 22050, 256, 1024, 0, 8000, center=False)

mspec = pytorch_mspec.squeeze(0).T.numpy()

mfcc_norm, emg_norm = pickle.load(open(normalizers_file,'rb'))


# mspec = normalize(mspec)
# mspec = minmax_scale(mspec,(-1,1))
# mspec = scale(mspec)
# mspec = power_transform(mspec)
# mspec = robust_scale(mspec)
mspec = mfcc_norm.normalize(mspec)

# fig, axs = plt.subplots(2,2,figsize=(10,5))
fig = plt.figure(figsize=(15,5))
gs = GridSpec(2, 2, width_ratios=[3,1], height_ratios=[1, 1])
axs = [[],[]]
axs[0].append(fig.add_subplot(gs[0]))
axs[0].append(fig.add_subplot(gs[1]))
axs[1].append(fig.add_subplot(gs[2]))
axs[1].append(fig.add_subplot(gs[3]))
axs = np.array(axs)

cax = axs[0,0].matshow(ref_audio_features.T)
fig.colorbar(cax)
axs[0,0].set_title("Reference audio features")
axs[0,1].hist(ref_audio_features.reshape(-1))
# axis off
axs[0,0].axis('off')
cax = axs[1,0].matshow(mspec.T)
fig.colorbar(cax)
axs[1,0].set_title("New audio features")
axs[1,0].axis('off')
axs[1,1].hist(mspec.reshape(-1))

print(ref_audio_features.shape, mspec.shape)
##
plt.matshow(ref_audio_features.T - mspec[:-1].T)
# plt.matshow(ref_audio_features.T - mspec[1:].T)
plt.colorbar()
##
first_one = (ref_audio_features.T - mspec[1:].T).abs().sum()
last_one = (ref_audio_features.T - mspec[:-1].T).abs().sum()
print(f"first_one: {first_one}, last_one: {last_one}")
(ref_audio_features - mspec[:-1]).abs().sum(0)

##
num_silent = 0
n = len(emg_train)
for b in emg_train:
    if b['silent']:
        num_silent += 1

print(f"n: {n}, num_silent: {num_silent}, {num_silent/n*100:.1f}%")
