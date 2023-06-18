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


librispeech_datasets = load_dataset("librispeech_asr")
librispeech_clean_train = torch.utils.data.ConcatDataset([librispeech_datasets['train.clean.100'],
                                                    librispeech_datasets['train.clean.360']])
                                                    # librispeech_datasets['train.other.500']])
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


datamodule = EMGDataModule(data_dir, togglePhones, normalizers_file, max_len=max_len)
emg_train = datamodule.train

mfcc_norm, emg_norm = pickle.load(open(normalizers_file,'rb'))
##
class LibrispeechDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, text_transform, mfcc_norm):
        self.dataset = dataset
        self.text_transform = text_transform
        self.mfcc_norm = mfcc_norm
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, index):
        audio = self.dataset[index]['audio']['array']
        text = self.dataset[index]['text']
        audio = librosa.resample(audio, orig_sr=16000, target_sr=22050)
        audio = np.clip(audio, -1, 1) # because resampling sometimes pushes things out of range
        # window is 1024, hop is 256, so length of output is (len(audio) - 1024) // 256 + 1
        # (or at least that's what co-pilot says)
        pytorch_mspec = mel_spectrogram(torch.tensor(audio, dtype=torch.float32).unsqueeze(0),
                                        1024, 80, 22050, 256, 1024, 0, 8000, center=False)
        mfccs = pytorch_mspec.squeeze(0).T.numpy()
        mfccs = mfcc_norm.normalize(mfccs)
        text_int = np.array(self.text_transform.text_to_int(text), dtype=np.int64)
        example = {'audio_features': mfccs,
            'text':text,
            'text_int':text_int,
            }
        return example
    
    @staticmethod
    def collate_raw(batch):
        batch_size = len(batch)
        audio_features = []
        audio_feature_lengths = []
        text_int = []
        text_int_lengths = []
        text = []
        for example in batch:
            audio_features.append(example['audio_features'])
            audio_feature_lengths.append(example['audio_features'].shape[0])
            text_int.append(example['text_int'])
            text_int_lengths.append(example['text_int'].shape[0])
            text.append(example['text'])
        return {
            'audio_features': audio_features,
            'audio_feature_lengths':audio_feature_lengths,
            'raw_emg': None,
            'text_int': text_int,
            'text_int_lengths': text_int_lengths,
            'text': text,
            'silent': False,
        }

    
class EMGAndSpeechModule(pl.LightningDataModule):
    def __init__(self, emg_data_module:pl.LightningDataModule,
                 speech_dataset:torch.utils.data.Dataset,
                 speech_bz:int=32):
        self.emg_data_module = emg_data_module
        self.speech_dataset = speech_dataset
        self.speech_bz = speech_bz
        
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.speech_dataset, batch_size=self.speech_bz, shuffle=True)
    
speech_train = LibrispeechDataset(librispeech_clean_train, emg_train.text_transform, mfcc_norm)
num_emg_train = len(emg_train)
num_speech_train = len(speech_train)

num_emg_train, num_speech_train
emg_speech_train = torch.utils.data.ConcatDataset([
    emg_train, speech_train
])
len(emg_speech_train)

# emg_speech_train[num_emg_train-1]
emg_speech_train[num_emg_train]
##
import matplotlib.pyplot as plt
# plt.matshow(emg_speech_train[num_emg_train]['audio_features'].T)
plt.matshow(emg_speech_train[-1]['audio_features'].T)
# plt.hist(emg_speech_train[num_emg_train]['audio_features'].reshape(-1))
plt.title("librispeech sample")
# plt.colorbar()
##
plt.matshow(emg_speech_train[num_emg_train-1]['audio_features'].T)
# plt.hist(emg_speech_train[num_emg_train-1]['audio_features'].reshape(-1))
plt.title("gaddy sample")
# plt.colorbar()
##
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
# plt.hist(mspec.reshape(-1))
#
# TODO:
# https://discuss.pytorch.org/t/how-to-balance-mini-batches-during-each-epoch/120055
# stratified sampling https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

# and reload every epoch
# pl.trainer(
#     reload_dataloaders_every_epoch=True
# )

# speech_bz = 32
# num_steps_per_epoch = 242
# train_loader = DataLoader(
#     librispeech_clean_train,
#     batch_size=speech_bz,
#     shuffle=False,
#     num_workers=1,
#     pin_memory=True,
#     drop_last=True,
#     sampler=SubsetRandomSampler(
#         torch.randint(high=len(librispeech_clean_train),
#                       size=(num_steps_per_epoch*speech_bz,))
#     ),
# )
##
