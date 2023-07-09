##

import os, pickle, sys, torch
# horrible hack to get around this repo not being a proper python package
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(SCRIPT_DIR)
from data_utils import TextTransform
from dataloaders import LibrispeechDataset, CachedDataset
from datasets import load_dataset
import subprocess, numpy as np
from tqdm import tqdm
from scipy import signal

from read_emg import EMGDataset, SizeAwareSampler, PreprocessedEMGDataset, \
    PreprocessedSizeAwareSampler, EMGDataModule, ensure_folder_on_scratch

##
base_bz = 12
val_bz = 8
  
hostname = subprocess.run("hostname", capture_output=True)
ON_SHERLOCK = hostname.stdout[:2] == b"sh"

# load our data file paths and metadata:
per_index_cache = True # read each index from disk separately
if per_index_cache:
    cache_suffix = "_per_index"
else:
    cache_suffix = ""
if ON_SHERLOCK:
    sessions_dir = '/oak/stanford/projects/babelfish/magneto/'
    # TODO: bechmark SCRATCH vs LOCAL_SCRATCH ...?
    scratch_directory = os.environ["SCRATCH"]
    gaddy_dir = '/oak/stanford/projects/babelfish/magneto/GaddyPaper/'
else:
    sessions_dir = '/data/magneto/'
    scratch_directory = "/scratch"
    gaddy_dir = '/scratch/GaddyPaper/'
    
librispeech_train_cache = os.path.join(scratch_directory, "librispeech_train_phoneme_cache")
librispeech_val_cache = os.path.join(scratch_directory, "librispeech_val_phoneme_cache")
librispeech_test_cache = os.path.join(scratch_directory, "librispeech_test_phoneme_cache")



max_len = 128000 * 2 # original Gaddy
# max_len = 128000
# max_len = 64000 # OOM
# max_len = 32000 # works for supNCE
# max_len = 48000
data_dir = os.path.join(gaddy_dir, 'processed_data/')
emg_dir = os.path.join(gaddy_dir, 'emg_data/')
lm_directory = os.path.join(gaddy_dir, 'pretrained_models/librispeech_lm/')
normalizers_file = os.path.join(SCRIPT_DIR, "normalizers.pkl")
togglePhones = False

train_pre = PreprocessedEMGDataset(base_dir = data_dir, train = True, dev = False, test = False,
    togglePhones = togglePhones, normalizers_file = normalizers_file)

train_gaddy = EMGDataset(base_dir = None, dev = False, test = False,
    togglePhones = togglePhones, normalizers_file = normalizers_file)

train_guy = EMGDataset(base_dir = None, dev = False, test = False,
    togglePhones = togglePhones, normalizers_file = normalizers_file, returnRaw=True)


data_dir = '/scratch/GaddyPaper/cached/'
train_cache = CachedDataset(EMGDataset, os.path.join(data_dir, 'emg_train'),
            base_dir = None, dev = False, test = False,
            togglePhones = togglePhones, normalizers_file = normalizers_file)
##
train_pre[0]['text'], train_cache[0]['text'], train_gaddy[0]['text'], train_guy[0]['text']
##
train_cache[0]

##
ptext = list(sorted(map(lambda x: str(x['text'][0]), train_pre)))
ctext = list(sorted(map(lambda x: x['text'], train_cache)))
##
ctext[100:105], ctext[100:105]
##
for i in range(len(ptext)):
    if ptext[i] != ctext[i]:
        print(i, ptext[i], ctext[i])
##
import re

def audio_filepath_to_uuid(audio_filepath):
    m = re.search(r"\/(\d+-\d+(\_\d+)*\/[\w_]+).flac", audio_filepath)
    return m.group(1)
for i in range(5):
    af_path = str(train_pre[i]['audio_file'][0])
    print(audio_filepath_to_uuid(af_path))
    
pre_uuid_to_index = {audio_filepath_to_uuid(str(train_pre[i]['audio_file'][0])): i for i in range(len(train_pre))}
cache_uuid_to_index = {audio_filepath_to_uuid(str(train_cache[i]['audio_file'])): i for i in range(len(train_cache))}
gaddy_uuid_to_index = {audio_filepath_to_uuid(str(train_gaddy[i]['audio_file'])): i for i in range(len(train_gaddy))}
guy_uuid_to_index = {audio_filepath_to_uuid(str(train_gaddy[i]['audio_file'])): i for i in range(len(train_guy))}
##
u = "4-21/517_audio_clean"
p = train_pre[pre_uuid_to_index[u]]
c = train_cache[cache_uuid_to_index[u]]
g = train_gaddy[gaddy_uuid_to_index[u]]
guy = train_guy[guy_uuid_to_index[u]]
# p['raw_emg'], c['raw_emg']
p['audio_features'], c['audio_features']
##
import matplotlib.pyplot as plt
# plot spectrogram
plt.imshow(p['audio_features'].T, aspect='auto', origin='lower')
plt.colorbar()
plt.title("pre")
plt.show()
plt.imshow(c['audio_features'].T, aspect='auto', origin='lower')
plt.title("cached")
plt.colorbar()
plt.show()
plt.imshow(guy['audio_features'].T, aspect='auto', origin='lower')
plt.title("guy")
plt.colorbar()
plt.show()
##
# calculate EMG spectrogram from raw signal
def plot_spectrogram(emg,title=None):
    emg = emg - np.mean(emg, axis=0)
    emg = emg / np.std(emg, axis=0)
    emg = emg.mean(axis=-1)
    f, t, Sxx = signal.spectrogram(emg, 1000, nperseg=128, noverlap=128-32, detrend=False)
    plt.pcolormesh(t, f, np.log(Sxx), shading='gouraud', clim=(0, -25))
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar()
    if title is not None:
        plt.title(title)
    plt.show()
plot_spectrogram(p['raw_emg'].numpy(), "PreprocessedEMGDataset")
plot_spectrogram(g['raw_emg'].numpy(), "EMGDataset")
plot_spectrogram(c['raw_emg'].numpy(), "CachedDataset(EMGDataset)")
plot_spectrogram(guy['raw_emg'].numpy(), "EMGDataset return raw")


##
np.isclose(guy['raw_emg'].numpy(), p['raw_emg'].numpy()).all()
##
g['raw_emg'].numpy().shape, guy['raw_emg'].numpy().shape

##
# random scratchpad :O
# import numpy as np
# affected = [np.array([i]) for i in range(64)]
# reference = []
# for i in range(64):
#     if i < 32:
#         r = np.delete(np.arange(32),i)
#     else:
#         r = np.delete(np.arange(32,64),i-32)
#     reference.append(r)
# common_average_reference(data, affected, reference)
##
