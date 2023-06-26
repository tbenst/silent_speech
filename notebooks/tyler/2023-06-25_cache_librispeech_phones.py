##
2
##
import os, pickle, sys, torch
# horrible hack to get around this repo not being a proper python package
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(SCRIPT_DIR)
from data_utils import TextTransform
from dataloaders import LibrispeechDataset, CachedDataset
from datasets import load_dataset
import subprocess
from tqdm import tqdm
hostname = subprocess.run("hostname", capture_output=True)
ON_SHERLOCK = hostname.stdout[:2] == b"sh"

##
librispeech_datasets = load_dataset("librispeech_asr")
librispeech_clean_train = torch.utils.data.ConcatDataset([librispeech_datasets['train.clean.100'],
                                                    librispeech_datasets['train.clean.360']])
                                                    # librispeech_datasets['train.other.500']])
librispeech_clean_val = librispeech_datasets['validation.clean']
librispeech_clean_test = librispeech_datasets['test.clean']

normalizers_file = os.path.join(SCRIPT_DIR, "normalizers.pkl")
mfcc_norm, emg_norm = pickle.load(open(normalizers_file,'rb'))
text_transform = TextTransform(togglePhones = False)
##
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
alignment_dir = os.path.join(scratch_directory, "librispeech-alignments")



# speech_train = LibrispeechDataset(librispeech_clean_train, text_transform, mfcc_norm)
# speech_val = LibrispeechDataset(librispeech_clean_val, text_transform, mfcc_norm)
# speech_test = LibrispeechDataset(librispeech_clean_test, text_transform, mfcc_norm)


##
alignment_dirs = [os.path.join(alignment_dir, d) for d in os.listdir(alignment_dir)]
cached_speech_val =  CachedDataset(LibrispeechDataset, librispeech_val_cache, True,
    librispeech_clean_val, text_transform, mfcc_norm,
    list(filter(lambda x: "dev" in x, alignment_dirs)))
del cached_speech_val
cached_speech_train =  CachedDataset(LibrispeechDataset, librispeech_train_cache, True,
    librispeech_clean_train, text_transform, mfcc_norm,
    list(filter(lambda x: "train" in x, alignment_dirs)))
del cached_speech_train
cached_speech_test =  CachedDataset(LibrispeechDataset, librispeech_test_cache, True,
    librispeech_clean_test, text_transform, mfcc_norm,
    list(filter(lambda x: "test" in x, alignment_dirs)))
del cached_speech_test

##