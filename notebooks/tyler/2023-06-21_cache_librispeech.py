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

librispeech_datasets = load_dataset("librispeech_asr")
librispeech_clean_train = torch.utils.data.ConcatDataset([librispeech_datasets['train.clean.100'],
                                                    librispeech_datasets['train.clean.360']])
                                                    # librispeech_datasets['train.other.500']])
librispeech_clean_val = librispeech_datasets['validation.clean']
librispeech_clean_test = librispeech_datasets['test.clean']

normalizers_file = os.path.join(SCRIPT_DIR, "normalizers.pkl")
mfcc_norm, emg_norm = pickle.load(open(normalizers_file,'rb'))
text_transform = TextTransform(togglePhones = False)

speech_train = LibrispeechDataset(librispeech_clean_train, text_transform, mfcc_norm)
# speech_val = LibrispeechDataset(librispeech_clean_val, text_transform, mfcc_norm)
# speech_test = LibrispeechDataset(librispeech_clean_test, text_transform, mfcc_norm)


librispeech_train_cache = os.path.join(os.environ["SCRATCH"], "librispeech_train_cache")
librispeech_val_cache = os.path.join(os.environ["SCRATCH"], "librispeech_val_cache")
librispeech_test_cache = os.path.join(os.environ["SCRATCH"], "librispeech_test_cache")
##
cached_speech_train =  CachedDataset(LibrispeechDataset, librispeech_train_cache, librispeech_clean_train, text_transform, mfcc_norm)
del cached_speech_train
cached_speech_val =  CachedDataset(LibrispeechDataset, librispeech_val_cache, librispeech_clean_val, text_transform, mfcc_norm)
del cached_speech_val
cached_speech_test =  CachedDataset(LibrispeechDataset, librispeech_test_cache, librispeech_clean_test, text_transform, mfcc_norm)

##
# %%timeit
# # 80ns
# cached_speech_train[0];

# ##
# %%timeit
# # 204ms
# speech_train[0];