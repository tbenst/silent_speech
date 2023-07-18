##
import time
import numpy as np, os
from joblib import Memory

scratch = os.environ['SCRATCH']
memory = Memory(os.path.join(scratch, "testcache"))

##
@memory.cache
def costly_compute(data, column_index=0):
    """Simulate an expensive computation"""
    time.sleep(5)
    return data[column_index]

rng = np.random.RandomState(42)
data = rng.randn(int(1e5), 10)
start = time.time()
data_trans = costly_compute(data)
end = time.time()

print('\nThe function took {:.2f} s to compute.'.format(end - start))
print('\nThe transformed data are:\n {}'.format(data_trans))
##
# will use cache even if the function has changed
@memory.cache
def costly_compute2(data, column_index=0):
    """Simulate an expensive computation"""
    time.sleep(7)
    return data[column_index]

rng = np.random.RandomState(42)
data = rng.randn(int(1e5), 10)
start = time.time()
data_trans = costly_compute(data)
end = time.time()

print('\nThe function took {:.2f} s to compute.'.format(end - start))
print('\nThe transformed data are:\n {}'.format(data_trans))
##
from datasets import load_dataset
import torch, sys, pickle
from data_utils import TextTransform

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(SCRIPT_DIR)
from dataloaders import LibrispeechDataset

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
speech_val = LibrispeechDataset(librispeech_clean_val, text_transform, mfcc_norm)
speech_test = LibrispeechDataset(librispeech_clean_test, text_transform, mfcc_norm)


memory = Memory(os.path.join(scratch, "librispeech_cache"))

@memory.cache
def cached_librispeech(index, split="train")