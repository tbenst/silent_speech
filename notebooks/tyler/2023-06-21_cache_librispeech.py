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

##
# cache to individual files
import subprocess
from tqdm import tqdm
hostname = subprocess.run("hostname", capture_output=True)
ON_SHERLOCK = hostname.stdout[:2] == b"sh"

assert os.environ["NEPTUNE_ALLOW_SELF_SIGNED_CERTIFICATE"] == 'TRUE', "run this in shell: export NEPTUNE_ALLOW_SELF_SIGNED_CERTIFICATE='TRUE'"

# load our data file paths and metadata:
if ON_SHERLOCK:
    sessions_dir = '/oak/stanford/projects/babelfish/magneto/'
    # TODO: bechmark SCRATCH vs LOCAL_SCRATCH ...?
    scratch_directory = os.environ["LOCAL_SCRATCH"]
    gaddy_dir = '/oak/stanford/projects/babelfish/magneto/GaddyPaper/'
    librispeech_train_cache = os.path.join(os.environ["SCRATCH"], "librispeech_train_cache")
    librispeech_val_cache = os.path.join(os.environ["SCRATCH"], "librispeech_val_cache")
    librispeech_test_cache = os.path.join(os.environ["SCRATCH"], "librispeech_test_cache")

else:
    sessions_dir = '/data/magneto/'
    scratch_directory = "/scratch"
    gaddy_dir = '/scratch/GaddyPaper/'
    librispeech_train_cache = os.path.join(scratch_directory, "librispeech_train_cache")
    librispeech_val_cache = os.path.join(scratch_directory, "librispeech_val_cache")
    librispeech_test_cache = os.path.join(scratch_directory, "librispeech_test_cache")



cached_speech_train =  CachedDataset(LibrispeechDataset, librispeech_train_cache)
##
def cache_each_index(dset:CachedDataset):
    """
    Convert a CachedDataset (one pickle) into a CachedDataset where
    each index is cached individually
    """
    cache_path = f"{dset.cache_path}_per_index"
    os.makedirs(cache_path, exist_ok=True)
    N = len(dset)
    for i in tqdm(range(N), desc='Caching each index', total=N):
        data = dset[i]
        idx_path = os.path.join(cache_path, f"{i}.pickle")
        pickle.dump(data, open(idx_path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
cached_speech_train =  CachedDataset(LibrispeechDataset, librispeech_train_cache)
del cached_speech_train
cached_speech_val =  CachedDataset(LibrispeechDataset, librispeech_val_cache)
cache_each_index(cached_speech_val)
del cached_speech_val
cached_speech_test =  CachedDataset(LibrispeechDataset, librispeech_test_cache)
cache_each_index(cached_speech_test)
##
# convert phonemes to tensor
# speech_train
# speech_val
# speech_test

def rewrite_with_phoneme_tensor(dset:CachedDataset):
    "For each index, save new pickled file with phonemes as tensor"
    N = len(dset)
    save_idx = 0
    for i in tqdm(range(N), desc='Caching each index', total=N):
        try:
            data = dset[i]
            data["phonemes"] = torch.from_numpy(data["phonemes"])
            idx_path = os.path.join(dset.cache_path, f"{save_idx}.pickle")
            with open(idx_path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            save_idx += 1
        except Exception as e:
            print(e)
            print(f"Failed to cache index {i}, skipping.")
            
rewrite_with_phoneme_tensor(speech_train)
##
rewrite_with_phoneme_tensor(speech_test)
