import torch, numpy as np, librosa, torchaudio, os, pytorch_lightning as pl
import dill as pickle # dill can serialize local classes
from data_utils import mel_spectrogram, read_phonemes
import torch.distributed as dist, sys, logging
from tqdm import tqdm
from functools import partial
from joblib import Memory

def persist_to_file(file_name):
    cache = {}
    try:
        with open(file_name, 'rb') as f:
            cache['k'] = pickle.load(f)
            logging.warn(f'Loaded cache from {file_name}')
    except:
        cache['k'] = None

    def decorator(original_func):
        def new_func(*args, **kwargs):
            if cache['k'] is None:
                cache['k'] = original_func(*args, **kwargs)
                with open(file_name, 'wb') as f:
                    pickle.dump(cache['k'], f)
            return cache['k']

        return new_func

    return decorator

class LibrispeechDataset(torch.utils.data.Dataset):
    """
    A wrapper for the Librispeech dataset that returns the audio features and text.
    
    Args:
        dataset: a torch.utils.data.Dataset object in HuggingFace format
        text_transform: a TextTransform object that converts text to integers
        mfcc_norm: an MFCCNormalizer object that normalizes MFCCs
        alignment_dirs: Folders with TextGrid alignments

    Can download alignments from e.g. https://zenodo.org/record/2619474
    
    Warning: Huggingface does unfortunate things like encode absolute paths such as
    /home/tyler/.cache/huggingface/datasets/librispeech_asr/...
    This breaks portability for a pickled class. terrible design.
    """
    def __init__(self, dataset, text_transform, mfcc_norm, alignment_dirs):
        super().__init__()
        self.dataset = dataset
        self.text_transform = text_transform
        self.mfcc_norm = mfcc_norm
        self.alignment_dirs = alignment_dirs
        
    def __len__(self):
        return len(self.dataset)
    
    def get_textgrid_path(self, item):
        speaker_id = item['speaker_id']
        chapter_id = item['chapter_id']
        id = item['id']
        for alignment_dir in self.alignment_dirs:
            textgrid_path = os.path.join(alignment_dir, str(speaker_id), str(chapter_id), f'{id}.TextGrid')
            if os.path.exists(textgrid_path):
                return textgrid_path
        else:
            raise ValueError(f'Could not find TextGrid for {speaker_id}/{chapter_id}/{id}')
        
        
    def __getitem__(self, index):
        "Reproduce the audio preprocessing from Gaddy on Librispeech data"
        item = self.dataset[index]
        audio = item['audio']['array']
        text = item['text']

        audio = librosa.resample(audio, orig_sr=16000, target_sr=22050)
        audio = np.clip(audio, -1, 1) # because resampling sometimes pushes things out of range
        
        # window is 1024, hop is 256, so length of output is (len(audio) - 1024) // 256 + 1
        # (or at least that's what co-pilot says)
        pytorch_mspec = mel_spectrogram(torch.tensor(audio, dtype=torch.float32).unsqueeze(0),
                                        1024, 80, 22050, 256, 1024, 0, 8000, center=False)
        mfccs = pytorch_mspec.squeeze(0).T.numpy()
        mfccs = self.mfcc_norm.normalize(mfccs)
        text_int = np.array(self.text_transform.text_to_int(text), dtype=np.int64)

        textgrid_path = self.get_textgrid_path(item)
        phonemes = read_phonemes(textgrid_path, max_len=mfccs.shape[0])

        example = {'audio_features': torch.from_numpy(mfccs),
            'text': text,
            'phonemes': torch.from_numpy(phonemes),
            # 'text': np.array([text]), # match Gaddy's format. seems unnecessary though, why not just str..?
            'text_int': torch.from_numpy(text_int),
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
            'raw_emg': [None] * batch_size,
            'text_int': text_int,
            'text_int_lengths': text_int_lengths,
            'text': text,
            'silent': [False] * batch_size,
        }
        
def collate_gaddy_or_speech(batch):
    batch_size = len(batch)
    audio_features = []
    audio_feature_lengths = []
    text_int = []
    text_int_lengths = []
    text = []
    raw_emg = []
    raw_emg_lengths = []
    parallel_emg = []
    parallel_emg_lengths = []
    phonemes = []
    silent = []
    audio_only = []
    for example in batch:
        text_int.append(example['text_int'])
        text_int_lengths.append(example['text_int'].shape[0])
        phonemes.append(example['phonemes'])
        if type(example['text']) == np.ndarray:
            text.append(example['text'][0]) # use a string instead of array([string])
        else:
            text.append(example['text'])
        if 'raw_emg' not in example:
            # audio only
            audio_features.append(example['audio_features'])
            audio_feature_lengths.append(example['audio_features'].shape[0])
            parallel_emg.append(None)
            parallel_emg_lengths.append(0)
            raw_emg.append(None)
            raw_emg_lengths.append(0)
            silent.append(False)
            audio_only.append(True)
        else:
            raw_emg.append(example['raw_emg'])
            raw_emg_lengths.append(example['raw_emg'].shape[0])
            audio_only.append(False)
            
            if example['silent']:
                # don't use annoying array([[1]], dtype=uint8)
                silent.append(True)
                audio_features.append(example['parallel_voiced_audio_features'])
                audio_feature_lengths.append(example['parallel_voiced_audio_features'].shape[0])
                parallel_emg.append(example['parallel_voiced_raw_emg'])
                parallel_emg_lengths.append(example['parallel_voiced_raw_emg'].shape[0])
            else:
                silent.append(False)
                audio_features.append(example['audio_features'])
                audio_feature_lengths.append(example['audio_features'].shape[0])
                parallel_emg.append(None)
                parallel_emg_lengths.append(0)

    return {
        'audio_features': audio_features,
        'audio_feature_lengths':audio_feature_lengths,
        'raw_emg': raw_emg,
        'raw_emg_lengths': raw_emg_lengths,
        'parallel_raw_emg': parallel_emg,
        'parallel_raw_emg_lengths': parallel_emg_lengths,
        'phonemes':phonemes,
        'silent': silent,
        'audio_only': audio_only,
        'text': text,
        'text_int': text_int,
        'text_int_lengths': text_int_lengths,
    }
    
def split_batch_into_emg_audio(batch):
    # Can compose with collate_gaddy_or_speech
    emg = []
    emg_phonemes = []
    length_emg = []
    y_length_emg = []
    y_emg = []
    
    audio = []
    audio_phonemes = []
    length_audio = []
    y_length_audio = []
    y_audio = []
    
    paired_emg_idx = []
    paired_audio_idx = [] # same length as paired_emg_idx
    
    silent_emg_idx = [] # silent emg
    parallel_audio_idx = [] # vocalized audio (parallel recordind with silent emg)
    
    # support other data collator
    if 'audio_only' in batch:
        audio_only = batch['audio_only']
    else:
        audio_only = [False] * len(batch['silent'])
    
    for i, (s,a) in enumerate(zip(batch['silent'], audio_only)):
        # logging.debug(f"{type(batch['phonemes'])=}")
        if not a:
            # Not audio only
            if s:
                # Silent EMG + parallel AUDIO
                silent_emg_idx.append(len(emg))
                parallel_audio_idx.append(len(audio))
                
                # INFO: we skip parallel emg and use only the parallel audio with silent emg
                # emg.append(batch['parallel_raw_emg'][i])
                # length_emg.append(batch['parallel_raw_emg_lengths'][i])
                
                
                # phoneme_len = len(batch['phonemes'][i])
                # emg_len = batch['raw_emg_lengths'][i] // 8
                # INFO: phonemes come from parallel dataset, so they are not always the same length as the emg
                # assert phoneme_len == emg_len, f"{phoneme_len} != {emg_len}. {batch['audio_features'][i].shape=}"
            else:
                # Paired EMG + AUDIO
                # TODO why doesn't this debug statement print..?
                logging.debug(f"appending these idxs for emg, audio: {len(emg)}, {len(audio)}")
                # print(f"appending these idxs for emg, audio: {len(emg)}, {len(audio)}")
                paired_emg_idx.append(len(emg))
                paired_audio_idx.append(len(audio))

            emg.append(batch['raw_emg'][i])
            length_emg.append(batch['raw_emg_lengths'][i])
            y_length_emg.append(batch['text_int_lengths'][i])
            y_emg.append(batch['text_int'][i])
            emg_phonemes.append(batch['phonemes'][i])
            
        audio.append(batch['audio_features'][i])
        length_audio.append(batch['audio_feature_lengths'][i])
        y_length_audio.append(batch['text_int_lengths'][i])
        y_audio.append(batch['text_int'][i])
        audio_phonemes.append(batch['phonemes'][i])
    
    emg_tup = (emg, length_emg, emg_phonemes, y_length_emg, y_emg)
    audio_tup = (audio, length_audio, audio_phonemes, y_length_audio, y_audio)
    idxs = (paired_emg_idx, paired_audio_idx, silent_emg_idx, parallel_audio_idx)
    return emg_tup, audio_tup, idxs
    
def cache_dataset(cache_path, Dataset=None, per_index_cache=False):
    """Class factory to modify Dataset to cache getitem to disk. Returns a Callable.
    
    This allows for retaining attributes & methods of the original Dataset class.
    
    Usage:
    >>> CachedMyDataset = cache_dataset('/path/my_dataset_cache.pkl', MyDataset)
    >>> dset = CachedMyDataset(*args, *kwargs)
    
    Filesystem for cache_path='/path/my_dataset_cache/' if per_index_cache=False
        /path/my_dataset_cache/
            instance.pkl
            0.pkl # if per_index_cache=True
            1.pkl # if per_index_cache=True
            
    """
    if cache_path[-4:] == '.pkl':
        assert not per_index_cache, "cache_path must be a directory if per_index_cache=True"
        instance_path = cache_path
    else:
        assert per_index_cache, "cache_path must be a .pkl file if per_index_cache=False"
        instance_path = os.path.join(cache_path, "instance.pkl")

    if os.path.isfile(instance_path):
        # load cached instance & return closure
        def wrapper(*args, **kwargs):
            with open(instance_path, 'rb') as f:
                return pickle.load(f)
        # for type stability, we return a Callable
        return wrapper
    # else: return Class that will cache instance to disk
    class CachedDataset(Dataset):
        """Cache a dataset to disk via pickle."""
        def __init__(self, *args, **kwargs):
            """
            If per_index_cache is True, cache each index individually.
            Otherwise, cache the whole dataset.
            """
            super().__init__(*args, **kwargs)
            
            cached_attrs = ['cache_path', 'per_index_cache', 'cache',
                           'approximate_memory_usage', 'populate_cache', 'len',
                           'cache_each_index_to_disk']
            for a in cached_attrs:
                if hasattr(super(), a):
                    logging.warning(f"{Dataset} already has attribute '{a}'. CachedDataset will clobber.")
                
            self.cache_path = cache_path
            self.per_index_cache = per_index_cache
            
            self.len = super().__len__()
            if per_index_cache:
                os.makedirs(cache_path)
                self.cache_each_index_to_disk()
            else:
                # populate cache and save to file
                self.cache = []
                self.populate_cache()
                
            # save instance to file
            with open(instance_path, 'wb') as f:
                pickle.dump(self, f)
                
        def approximate_memory_usage(self):
            sz = len(pickle.dumps(super().__getitem__(0), protocol=pickle.HIGHEST_PROTOCOL))
            gb = self.len * sz / 1e9
            print("Approximate memory usage of dataset: {} GB".format(gb))
            return gb
            
        def populate_cache(self):
            self.approximate_memory_usage()
            # __getitem__ can be expensive, so we cache the whole dataset once
            for i in tqdm(range(self.len), desc='Caching dataset', total=self.len):
                try:
                    data = super().__getitem__(i)
                    self.cache.append(data)
                except Exception as e:
                    print(e)
                    print(f"Failed to cache index {i}, skipping.")

        def cache_each_index_to_disk(self):
            self.approximate_memory_usage()
            save_idx = 0
            for i in tqdm(range(self.len), desc='Caching each index', total=self.len):
                # Librispeech is missing some aligned phonemes, so we need to skip those
                # this does mean the cache will be smaller than the dataset, and some
                # indices may not match up
                try:
                    data = super().__getitem__(i)
                    idx_path = os.path.join(self.cache_path, f"{save_idx}.pkl")
                    with open(idx_path, 'wb') as f:
                        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
                    save_idx += 1
                except Exception as e:
                    print(e)
                    print(f"Failed to cache index {i}, skipping.")
                    self.len -= 1
        
        def __len__(self):
            return self.len
            
        def __getitem__(self, index):
            if self.per_index_cache:
                idx_path = os.path.join(self.cache_path, f"{index}.pkl")
                with open(idx_path, 'rb') as f:
                    return pickle.load(f)
            else:
                return self.cache[index]
        
    return CachedDataset       
    
class StratifiedBatchSampler(torch.utils.data.Sampler):
    """"Given the class of each example, sample batches without replacement
    with desired proportions of each class.
    
    If we run out of examples of a given class, we stop yielding batches.
    
    Args:
        classes: array of class labels for each example
        class_proportion: array of desired proportion of each class in each batch
        batch_size: number of examples in each batch
        shuffle: whether to shuffle the examples before sampling
        drop_last: not used
        
    >>> x = np.arange(17)
    >>> classes = np.array([0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1])
    >>> for i in StratifiedBatchSampler(classes, np.array([0.5, 0.5]), 4, shuffle=False):
    ...   print(f"{x[i]=} {classes[i]=}")
    x[i]=array([0, 1, 5, 6]) classes[i]=array([0, 0, 1, 1])
    x[i]=array([2, 3, 7, 8]) classes[i]=array([0, 0, 1, 1])
    """
    def __init__(self, classes:np.ndarray, class_proportion:np.ndarray,
                 batch_size:int, shuffle:bool=True, drop_last:bool=False):        
        assert np.allclose(np.sum(class_proportion), 1)
        assert np.all(class_proportion >= 0)
        assert np.all(class_proportion <= 1)
        assert np.all(np.unique(classes) == np.arange(class_proportion.shape[0]))
        self.classes = classes
        self.class_proportion = class_proportion
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.class_indices = []
        for i in range(class_proportion.shape[0]):
            self.class_indices.append(np.where(classes == i)[0])
        self.class_indices = self.class_indices
        self.num_examples_per_class = np.array([len(x) for x in self.class_indices])
        self.class_n_per_batch = np.round(self.class_proportion * batch_size).astype(int)
        assert self.class_n_per_batch.sum() == batch_size, "Class proportion must evenly divide batch size"

        self.num_batches = int(np.floor(np.min(self.num_examples_per_class / self.class_n_per_batch)))
        self.drop_last = drop_last # not used

        self.epoch = 0 # not used by base class
    
    def set_epoch(self, epoch:int):
        self.epoch = epoch
        
    def __iter__(self):
        if self.shuffle:
            self.class_indices = [np.random.permutation(x) for x in self.class_indices]
        for batch in range(self.num_batches):
            batch_indices = []
            for i in range(self.class_n_per_batch.shape[0]):
                s = batch * self.class_n_per_batch[i]
                e = (batch+1) * self.class_n_per_batch[i]
                idxs = self.class_indices[i][s:e]
                batch_indices.extend(idxs)
            batch_indices = [int(x) for x in batch_indices]
            # not needed for our purposes as model doesn't care about order
            # if self.shuffle:
            #     batch_indices = np.random.permutation(batch_indices)
            yield batch_indices
            
    def __len__(self):
        return self.num_batches

class SizeAwareStratifiedBatchSampler(StratifiedBatchSampler):
    """Sample batches without replacement with desired proportions of each class,
    constraining max_len such that sum of all lengths in batch < max_len.
    
    If we run out of examples of a given class, we stop yielding batches.
    batch_size is used indirectly to estimate the split, while max_len
    actually determines ultimate number of examples per batch
    
    Args:
        classes: array of class labels for each example
        lengths: array of length for each example
        class_proportion: array of desired proportion of each class in each batch
        batch_size: anticipated number of examples in each batch
        max_len: maximum length of batch in samples
        shuffle: whether to shuffle the examples before sampling
    """
    
    def __init__(self, classes:np.ndarray, lengths:np.ndarray,
                 class_proportion:np.ndarray,
                 batch_size:int, max_len:int, shuffle:bool=True):        
        raise NotImplementedError("This is not yet tested. update to match DistributedSizeAwareStratifiedBatchSampler")
        super().__init__(classes, class_proportion, batch_size, shuffle)
        self.max_len = max_len
        self.lengths = lengths
        self.mini_batch_classes = np.concatenate([np.full(self.class_n_per_batch[i], i) for i in range(self.class_n_per_batch.shape[0])])
        
    def __iter__(self):
        if self.shuffle:
            class_indices = [np.random.permutation(x) for x in self.class_indices]
        else:
            class_indices = self.class_indices.copy()
            classes = self.classes.copy()
        class_indices = [list(x) for x in class_indices]
        batch = []
        batch_length = 0

        # what we actually want is to have summed length from each class in
        # each batch that is proportional to class_proportion. this approximates that

        # first, we choose class baseed on randomly sampling proportion
        while True:
            if self.shuffle:
                mini_batch_classes = np.random.permutation(self.mini_batch_classes)
            else:
                mini_batch_classes = self.mini_batch_classes
                
            for cl in mini_batch_classes:
                if len(class_indices[cl]) == 0:
                    # stop yielding batches if we run out of examples of a given class
                    break
                idx = class_indices[cl].pop()
                length = self.lengths[idx]
                if length > self.max_len:
                    logging.warning(f'Warning: example {idx} cannot fit within desired batch length')
                if length + batch_length > self.max_len:
                    yield batch
                    batch = []
                    batch_length = 0
                batch.append(idx)
                batch_length += length
            else: # no break so we continue while loop
                # https://stackoverflow.com/a/3150107
                continue
            break # break out of while loop when we run out of examples

class DistributedStratifiedBatchSampler(StratifiedBatchSampler):
    """Given the class of each example, sample batches without replacement
    with desired proportions of each class.
    
    If we run out of examples of a given class, we stop yielding batches.
    
    Args:
        classes: array of class labels for each example
        lengths: array of length for each example
        class_proportion: array of desired proportion of each class in each batch
        batch_size: number of examples in each batch
        max_len: maximum length of batch in samples
        shuffle: whether to shuffle the examples before sampling
        seed: random seed
        num_replicas: number of GPUs
    """
    def __init__(self, classes:np.ndarray, class_proportion:np.ndarray,
                 batch_size:int, shuffle:bool=True, drop_last:bool=False, seed:int=61923,
                 num_replicas:int=None):        
        if num_replicas is None:
            raise ValueError("num_replicas must be specified")
        # self.num_replicas = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
        self.num_replicas = num_replicas
        assert batch_size % self.num_replicas == 0, "Batch size must be divisible by number of GPUs"
        internal_bz = batch_size // self.num_replicas
        super().__init__(classes, class_proportion, internal_bz, shuffle, drop_last)
        mod = self.num_batches % self.num_replicas
        if mod != 0:
            self.num_batches -= mod
        
        if not dist.is_available():
            raise RuntimeError("Requires distributed package to be available")
        
        rank_key = "RANK" if "RANK" in os.environ else "LOCAL_RANK"

        self.rank = int(os.environ[rank_key]) if rank_key in os.environ else 0
        
        print(f"Initializing dataloader on Rank: {self.rank}")

        self.seed = seed
        
    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            self.class_indices = [x[torch.randperm(len(x), generator=g).tolist()]
                                    for x in self.class_indices]
        for batch in range(self.rank,self.num_batches,self.num_replicas):
            batch_indices = []
            for i in range(self.class_n_per_batch.shape[0]):
                s = batch * self.class_n_per_batch[i]
                e = (batch+1) * self.class_n_per_batch[i]
                idxs = self.class_indices[i][s:e]
                batch_indices.extend(idxs)
            batch_indices = [int(x) for x in batch_indices]
            # not needed for our purposes as model doesn't care about order
            # if self.shuffle:
            #     batch_indices = np.random.permutation(batch_indices)
            yield batch_indices
            
    def __len__(self):
        return int(np.floor(self.num_batches / self.num_replicas))
    
class DistributedSizeAwareStratifiedBatchSampler(DistributedStratifiedBatchSampler):
    """Sample batches without replacement with desired proportions of each class,
    constraining max_len such that sum of all lengths in batch < max_len.
    
    If we run out of examples of a given class, we stop yielding batches.
    
    Args:
        classes: array of class labels for each example
        lengths: array of length for each example
        class_proportion: array of desired proportion of each class in each batch
        batch_size: number of examples in each batch
        max_len: maximum length of batch in samples
        shuffle: whether to shuffle the examples before sampling
        seed: random seed
        num_replicas: number of GPUs
        constant_num_batches: always return same number of batches
        always_include_class: first example in each batch is always from this class
        
    always_include_class is useful for when models need at least one certain class of
    example in each batch, e.g. for cross contrastive loss between EMG & Audio.
    
    constant_num_batches is useful for pytorch lightning compatibility
    """
    def __init__(self, classes:np.ndarray, lengths:np.ndarray,
                class_proportion:np.ndarray,
                batch_size:int, max_len:int, shuffle:bool=True, seed:int=61923,
                num_replicas:int=None, constant_num_batches:bool=True,
                always_include_class:int=None):        
        super().__init__(classes, class_proportion, batch_size, shuffle,
                         seed=seed, num_replicas=num_replicas)
        self.max_len = max_len
        self.lengths = lengths
        self.always_include_class = always_include_class
        if always_include_class is not None:
            # prevent oversampling
            self.class_n_per_batch[self.always_include_class] -= 1
        self.mini_batch_classes = torch.from_numpy(np.concatenate([np.full(self.class_n_per_batch[i], i)
            for i in range(self.class_n_per_batch.shape[0])]))
        self.len = None

        self.constant_num_batches = False
        if constant_num_batches:
            self.hardcode_len = self.min_len(200) # assume 200 epochs
            self.constant_num_batches = True
            logging.warning(f"Hard coding len to {self.hardcode_len} as hack to get pytorch lightning to work")

        
    def min_len(self, num_epochs:int):
        """Minimum number of batches in dataset on any GPU."""
        cur_epoch = self.epoch
        min_length = np.inf
        # minimum per epoch
        for epoch in range(num_epochs):
            self.set_epoch(epoch)
            # minimum per GPU
            for rank in range(self.num_replicas):
                N = len(list(self.iter_batches(rank)))
                if N < min_length:
                    min_length = N
        self.set_epoch(cur_epoch)
        return min_length
    
    def iter_batches(self, rank):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            class_indices = [x[torch.randperm(len(x), generator=g)].tolist()
                                    for x in self.class_indices]
            # different indices per GPU
            class_indices = [x[rank::self.num_replicas]
                                  for x in class_indices]
        else:
            class_indices = [x.tolist() for x in self.class_indices]


        batch = [] # resets each time over max_len
        batch_length = 0
        batches = [] # accumulates
            
        while True:
            if self.shuffle:
                p = torch.randperm(self.mini_batch_classes.shape[0], generator=g)
                mini_batch_classes = self.mini_batch_classes[p]
            else:
                mini_batch_classes = self.mini_batch_classes
                
            if self.always_include_class is not None:
                mini_batch_classes = np.concatenate([[self.always_include_class], mini_batch_classes])
                
            for cl in mini_batch_classes:
                if len(class_indices[cl]) == 0:
                    # stop yielding batches if we run out of examples of a given class
                    # break
                    self.len = len(batches)
                    # logging.warning(f"DEBUG:return {self.len} batches. {self.epoch=}")
                    # logging.warning(f"DEBUG: {batches[10]=}, {batches[11]=}, {batches[12]=}")
                    avg_num_ex = np.mean([len(x) for x in batches])
                    logging.debug(f"Average number of examples per batch: {avg_num_ex}")
                    # if self.len < self.hardcode_len:
                    #     logging.warning(f"Warning: returning {self.len} batches, which is less than hardcode_len {self.hardcode_len}")
                    if self.constant_num_batches:
                        return iter(batches[:self.hardcode_len])
                    else:
                        return iter(batches)
                # class_indices shrink as we pop from them
                idx = class_indices[cl].pop()
                length = self.lengths[idx]
                if length > self.max_len:
                    logging.warning(f'Warning: example {idx} cannot fit within desired batch length, skipping')
                    continue
                if length + batch_length > self.max_len:
                    batches.append(batch)
                    batch = []
                    batch_length = 0
                    if self.always_include_class is not None:
                        break # ensure we always include at least one example from this class
                batch.append(idx)
                batch_length += length

    def __iter__(self):
        logging.debug("Initializing DistributedSizeAwareStratifiedBatchSampler")
        return self.iter_batches(self.rank)
                
    def approx_len(self, class_indices=None):
        """Return approximate number of batches per epoch.
        
        We can't know for sure how many batches we'll yield until we call iter,
        as it's stochastic.
        
        TODO: why is this estimate so bad??
        """
        if class_indices is None:
            class_indices = self.class_indices
        length_per_class = np.array([np.sum(np.array(self.lengths)[class_indices[i]])
                                     for i in range(self.class_n_per_batch.shape[0])])
        # logging.warning(f'length_per_class: {length_per_class}')
        # num batches limiting class 
        num_batches_per_class = np.ceil(length_per_class / self.class_proportion / self.max_len)
        # logging.warning(f'num_batches_per_class: {num_batches_per_class}')
        optimal_batches = np.min(num_batches_per_class)
        # logging.warning(f'{optimal_batches=}')
        # we could potentially have more batches if we don't always fill max_len
        # but if len > actual num_batches, pytorch lightning stalls
        return int(np.floor(optimal_batches))

    def __len__(self):
        "Return approximate number of batches per epoch"
        # https://github.com/Lightning-AI/lightning/issues/18023
        if self.constant_num_batches:
            return self.hardcode_len
        else:
            return len(iter(self))
        

# @persist_to_file("/tmp/2023-07-07_emg_speech_dset_lengths.pkl")
# @persist_to_file("/tmp/2023-07-20_emg_only_dset_lengths.pkl")
# isotime = datetime.datetime.now().isoformat()
# @persist_to_file(f"/tmp/{isotime}.pkl")
# @persist_to_file(f"/tmp/2023-07-24_emg-only.pkl")
@persist_to_file(f"/tmp/2023-07-25_emg_speech_dset_lengths.pkl")
def emg_speech_dset_lengths(dset:torch.utils.data.Dataset):
    """Calculate length of latent space for each example in dataset.
    
    Useful as contrastive loss is quadratic in length of latent space.
    """
    lengths = []
    for d in tqdm(dset, desc="calc lengths for sampler"):
        if 'silent' in d:
            # add length in latent space
            lengths.append(d['raw_emg'].shape[0] // 8)
        elif 'raw_emg' not in d:
            # audio only
            # same dim as latent space, no need to divide by 8
            lengths.append(d['audio_features'].shape[0])
        else:
            # EMG + audio, so length is sum of both
            emg_z_len = d['raw_emg'].shape[0] // 8
            audio_z_len = d['audio_features'].shape[0]
            assert emg_z_len == audio_z_len
            # lengths.append(emg_z_len + audio_z_len)
            # WARN/TODO: for EMG only
            lengths.append(emg_z_len)
    return lengths

class EMGAndSpeechModule(pl.LightningDataModule):
    def __init__(self, emg_train:torch.utils.data.Dataset,
            emg_val:torch.utils.data.Dataset, emg_test:torch.utils.data.Dataset,
            speech_train:torch.utils.data.Dataset, speech_val:torch.utils.data.Dataset,
            speech_test:torch.utils.data.Dataset,
            bz:int=64, val_bz:int=16, num_replicas:int=1, num_workers:int=0,
            TrainBatchSampler:torch.utils.data.Sampler=StratifiedBatchSampler,
            ValSampler:torch.utils.data.Sampler=None,
            TestSampler:torch.utils.data.Sampler=None,
            batch_class_proportions:np.ndarray=np.array([0.08, 0.42, 0.5]),
            pin_memory:bool=True
            ):
        """Given an EMG data module and a speech dataset, create a new data module.

        Args:
            emg_data_module (pl.LightningDataModule): train, val and test datasets for Gaddy-style EMG
            speech_train (torch.utils.data.Dataset): audio-only speech-to-text train dataset
            speech_val (torch.utils.data.Dataset): audio-only speech-to-text val dataset
            speech_test (torch.utils.data.Dataset): audio-only speech-to-text test dataset
            bz (int, optional): batch size. Defaults to 64.
            batch_class_proportions (np.ndarray, optional):  [EMG only (silent), EMG & Audio, Audio only]
            
        Gaddy's data has 1289 EMG-only examples (16%), and 6766 EMG & Audio examples (84%).
        """
        super().__init__()
        self.train = torch.utils.data.ConcatDataset([
            emg_train, speech_train
        ])
        train_emg_len = len(emg_train)
        
        self.val = emg_val
        # self.val = torch.utils.data.ConcatDataset([
        #     emg_data_module.val, speech_val
        # ])
        self.val_emg_len = len(self.val)

        self.test  = emg_test
        # self.test = torch.utils.data.ConcatDataset([
        #     emg_data_module.test, speech_test
        # ])
        self.test_emg_len = len(self.test)
        
        # 0: EMG only (silent), 1: EMG & Audio, 2: Audio only
        classes = np.concatenate([np.zeros(train_emg_len), 2 * np.ones(len(speech_train))])
        for i,b in enumerate(emg_train):
            if not b['silent']:
                classes[i] = 1
    
        isDSASBS = (TrainBatchSampler == DistributedSizeAwareStratifiedBatchSampler) or \
            (type(TrainBatchSampler) is partial) and \
            (TrainBatchSampler.func == DistributedSizeAwareStratifiedBatchSampler)
        # TODO: could make this more general when need arises
        if isDSASBS:
            self.train_lengths = emg_speech_dset_lengths(self.train)
            self.TrainBatchSampler = TrainBatchSampler(classes, self.train_lengths,
                batch_class_proportions, bz, num_replicas=num_replicas)
        else:
            self.TrainBatchSampler = TrainBatchSampler(classes, batch_class_proportions, bz)
        self.ValSampler = ValSampler
        self.TestSampler = TestSampler
        self.collate_fn = collate_gaddy_or_speech
        self.val_bz = val_bz // num_replicas
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        
        # self.prepare_data_per_node = False # we don't prepare data here
        # # https://github.com/Lightning-AI/lightning/pull/16712#discussion_r1237629807
        # self._log_hyperparams = False
        
    # avoids crash due to DDP when using distributed samplers
    def setup(self, stage=None):
        if self.ValSampler is None:
            self.val_sampler = None
        else:
            self.val_sampler = self.ValSampler()

        if self.TestSampler is None:
            self.test_sampler = None
        else:
            self.test_sampler = self.TestSampler()
        
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train,
            collate_fn=self.collate_fn,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            batch_sampler=self.TrainBatchSampler
        )
        
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val,
            collate_fn=self.collate_fn,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            batch_size=self.val_bz,
            sampler=self.val_sampler
        )
        
    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test,
            collate_fn=self.collate_fn,
            batch_size=self.val_bz,
            sampler=self.test_sampler
        )