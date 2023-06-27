import torch, numpy as np, librosa, pickle, torchaudio, os, pytorch_lightning as pl
from data_utils import mel_spectrogram, read_phonemes
import torch.distributed as dist, sys, logging
from tqdm import tqdm

class LibrispeechDataset(torch.utils.data.Dataset):
    """
    A wrapper for the Librispeech dataset that returns the audio features and text.
    
    Args:
        dataset: a torch.utils.data.Dataset object in HuggingFace format
        text_transform: a TextTransform object that converts text to integers
        mfcc_norm: an MFCCNormalizer object that normalizes MFCCs
        alignment_dirs: Folders with TextGrid alignments

    Can download alignments from e.g. https://zenodo.org/record/2619474
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
            'phonemes': phonemes,
            # 'text': np.array([text]), # match Gaddy's format. seems unnecessary though, why not just str..?
            'text_int':torch.from_numpy(text_int),
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
    phonemes = []
    silent = []
    audio_only = []
    for example in batch:
        audio_features.append(example['audio_features'])
        audio_feature_lengths.append(example['audio_features'].shape[0])
        text_int.append(example['text_int'])
        text_int_lengths.append(example['text_int'].shape[0])
        # phonemes.append(example['phonemes'])
        if type(example['text']) == np.ndarray:
            text.append(example['text'][0]) # use a string instead of array([string])
        else:
            text.append(example['text'])
        if 'raw_emg' not in example:
            # audio only
            raw_emg.append(None)
            raw_emg_lengths.append(0)
            silent.append(False)
            audio_only.append(True)
        else:
            raw_emg.append(example['raw_emg'])
            raw_emg_lengths.append(example['raw_emg'].shape[0])
            if example['silent']:
                # don't use annoying array([[1]], dtype=uint8)
                silent.append(True)
            else:
                silent.append(False)
            audio_only.append(False)
    return {
        'audio_features': audio_features,
        'audio_feature_lengths':audio_feature_lengths,
        'raw_emg': raw_emg,
        'phonemes':phonemes,
        'raw_emg_lengths': raw_emg_lengths,
        'silent': silent,
        'audio_only': audio_only,
        'text': text,
        'text_int': text_int,
        'text_int_lengths': text_int_lengths,
    }
    
def split_batch_into_emg_audio(batch):
    # Can compose with collate_gaddy_or_speech
    emg = []
    # emg_phonemes = []
    length_emg = []
    y_length_emg = []
    y_emg = []
    
    audio = []
    # audio_phonemes = []
    length_audio = []
    y_length_audio = []
    y_audio = []
    
    paired_emg_idx = []
    paired_audio_idx = [] # same length as paired_emg_idx
    
    for i, (s,a) in enumerate(zip(batch['silent'], batch['audio_only'])):
        if s:
            # EMG
            emg.append(batch['raw_emg'][i])
            length_emg.append(batch['raw_emg_lengths'][i])
            y_length_emg.append(batch['text_int_lengths'][i])
            y_emg.append(batch['text_int'][i])
        elif a:
            # AUDIO
            audio.append(batch['audio_features'][i])
            length_audio.append(batch['audio_feature_lengths'][i])
            y_length_audio.append(batch['text_int_lengths'][i])
            y_audio.append(batch['text_int'][i])
        else:
            # EMG + AUDIO
            logging.debug(f"appending these idxs for emg, audio: {len(emg)}, {len(audio)}")
            paired_emg_idx.append(len(emg))
            paired_audio_idx.append(len(audio))
            
            emg.append(batch['raw_emg'][i])
            length_emg.append(batch['raw_emg_lengths'][i])
            y_length_emg.append(batch['text_int_lengths'][i])
            y_emg.append(batch['text_int'][i])
            
            audio.append(batch['audio_features'][i])
            length_audio.append(batch['audio_feature_lengths'][i])
            y_length_audio.append(batch['text_int_lengths'][i])
            y_audio.append(batch['text_int'][i])
            logging.debug(f"{len(emg)=}, {len(audio)=}")
    
    emg_tup = (emg, length_emg, y_length_emg, y_emg)
    audio_tup = (audio, length_audio, y_length_audio, y_audio)
    idxs = (paired_emg_idx, paired_audio_idx)
    return emg_tup, audio_tup, idxs
    
class CachedDataset(torch.utils.data.Dataset):
    """Cache a dataset to disk via pickle."""
    def __init__(self, Dataset, cache_path, per_index_cache=False, *args, **kwargs):
        """
        If per_index_cache is True, cache each index individually.
        Otherwise, cache the whole dataset.
        """
        super().__init__()
        self.cache_path = cache_path
        self.per_index_cache = per_index_cache
        
        if per_index_cache:
            if os.path.isdir(cache_path):
                self.len = len(os.listdir(cache_path))
            else:
                os.makedirs(cache_path)
                # cache each index individually to disk
                dset = Dataset(*args, **kwargs)
                self.len = len(dset)
                self.cache_each_index(dset)           
        else:
            if os.path.isfile(cache_path):
                # read entire dataset into memory
                with open(cache_path, 'rb') as f:
                    self.cache = pickle.load(f)
            else:
                # populate cache and save to file
                dset = Dataset(*args, **kwargs)
                self.cache = []
                self.populate_cache(dset)
            self.len = len(self.cache)

    def approximate_memory_usage(self, dset):
        sz = len(pickle.dumps(dset[0], protocol=pickle.HIGHEST_PROTOCOL))
        gb = self.len * sz / 1e9
        print("Approximate memory usage of dataset: {} GB".format(gb))
        return gb

        
    def populate_cache(self, dset):
        N = len(dset)
        self.approximate_memory_usage(dset)
        for data in tqdm(dset, desc='Caching dataset', total=N):
            self.cache.append(data)
        
        # save to file
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.cache, f, protocol=pickle.HIGHEST_PROTOCOL)

    def cache_each_index(self, dset):
        N = len(dset)
        self.approximate_memory_usage(dset)
        save_idx = 0
        for i in tqdm(range(N), desc='Caching each index', total=N):
            # Librispeech is missing some aligned phonemes, so we need to skip those
            # this does mean the cache will be smaller than the dataset, and some
            # indices may not match up
            try:
                data = dset[i]
                idx_path = os.path.join(self.cache_path, f"{save_idx}.pickle")
                with open(idx_path, 'wb') as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
                save_idx += 1
            except Exception as e:
                print(e)
                print(f"Failed to cache index {i}, skipping.")
    
    def __len__(self):
        return self.len
        
    def __getitem__(self, index):
        if self.per_index_cache:
            idx_path = os.path.join(self.cache_path, f"{index}.pickle")
            with open(idx_path, 'rb') as f:
                return pickle.load(f)
        else:
            return self.cache[index]
        
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

class DistributedStratifiedBatchSampler(StratifiedBatchSampler):
    """"Given the class of each example, sample batches without replacement
    with desired proportions of each class.
    
    If we run out of examples of a given class, we stop yielding batches.
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

class EMGAndSpeechModule(pl.LightningDataModule):
    def __init__(self, emg_train:torch.utils.data.Dataset,
            emg_val:torch.utils.data.Dataset, emg_test:torch.utils.data.Dataset,
            speech_train:torch.utils.data.Dataset, speech_val:torch.utils.data.Dataset,
            speech_test:torch.utils.data.Dataset,
            bz:int=64, num_replicas:int=1, num_workers:int=0,
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
    
        # TODO: could make this more general when need arises
        self.TrainSampler = TrainBatchSampler(classes, batch_class_proportions, bz)
        self.ValSampler = ValSampler
        self.TestSampler = TestSampler
        self.collate_fn = collate_gaddy_or_speech
        self.val_bz = bz // num_replicas
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        # self.prepare_data_per_node = False # we don't prepare data here
        # # https://github.com/Lightning-AI/lightning/pull/16712#discussion_r1237629807
        # self._log_hyperparams = False
        
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
            batch_sampler=self.TrainSampler
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