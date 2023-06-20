import torch, numpy as np, librosa, pickle, torchaudio, os, pytorch_lightning as pl
from data_utils import mel_spectrogram

class LibrispeechDataset(torch.utils.data.Dataset):
    """
    A wrapper for the Librispeech dataset that returns the audio features and text.
    
    Args:
        dataset: a torch.utils.data.Dataset object in HuggingFace format
        text_transform: a TextTransform object that converts text to integers
        mfcc_norm: an MFCCNormalizer object that normalizes MFCCs
    """
    def __init__(self, dataset, text_transform, mfcc_norm):
        self.dataset = dataset
        self.text_transform = text_transform
        self.mfcc_norm = mfcc_norm
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, index):
        "Reproduce the audio preprocessing from Gaddy on Librispeech data"
        audio = self.dataset[index]['audio']['array']
        text = self.dataset[index]['text']
        audio = librosa.resample(audio, orig_sr=16000, target_sr=22050)
        audio = np.clip(audio, -1, 1) # because resampling sometimes pushes things out of range
        # window is 1024, hop is 256, so length of output is (len(audio) - 1024) // 256 + 1
        # (or at least that's what co-pilot says)
        pytorch_mspec = mel_spectrogram(torch.tensor(audio, dtype=torch.float32).unsqueeze(0),
                                        1024, 80, 22050, 256, 1024, 0, 8000, center=False)
        mfccs = pytorch_mspec.squeeze(0).T.numpy()
        mfccs = self.mfcc_norm.normalize(mfccs)
        text_int = np.array(self.text_transform.text_to_int(text), dtype=np.int64)
        example = {'audio_features': torch.from_numpy(mfccs),
            'text': text,
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
    silent = []
    audio_only = []
    for example in batch:
        audio_features.append(example['audio_features'])
        audio_feature_lengths.append(example['audio_features'].shape[0])
        text_int.append(example['text_int'])
        text_int_lengths.append(example['text_int'].shape[0])
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
        # 'phonemes':phonemes,
        'raw_emg_lengths': raw_emg_lengths,
        'silent': silent,
        'audio_only': audio_only,
        'text': text,
        'text_int': text_int,
        'text_int_lengths': text_int_lengths,
    }

class StratifiedBatchSampler(torch.utils.data.Sampler):
    """"Given the class of each example, sample batches without replacement
    with desired proportions of each class.
    
    If we run out of examples of a given class, we stop yielding batches.
    
    Args:
        classes: array of class labels for each example
        class_proportion: array of desired proportion of each class in each batch
        batch_size: number of examples in each batch
        shuffle: whether to shuffle the examples before sampling
        
    >>> x = np.arange(17)
    >>> classes = np.array([0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1])
    >>> for i in StratifiedBatchSampler(classes, np.array([0.5, 0.5]), 4, shuffle=False):
    ...   print(f"{x[i]=} {classes[i]=}")
    x[i]=array([0, 1, 5, 6]) classes[i]=array([0, 0, 1, 1])
    x[i]=array([2, 3, 7, 8]) classes[i]=array([0, 0, 1, 1])
    """
    def __init__(self, classes:np.ndarray, class_proportion:np.ndarray,
                 batch_size:int, shuffle:bool=True):        
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
        self.num_batches = int(np.min(self.num_examples_per_class // (batch_size * class_proportion)))
        
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

class EMGAndSpeechModule(pl.LightningDataModule):
    def __init__(self, emg_data_module:pl.LightningDataModule,
            speech_train:torch.utils.data.Dataset, speech_val:torch.utils.data.Dataset,
            speech_test:torch.utils.data.Dataset,
            bz:int=64,
            batch_class_proportions:np.ndarray=np.array([0.08, 0.42, 0.5])
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

        emg_train = emg_data_module.train
        self.train = torch.utils.data.ConcatDataset([
            emg_train, speech_train
        ])
        train_emg_len = len(emg_train)
        
        self.val = emg_data_module.val
        # self.val = torch.utils.data.ConcatDataset([
        #     emg_data_module.val, speech_val
        # ])
        self.val_emg_len = len(self.val)

        self.test  = emg_data_module.test        
        # self.test = torch.utils.data.ConcatDataset([
        #     emg_data_module.test, speech_test
        # ])
        self.test_emg_len = len(self.test)
        
        # 0: EMG only (silent), 1: EMG & Audio, 2: Audio only
        classes = np.concatenate([np.zeros(train_emg_len), 2 * np.ones(len(speech_train))])
        for i,b in enumerate(emg_train):
            if not b['silent']:
                classes[i] = 1
    
        self.batch_sampler = StratifiedBatchSampler(classes, batch_class_proportions, bz)
        self.collate_fn = collate_gaddy_or_speech
        self.bz = bz
        
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train,
            collate_fn=self.collate_fn,
            pin_memory=True,
            batch_sampler=self.batch_sampler
        )
        
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val,
            collate_fn=self.collate_fn,
            pin_memory=True,
            batch_size=1
        )
        
    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test,
            collate_fn=self.collate_fn,
            batch_size=1
        )