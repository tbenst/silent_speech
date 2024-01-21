import torch, numpy as np, librosa, torchaudio, os, pytorch_lightning as pl
import dill as pickle  # dill can serialize local classes
from data_utils import mel_spectrogram, read_phonemes, TextTransform
import torch.distributed as dist, sys, logging
import scipy, glob, random
from tqdm import tqdm
from functools import partial
from joblib import Memory
from typing import List, Tuple
from collections import defaultdict
from joblib import Parallel, delayed


def persist_to_file(file_name):
    # TODO: should open file only when function called not initialized
    cache = {}

    def decorator(original_func):
        def new_func(*args, **kwargs):
            if not "k" in cache:
                try:
                    # load cache from disk
                    with open(file_name, "rb") as f:
                        cache["k"] = pickle.load(f)
                        logging.warn(f"Loaded cache from {file_name}")
                except:
                    # populate cache in memory & save to disk
                    cache["k"] = original_func(*args, **kwargs)
                    with open(file_name, "wb") as f:
                        pickle.dump(cache["k"], f)
            return cache["k"]

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
        speaker_id = item["speaker_id"]
        chapter_id = item["chapter_id"]
        id = item["id"]
        for alignment_dir in self.alignment_dirs:
            textgrid_path = os.path.join(
                alignment_dir, str(speaker_id), str(chapter_id), f"{id}.TextGrid"
            )
            if os.path.exists(textgrid_path):
                return textgrid_path
        else:
            raise ValueError(
                f"Could not find TextGrid for {speaker_id}/{chapter_id}/{id}"
            )

    def __getitem__(self, index):
        "Reproduce the audio preprocessing from Gaddy on Librispeech data"
        item = self.dataset[index]
        audio = item["audio"]["array"]
        text = item["text"]

        audio = librosa.resample(audio, orig_sr=16000, target_sr=22050)
        audio = np.clip(
            audio, -1, 1
        )  # because resampling sometimes pushes things out of range

        # window is 1024, hop is 256, so length of output is (len(audio) - 1024) // 256 + 1
        # (or at least that's what co-pilot says)
        pytorch_mspec = mel_spectrogram(
            torch.tensor(audio, dtype=torch.float32).unsqueeze(0),
            1024,
            80,
            22050,
            256,
            1024,
            0,
            8000,
            center=False,
        )
        mfccs = pytorch_mspec.squeeze(0).T.numpy()
        mfccs = self.mfcc_norm.normalize(mfccs)
        text_int = np.array(self.text_transform.text_to_int(text), dtype=np.int64)

        textgrid_path = self.get_textgrid_path(item)
        phonemes = read_phonemes(textgrid_path, max_len=mfccs.shape[0])

        example = {
            "audio_features": torch.from_numpy(mfccs),
            "text": text,
            "phonemes": torch.from_numpy(phonemes),
            # 'text': np.array([text]), # match Gaddy's format. seems unnecessary though, why not just str..?
            "text_int": torch.from_numpy(text_int),
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
            audio_features.append(example["audio_features"])
            audio_feature_lengths.append(example["audio_features"].shape[0])
            text_int.append(example["text_int"])
            text_int_lengths.append(example["text_int"].shape[0])
            text.append(example["text"])
        return {
            "audio_features": audio_features,
            "audio_feature_lengths": audio_feature_lengths,
            "raw_emg": [None] * batch_size,
            "text_int": text_int,
            "text_int_lengths": text_int_lengths,
            "text": text,
            "silent": [False] * batch_size,
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
        text_int.append(example["text_int"])
        text_int_lengths.append(example["text_int"].shape[0])
        phonemes.append(example["phonemes"])
        if type(example["text"]) == np.ndarray:
            text.append(example["text"][0])  # use a string instead of array([string])
        else:
            text.append(example["text"])
        if "raw_emg" not in example:
            # audio only
            audio_features.append(example["audio_features"])
            audio_feature_lengths.append(example["audio_features"].shape[0])
            parallel_emg.append(None)
            parallel_emg_lengths.append(0)
            raw_emg.append(None)
            raw_emg_lengths.append(0)
            silent.append(False)
            audio_only.append(True)
        else:
            raw_emg.append(example["raw_emg"])
            raw_emg_lengths.append(example["raw_emg"].shape[0])
            audio_only.append(False)

            if example["silent"]:
                # don't use annoying array([[1]], dtype=uint8)
                silent.append(True)
                audio_features.append(example["parallel_voiced_audio_features"])
                audio_feature_lengths.append(
                    example["parallel_voiced_audio_features"].shape[0]
                )
                # logging.debug(f'append parrallel emg {example["parallel_voiced_raw_emg"].shape}')
                parallel_emg.append(example["parallel_voiced_raw_emg"])
                parallel_emg_lengths.append(example["parallel_voiced_raw_emg"].shape[0])
            else:
                silent.append(False)
                audio_features.append(example["audio_features"])
                audio_feature_lengths.append(example["audio_features"].shape[0])
                parallel_emg.append(None)
                parallel_emg_lengths.append(0)

    return {
        "audio_features": audio_features,
        "audio_feature_lengths": audio_feature_lengths,
        "raw_emg": raw_emg,
        "raw_emg_lengths": raw_emg_lengths,
        "parallel_raw_emg": parallel_emg,
        "parallel_raw_emg_lengths": parallel_emg_lengths,
        "phonemes": phonemes,
        "silent": silent,
        "audio_only": audio_only,
        "text": text,
        "text_int": text_int,
        "text_int_lengths": text_int_lengths,
    }


def collate_gaddy_speech_or_neural(batch):
    batch_size = len(batch)
    audio_features = []
    audio_feature_lengths = []
    neural_features = []
    neural_feature_lengths = []
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
    has_neural = []
    sessions = []
    for example in batch:
        text_int.append(example["text_int"])
        text_int_lengths.append(example["text_int"].shape[0])
        phonemes.append(example["phonemes"])
        sessions.append(example["session"])
        if type(example["text"]) == np.ndarray:
            text.append(example["text"][0])  # use a string instead of array([string])
        else:
            text.append(example["text"])
        if "neural_features" in example:
            # T12 data
            if example["audio_features"] is None:
                pass
                # INFO: be careful here as indexing now thrown off if
                # we don't append None for eg parallel / cross contrastive
                # TODO: fix this
                audio_features.append(None)
                audio_feature_lengths.append(0)
            else:
                audio_features.append(example["audio_features"])
                audio_feature_lengths.append(example["audio_features"].shape[0])
            neural_features.append(example["neural_features"])
            neural_feature_lengths.append(example["neural_features"].shape[0])
            parallel_emg.append(None)
            parallel_emg_lengths.append(0)
            raw_emg.append(None)
            raw_emg_lengths.append(0)
            silent.append(False)
            audio_only.append(False)
            has_neural.append(True)

        elif "raw_emg" not in example:
            # audio only
            audio_features.append(example["audio_features"])
            audio_feature_lengths.append(example["audio_features"].shape[0])
            neural_features.append(None)
            neural_feature_lengths.append(None)
            parallel_emg.append(None)
            parallel_emg_lengths.append(0)
            raw_emg.append(None)
            raw_emg_lengths.append(0)
            silent.append(False)
            audio_only.append(True)
            has_neural.append(False)
        else:
            raw_emg.append(example["raw_emg"])
            raw_emg_lengths.append(example["raw_emg"].shape[0])
            audio_only.append(False)
            has_neural.append(False)
            neural_features.append(None)
            neural_feature_lengths.append(None)

            if example["silent"]:
                # don't use annoying array([[1]], dtype=uint8)
                silent.append(True)
                audio_features.append(example["parallel_voiced_audio_features"])
                audio_feature_lengths.append(
                    example["parallel_voiced_audio_features"].shape[0]
                )
                # logging.debug(f'append parrallel emg {example["parallel_voiced_raw_emg"].shape}')
                parallel_emg.append(example["parallel_voiced_raw_emg"])
                parallel_emg_lengths.append(example["parallel_voiced_raw_emg"].shape[0])
            else:
                silent.append(False)
                audio_features.append(example["audio_features"])
                audio_feature_lengths.append(example["audio_features"].shape[0])
                parallel_emg.append(None)
                parallel_emg_lengths.append(0)

    return {
        "audio_features": audio_features,
        "audio_feature_lengths": audio_feature_lengths,
        "neural_features": neural_features,
        "neural_feature_lengths": neural_feature_lengths,
        "raw_emg": raw_emg,
        "raw_emg_lengths": raw_emg_lengths,
        "parallel_raw_emg": parallel_emg,
        "parallel_raw_emg_lengths": parallel_emg_lengths,
        "phonemes": phonemes,
        "silent": silent,
        "audio_only": audio_only,
        "has_neural": has_neural,
        "sessions": sessions,
        "text": text,
        "text_int": text_int,
        "text_int_lengths": text_int_lengths,
    }


def split_batch_into_emg_neural_audio(batch):
    # Can compose with collate_gaddy_or_speech
    emg = []
    emg_phonemes = []
    length_emg = []
    y_length_emg = []
    y_emg = []
    text_emg = []

    audio = []
    audio_phonemes = []
    length_audio = []
    y_length_audio = []
    y_audio = []
    text_audio = []

    neural = []
    neural_phonemes = []
    length_neural = []
    y_length_neural = []
    y_neural = []
    text_neural = []

    paired_emg_idx = []  # simultaneous emg + audio
    paired_audio_idx = []  # same length as paired_emg_idx

    silent_emg_idx = []  # silent emg
    parallel_audio_idx = []  # vocalized audio (parallel recordind with silent emg)
    parallel_emg_idx = []  # vocalized emg (parallel recordind with silent emg)

    # support other data collator
    if "audio_only" in batch:
        audio_only = batch["audio_only"]
    else:
        audio_only = [False] * len(batch["silent"])

    if "has_neural" in batch:
        has_neural = batch["has_neural"]
    else:
        has_neural = [False] * len(batch["silent"])

    for i, (s, a, n) in enumerate(zip(batch["silent"], audio_only, has_neural)):
        aud = batch["audio_features"][i]
        # logging.debug(f"{type(batch['phonemes'])=}")
        if n:
            # T12 neural data
            neural.append(batch["neural_features"][i])
            length_neural.append(batch["neural_feature_lengths"][i])
            y_length_neural.append(batch["text_int_lengths"][i])
            text_neural.append(batch["text"][i])
            y_neural.append(batch["text_int"][i])
            neural_phonemes.append(batch["phonemes"][i])

        elif not a:
            # Not audio only
            if s:
                # Silent EMG + parallel AUDIO + parallel EMG
                parallel_emg_idx.append(len(emg))
                emg.append(batch["parallel_raw_emg"][i])
                length_emg.append(batch["parallel_raw_emg_lengths"][i])
                y_length_emg.append(batch["text_int_lengths"][i])
                text_emg.append(batch["text"][i])
                y_emg.append(batch["text_int"][i])
                emg_phonemes.append(batch["phonemes"][i])

                # we append the silent emg data down below
                silent_emg_idx.append(len(emg))
                parallel_audio_idx.append(len(audio))
                assert aud is not None

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
                logging.debug(
                    f"appending these idxs for emg, audio: {len(emg)}, {len(audio)}"
                )
                # print(f"appending these idxs for emg, audio: {len(emg)}, {len(audio)}")
                # we append the vocalized/simultaneous/paired emg data down below
                paired_emg_idx.append(len(emg))
                paired_audio_idx.append(len(audio))

            emg.append(batch["raw_emg"][i])
            length_emg.append(batch["raw_emg_lengths"][i])
            y_length_emg.append(batch["text_int_lengths"][i])
            text_emg.append(batch["text"][i])
            y_emg.append(batch["text_int"][i])
            emg_phonemes.append(batch["phonemes"][i])

        if aud is not None:
            audio.append(aud)
            length_audio.append(batch["audio_feature_lengths"][i])
            y_length_audio.append(batch["text_int_lengths"][i])
            text_audio.append(batch["text"][i])
            y_audio.append(batch["text_int"][i])
            audio_phonemes.append(batch["phonemes"][i])

    emg_tup = (emg, length_emg, emg_phonemes, y_length_emg, y_emg, text_emg)
    neural_tup = (
        neural,
        length_neural,
        neural_phonemes,
        y_length_neural,
        y_neural,
        text_neural,
    )
    audio_tup = (
        audio,
        length_audio,
        audio_phonemes,
        y_length_audio,
        y_audio,
        text_audio,
    )
    idxs = (
        paired_emg_idx,
        paired_audio_idx,
        silent_emg_idx,
        parallel_emg_idx,
        parallel_audio_idx,
    )
    return emg_tup, neural_tup, audio_tup, idxs


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
    if cache_path[-4:] == ".pkl":
        assert (
            not per_index_cache
        ), "cache_path must be a directory if per_index_cache=True"
        instance_path = cache_path
    else:
        assert (
            per_index_cache
        ), "cache_path must be a .pkl file if per_index_cache=False"
        instance_path = os.path.join(cache_path, "instance.pkl")

    if os.path.isfile(instance_path):
        # load cached instance & return closure
        def wrapper(*args, **kwargs):
            with open(instance_path, "rb") as f:
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

            cached_attrs = [
                "cache_path",
                "per_index_cache",
                "cache",
                "approximate_memory_usage",
                "populate_cache",
                "len",
                "cache_each_index_to_disk",
            ]
            for a in cached_attrs:
                if hasattr(super(), a):
                    logging.warning(
                        f"{Dataset} already has attribute '{a}'. CachedDataset will clobber."
                    )

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
            with open(instance_path, "wb") as f:
                pickle.dump(self, f)

        def approximate_memory_usage(self):
            sz = len(
                pickle.dumps(super().__getitem__(0), protocol=pickle.HIGHEST_PROTOCOL)
            )
            gb = self.len * sz / 1e9
            print("Approximate memory usage of dataset: {} GB".format(gb))
            return gb

        def populate_cache(self):
            self.approximate_memory_usage()
            # __getitem__ can be expensive, so we cache the whole dataset once
            for i in tqdm(range(self.len), desc="Caching dataset", total=self.len):
                try:
                    data = super().__getitem__(i)
                    self.cache.append(data)
                except Exception as e:
                    print(e)
                    print(f"Failed to cache index {i}, skipping.")

        def cache_each_index_to_disk(self):
            self.approximate_memory_usage()
            save_idx = 0
            for i in tqdm(range(self.len), desc="Caching each index", total=self.len):
                # Librispeech is missing some aligned phonemes, so we need to skip those
                # this does mean the cache will be smaller than the dataset, and some
                # indices may not match up
                try:
                    data = super().__getitem__(i)
                    idx_path = os.path.join(self.cache_path, f"{save_idx}.pkl")
                    with open(idx_path, "wb") as f:
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
                with open(idx_path, "rb") as f:
                    return pickle.load(f)
            else:
                return self.cache[index]

    return CachedDataset


class StratifiedBatchSampler(torch.utils.data.Sampler):
    """ "Given the class of each example, sample batches without replacement
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

    def __init__(
        self,
        classes: np.ndarray,
        class_proportion: np.ndarray,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
    ):
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
        self.class_n_per_batch = np.round(self.class_proportion * batch_size).astype(
            int
        )
        assert (
            self.class_n_per_batch.sum() == batch_size
        ), "Class proportion must evenly divide batch size"

        # When we ran out of examples of a given class, we stop yielding
        self.num_batches = int(
            np.floor(np.min(self.num_examples_per_class / self.class_n_per_batch))
        )
        self.drop_last = drop_last  # not used

        self.epoch = 0  # not used by base class

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        if self.shuffle:
            self.class_indices = [np.random.permutation(x) for x in self.class_indices]
        # num_batches is set to ensure we have enough examples of each class
        for batch in range(self.num_batches):
            batch_indices = []
            for i in range(self.class_n_per_batch.shape[0]):
                # we add n of this class to the batch
                s = batch * self.class_n_per_batch[i]
                e = (batch + 1) * self.class_n_per_batch[i]
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

    def __init__(
        self,
        classes: np.ndarray,
        lengths: np.ndarray,
        class_proportion: np.ndarray,
        batch_size: int,
        max_len: int,
        shuffle: bool = True,
    ):
        raise NotImplementedError(
            "This is not yet tested. update to match DistributedSizeAwareStratifiedBatchSampler"
        )
        super().__init__(classes, class_proportion, batch_size, shuffle)
        self.max_len = max_len
        self.lengths = lengths
        self.mini_batch_classes = np.concatenate(
            [
                np.full(self.class_n_per_batch[i], i)
                for i in range(self.class_n_per_batch.shape[0])
            ]
        )

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
                    logging.warning(
                        f"Warning: example {idx} cannot fit within desired batch length"
                    )
                if length + batch_length > self.max_len:
                    yield batch
                    batch = []
                    batch_length = 0
                batch.append(idx)
                batch_length += length
            else:  # no break so we continue while loop
                # https://stackoverflow.com/a/3150107
                continue
            break  # break out of while loop when we run out of examples


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

    # TODO: perhaps refactor out the Distributed code into a separate class
    def __init__(
        self,
        classes: np.ndarray,
        class_proportion: np.ndarray,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 61923,
        num_replicas: int = None,
    ):
        if num_replicas is None:
            raise ValueError("num_replicas must be specified")
        # self.num_replicas = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
        self.num_replicas = num_replicas
        assert (
            batch_size % self.num_replicas == 0
        ), "Batch size must be divisible by number of GPUs"
        # we consider the batch_size to be the overall batch size, not the per-GPU batch size
        # internal_bz is the batch size per GPU
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
            self.class_indices = [
                x[torch.randperm(len(x), generator=g).tolist()]
                for x in self.class_indices
            ]
        # we split the dataset into num_replicas splits, and each GPU gets a different split
        for batch in range(self.rank, self.num_batches, self.num_replicas):
            batch_indices = []
            for i in range(self.class_n_per_batch.shape[0]):
                s = batch * self.class_n_per_batch[i]
                e = (batch + 1) * self.class_n_per_batch[i]
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

    batch_size has a somewhat complex relationship with class_n_per_batch
    and max_len. We want to ensure that each batch has the desired proportion
    of each class, but we also want to ensure that each batch is not too long.
    We do this by first sampling a batch of size batch_size that is class-
    balanced, then sampling from that batch to ensure that we construct a
    minibatch that is not too long.

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

    def __init__(
        self,
        classes: np.ndarray,
        lengths: np.ndarray,
        class_proportion: np.ndarray,
        batch_size: int,
        max_len: int,
        shuffle: bool = True,
        seed: int = 61923,
        num_replicas: int = None,
        constant_num_batches: bool = True,
        always_include_class: int = None,
    ):
        super().__init__(
            classes,
            class_proportion,
            batch_size,
            shuffle,
            seed=seed,
            num_replicas=num_replicas,
        )
        self.max_len = max_len
        self.lengths = lengths
        self.always_include_class = always_include_class
        if always_include_class is not None:
            # prevent oversampling
            self.class_n_per_batch[self.always_include_class] -= 1
        # desired class labels for each mini batch, eg: [0, 0, 1, 2, 2, 2]
        self.mini_batch_classes = torch.from_numpy(
            np.concatenate(
                [
                    np.full(self.class_n_per_batch[i], i)
                    for i in range(self.class_n_per_batch.shape[0])
                ]
            )
        )
        self.len = None

        self.constant_num_batches = False
        if constant_num_batches:
            self.hardcode_len = self.min_len(200)  # assume 200 epochs
            self.constant_num_batches = True
            logging.warning(
                f"Hard coding len to {self.hardcode_len} as hack to get pytorch lightning to work, assuming 200 epochs"
            )

    def min_len(self, num_epochs: int):
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
            class_indices = [
                x[torch.randperm(len(x), generator=g)].tolist()
                for x in self.class_indices
            ]
            # different indices per GPU
            class_indices = [x[rank :: self.num_replicas] for x in class_indices]
        else:
            class_indices = [x.tolist() for x in self.class_indices]

        batch = []  # resets each time over max_len
        batch_length = 0
        batches = []  # accumulates

        while True:
            if self.shuffle:
                p = torch.randperm(self.mini_batch_classes.shape[0], generator=g)
                mini_batch_classes = self.mini_batch_classes[p]
            else:
                mini_batch_classes = self.mini_batch_classes

            if self.always_include_class is not None:
                mini_batch_classes = np.concatenate(
                    [[self.always_include_class], mini_batch_classes]
                )

            for cl in mini_batch_classes:
                if len(class_indices[cl]) == 0:
                    # stop yielding batches if we run out of examples of a given class
                    # break
                    self.len = len(batches)
                    # logging.warning(f"DEBUG:return {self.len} batches. {self.epoch=}")
                    # logging.warning(f"DEBUG: {batches[10]=}, {batches[11]=}, {batches[12]=}")
                    avg_num_ex = np.mean([len(x) for x in batches])
                    logging.debug(
                        f"Average number of examples per batch: {avg_num_ex} for epoch {self.epoch} and rank {rank}"
                    )
                    # if self.len < self.hardcode_len:
                    #     logging.warning(f"Warning: returning {self.len} batches, which is less than hardcode_len {self.hardcode_len}")
                    if self.constant_num_batches:
                        return iter(batches[: self.hardcode_len])
                    else:
                        return iter(batches)
                # class_indices shrink as we pop from them
                idx = class_indices[cl].pop()
                length = self.lengths[idx]
                if length > self.max_len:
                    logging.warning(
                        f"Warning: example {idx} cannot fit within desired batch length, skipping"
                    )
                    continue
                if length + batch_length > self.max_len:
                    # add batch before it gets too long
                    batches.append(batch)
                    batch = []
                    batch_length = 0
                    if self.always_include_class is not None:
                        break  # ensure we always include at least one example from this class
                        # this means we will drop this index, but next batch will resample in
                        # a proportionally correct way
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
        length_per_class = np.array(
            [
                np.sum(np.array(self.lengths)[class_indices[i]])
                for i in range(self.class_n_per_batch.shape[0])
            ]
        )
        # logging.warning(f'length_per_class: {length_per_class}')
        # num batches limiting class
        num_batches_per_class = np.ceil(
            length_per_class / self.class_proportion / self.max_len
        )
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


@persist_to_file(f"/lscratch/tbenst/2024-01-20c_emg_speech_dset_lengths.pkl")
def emg_speech_dset_lengths(dset: torch.utils.data.Dataset):
    """Calculate length of latent space for each example in dataset.

    Useful as contrastive loss is quadratic in length of latent space.
    """
    lengths = []
    for d in tqdm(dset, desc="calc lengths for sampler"):
        if "silent" in d:
            new_len = d["raw_emg"].shape[0] // 8
            if d["silent"]:
                # will do forward pass on all of this data
                new_len += (
                    d["parallel_voiced_raw_emg"].shape[0]
                    + d["parallel_voiced_audio_features"].shape[0]
                )
            else:
                new_len += d["audio_features"].shape[0]
            lengths.append(new_len)
        elif "raw_emg" not in d:
            # audio only
            # same dim as latent space, no need to divide by 8
            lengths.append(d["audio_features"].shape[0])
        else:
            # perhaps will crash neural now
            raise ValueError("Unknown dataset format")
            # EMG + audio, so length is sum of both
            emg_z_len = d["raw_emg"].shape[0] // 8
            audio_z_len = d["audio_features"].shape[0]
            assert emg_z_len == audio_z_len
            # lengths.append(emg_z_len + audio_z_len)
            # WARN/TODO: for EMG only
            lengths.append(emg_z_len)
    return lengths


class EMGAndSpeechModule(pl.LightningDataModule):
    def __init__(
        self,
        emg_train: torch.utils.data.Dataset,
        emg_val: torch.utils.data.Dataset,
        emg_test: torch.utils.data.Dataset,
        speech_train: torch.utils.data.Dataset,
        speech_val: torch.utils.data.Dataset,
        speech_test: torch.utils.data.Dataset,
        bz: int = 64,
        val_bz: int = 16,
        num_replicas: int = 1,
        num_workers: int = 0,
        TrainBatchSampler: torch.utils.data.Sampler = StratifiedBatchSampler,
        ValSampler: torch.utils.data.Sampler = None,
        TestSampler: torch.utils.data.Sampler = None,
        batch_class_proportions: np.ndarray = np.array([0.08, 0.42, 0.5]),
        pin_memory: bool = True,
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
        self.train = torch.utils.data.ConcatDataset([emg_train, speech_train])
        train_emg_len = len(emg_train)

        self.val = emg_val
        # self.val = torch.utils.data.ConcatDataset([
        #     emg_data_module.val, speech_val
        # ])
        self.val_emg_len = len(self.val)

        self.test = emg_test
        # self.test = torch.utils.data.ConcatDataset([
        #     emg_data_module.test, speech_test
        # ])
        self.test_emg_len = len(self.test)

        # 0: EMG only (silent), 1: EMG & Audio, 2: Audio only
        classes = np.concatenate(
            [
                np.zeros(train_emg_len, dtype=int),
                2 * np.ones(len(speech_train), dtype=int),
            ]
        )
        for i, b in enumerate(emg_train):
            if not b["silent"]:
                classes[i] = 1

        self.train_lengths = emg_speech_dset_lengths(self.train)
        self.TrainBatchSampler = TrainBatchSampler(
            classes,
            self.train_lengths,
            batch_class_proportions,
            batch_size=bz,
            num_replicas=num_replicas,
        )

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
            batch_sampler=self.TrainBatchSampler,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val,
            collate_fn=self.collate_fn,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            batch_size=self.val_bz,
            sampler=self.val_sampler,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test,
            collate_fn=self.collate_fn,
            batch_size=self.val_bz,
            sampler=self.test_sampler,
        )


class DistributedSizeAwareSampler(torch.utils.data.Sampler):
    """Sample batches of examples from the dataset,
    ensuring that each batch fits within max_len."""

    def __init__(
        self,
        lengths: np.ndarray,
        max_len: int = 256000,
        shuffle: bool = True,
        seed: int = 20230819,
        epoch: int = 0,
        num_replicas: int = 1,
        constant_num_batches: bool = True,
    ):
        self.lengths = lengths
        self.max_len = max_len
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = epoch

        # for distributed training
        rank_key = "RANK" if "RANK" in os.environ else "LOCAL_RANK"
        self.rank = int(os.environ[rank_key]) if rank_key in os.environ else 0
        self.num_replicas = num_replicas

        self.constant_num_batches = False
        if constant_num_batches:
            self.hardcode_len = self.min_len(200)  # assume 200 epochs
            self.constant_num_batches = True
            logging.warning(
                f"Hard coding len to {self.hardcode_len} as hack to get pytorch lightning to work"
            )

    def __iter__(self):
        return self.iter_batches(self.rank)

    def iter_batches(self, rank):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(len(self.lengths), generator=g).tolist()
        indices = indices[rank :: self.num_replicas]
        batch = []
        batch_length = 0
        for idx in indices:
            length = self.lengths[idx]
            if length > self.max_len:
                logging.warning(
                    f"Warning: example {idx} cannot fit within desired batch length"
                )
            if length + batch_length > self.max_len:
                yield batch
                batch = []
                batch_length = 0
            batch.append(idx)
            batch_length += length
        # dropping last incomplete batch

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def min_len(self, num_epochs: int):
        """Minimum number of batches in any epoch."""
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

    def __len__(self):
        "Return approximate number of batches per epoch"
        # https://github.com/Lightning-AI/lightning/issues/18023
        if self.constant_num_batches:
            return self.hardcode_len
        else:
            return len(iter(self))


class NeuralDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        neural,
        audio,
        phonemes,
        sentences,
        text_transform,
        sessions=None,
        white_noise_sd=0,
        constant_offset_sd=0,
        no_audio=False,
    ):
        self.neural = neural
        self.audio = audio
        self.phonemes = phonemes
        self.sentences = sentences
        self.sessions = sessions
        self.text_transform = text_transform
        self.n_features = neural[0].shape[1]
        self.white_noise_sd = white_noise_sd
        self.constant_offset_sd = constant_offset_sd
        self.no_audio = no_audio
        if sessions is not None:
            self.unique_sessions = np.unique(sessions)
        else:
            self.unique_sessions = np.array([])

        super().__init__()

    def __getitem__(self, idx):
        return self.getitem(idx)

    def getitem(self, idx, white_noise_sd=None, constant_offset_sd=None):
        text_int = np.array(
            self.text_transform.text_to_int(self.sentences[idx]), dtype=np.int64
        )
        if self.no_audio:
            aud = None
        else:
            aud = self.audio[idx]
            aud = aud if aud is None else torch.from_numpy(aud)
        phon = self.phonemes[idx]
        phon = phon if phon is None else torch.from_numpy(phon)
        nf = torch.from_numpy(self.neural[idx].copy())
        # TODO: check if this is correct. broadcasting error on constant..?
        white_noise_sd = (
            self.white_noise_sd if white_noise_sd is None else white_noise_sd
        )
        constant_offset_sd = (
            self.constant_offset_sd
            if constant_offset_sd is None
            else constant_offset_sd
        )
        if white_noise_sd > 0:
            nf += torch.randn_like(nf) * white_noise_sd
        if constant_offset_sd > 0:
            nf += torch.randn(self.n_features) * constant_offset_sd
        ret = {
            "audio_features": aud,
            "neural_features": nf,
            "text": self.sentences[idx],
            "text_int": torch.from_numpy(text_int),
            "phonemes": phon,
        }
        if self.sessions is not None:
            ret["session"] = self.sessions[idx]
        else:
            ret["session"] = None
        return ret

    def __len__(self):
        return len(self.neural)


class T12Dataset(NeuralDataset):
    def __init__(
        self,
        t12_npz,
        partition="train",
        no_audio=False,
        audio_type="tts_mspecs",
        white_noise_sd=0,
        constant_offset_sd=0,
    ):
        #  audio_type="tts_mspecs"):
        """T12 BCI dataset.

        partition: train or test
        audio_type: mspecs, tts_mspecs, or aligned_tts_mspecs

        """
        idx = np.where(t12_npz["dataset_partition"] == partition)[0]
        neural = []
        audio = []
        spikePow = t12_npz["spikePow"]
        tx1 = t12_npz["tx1"]
        tx2 = t12_npz["tx2"]
        tx3 = t12_npz["tx3"]
        tx4 = t12_npz["tx4"]
        # mean, variance per block
        aud = t12_npz[audio_type]
        for i in tqdm(idx, desc="concatenating neural data"):
            # block_idx = t12_npz["block"][i][0]
            # session = t12_npz["session"][i]
            # print(session, block_idx)

            neural.append(
                np.concatenate(
                    [
                        # np.log10(spikePow[i][:,:128]+1) / 4, # map to approx 0-1
                        spikePow[i][:, :128],
                        tx1[i][:, :128],
                        # tx1[i][:,:128] / 25, # max val is 56
                        # tx2[i] / 25,
                        # tx3[i] / 25,
                        # tx4[i] / 25
                    ],  # max val is 52
                    axis=1,
                ).astype(np.float32)
            )
            if aud[i] is None:
                # for example, if audio_type is "mspecs" then we have no
                # audio for the silent trials
                if audio_type == "tts_mspecs":
                    print(f"WARNING: no audio for index {i}")
                audio.append(None)
            else:
                audio.append((aud[i] + 5) / 5)  # TODO: match librispeech
        phonemes = t12_npz["aligned_phonemes"][idx]
        sentences = t12_npz["sentences"][idx]
        sessions = t12_npz["session"][idx]
        text_transform = TextTransform(togglePhones=False)
        super().__init__(
            neural,
            audio,
            phonemes,
            sentences,
            text_transform,
            sessions=sessions,
            white_noise_sd=white_noise_sd,
            constant_offset_sd=constant_offset_sd,
            no_audio=no_audio,
        )


class T12DataModule(pl.LightningDataModule):
    def __init__(
        self,
        t12_npz,
        audio_type="tts_mspecs",
        max_len=32000,
        num_replicas=1,
        train_bz: int = 32,
        val_bz: int = 16,
        fixed_length=False,
        white_noise_sd=1.0,
        constant_offset_sd=0.2,
        no_audio=True,
    ):
        super().__init__()
        self.train = T12Dataset(
            t12_npz,
            partition="train",
            audio_type="tts_mspecs",
            white_noise_sd=white_noise_sd,
            constant_offset_sd=constant_offset_sd,
            no_audio=no_audio,
        )
        self.val = T12Dataset(
            t12_npz, partition="test", audio_type="tts_mspecs", no_audio=no_audio
        )
        self.collate_fn = collate_gaddy_speech_or_neural

        self.train_bz = train_bz
        self.val_bz = val_bz
        self.fixed_length = fixed_length

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train,
            collate_fn=self.collate_fn,
            pin_memory=True,
            num_workers=0,
            batch_size=self.train_bz,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val,
            collate_fn=self.collate_fn,
            pin_memory=True,
            num_workers=0,
            batch_size=self.val_bz,
        )

    def test_dataloader(self):
        return None


class T12CompDataset(NeuralDataset):
    def __init__(
        self,
        mat_files,
        white_noise_sd=0,
        constant_offset_sd=0,
        togglePhones=False,
        smoothing_sigma=0,
    ):
        #  audio_type="tts_mspecs"):
        """T12 BCI dataset.

        partition: train or test
        audio_type: mspecs, tts_mspecs, or aligned_tts_mspecs

        """
        sentences = []
        neural = []
        sessions = []
        for f in tqdm(mat_files):
            mat_file = scipy.io.loadmat(f)
            blocks = mat_file["blockIdx"].squeeze()
            # INFO: in commit 4e30582db8abdb71bceb92b56879d972a637d4af,
            # we used the last 20 sentences for z-scoring.
            # TODO: z-score on each block
            # https://github.com/fwillett/speechBCI/blob/ba3440432893e75d9413e55ed15e8a6d31034f9b/AnalysisExamples/makeTFRecordsFromSession.py#L51https://github.com/fwillett/speechBCI/blob/ba3440432893e75d9413e55ed15e8a6d31034f9b/AnalysisExamples/makeTFRecordsFromSession.py#L51
            n_trials = len(mat_file["sentenceText"])

            spikePows = []
            tx1s = []
            for i in range(n_trials):
                spikePow = mat_file["spikePow"].squeeze()[i][:, :128]
                tx1 = mat_file["tx1"].squeeze()[i][:, :128]
                spikePows.append(spikePow)
                tx1s.append(tx1)

            for block in np.unique(blocks):
                idxs = np.where(blocks == block)[0]
                spikepow_mean = np.mean(
                    np.concatenate([spikePows[i] for i in idxs], axis=0),
                    keepdims=True,
                    axis=0,
                )
                spikepow_std = (
                    np.std(
                        np.concatenate([spikePows[i] for i in idxs], axis=0),
                        keepdims=True,
                        axis=0,
                    )
                    + 1e-8
                )
                tx1_mean = np.mean(
                    np.concatenate([tx1s[i] for i in idxs], axis=0),
                    keepdims=True,
                    axis=0,
                )
                tx1_std = (
                    np.std(
                        np.concatenate([tx1s[i] for i in idxs], axis=0),
                        keepdims=True,
                        axis=0,
                    )
                    + 1e-8
                )
                for i in idxs:
                    spikePow = spikePows[i]
                    tx1 = tx1s[i]
                    spikePow = (spikePow - spikepow_mean) / spikepow_std
                    if smoothing_sigma > 0:
                        spikePow = scipy.ndimage.gaussian_filter1d(
                            spikePow, sigma=smoothing_sigma, axis=0
                        )
                    tx1 = (tx1 - tx1_mean) / tx1_std
                    if smoothing_sigma > 0:
                        tx1 = scipy.ndimage.gaussian_filter1d(
                            tx1, sigma=smoothing_sigma, axis=0
                        )
                    neural.append(
                        np.concatenate(
                            [
                                spikePow,
                                tx1,
                            ],
                            axis=1,
                        ).astype(np.float32)
                    )
                    sentences.append(mat_file["sentenceText"][i].rstrip())
                    sessions.append(os.path.split(f)[-1])

        audio = [None] * len(neural)
        phonemes = [None] * len(neural)
        text_transform = TextTransform(togglePhones=togglePhones)
        super().__init__(
            neural,
            audio,
            phonemes,
            sentences,
            text_transform,
            sessions=sessions,
            white_noise_sd=white_noise_sd,
            constant_offset_sd=constant_offset_sd,
            no_audio=True,
        )


class T12CompDataModule(pl.LightningDataModule):
    def __init__(
        self,
        datadir,
        train_bz: int = 32,
        val_bz: int = 16,
        fixed_length=False,
        white_noise_sd=1.0,
        constant_offset_sd=0.2,
        smoothing_sigma=0,
        no_audio=True,
        togglePhones=False,
    ):
        super().__init__()

        train_files = glob.glob(datadir + "*/train/*")
        test_files = glob.glob(datadir + "*/test/*")

        self.train = T12CompDataset(
            train_files,
            white_noise_sd=white_noise_sd,
            constant_offset_sd=constant_offset_sd,
            togglePhones=togglePhones,
            smoothing_sigma=smoothing_sigma,
        )
        self.val = T12CompDataset(
            test_files, togglePhones=togglePhones, smoothing_sigma=smoothing_sigma
        )
        self.collate_fn = collate_gaddy_speech_or_neural

        self.train_bz = train_bz
        self.val_bz = val_bz
        self.fixed_length = fixed_length

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train,
            collate_fn=self.collate_fn,
            pin_memory=True,
            num_workers=0,
            batch_size=self.train_bz,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val,
            collate_fn=self.collate_fn,
            pin_memory=True,
            num_workers=0,
            batch_size=self.val_bz,
        )

    def test_dataloader(self):
        return None


# see 2024-01-18_approx_class_balance_bin_packing.py for example usage, etc.
def fill_remaining_bins(
    bins,
    bin_sums,
    idx_per_class,
    lengths,
    max_len,
    class_proportion,
    class_debt,
    generator,
):
    "Try to fill remaining bins with random classes"
    rejected_sample_count = 0

    # keep trying until 100 consecutive rejections
    while rejected_sample_count < 100:
        # randomly sample remaining classes with indices
        valid_classes = [c for c in idx_per_class if len(idx_per_class[c]) > 0]
        if len(valid_classes) == 0:
            break
        # adjust proportion for remaining classes
        # print(f"DEBUG: {valid_classes=}, {class_proportion=}")
        proportion = np.array([class_proportion[c] for c in valid_classes])
        proportion /= proportion.sum()

        sampled_class = seeded_random_choice(valid_classes, proportion, generator)

        # if we can pay off debt, do so and sample again
        if class_debt[sampled_class] > 0:
            class_debt[sampled_class] -= 1
            rejected_sample_count = 0
            continue

        # try to pack sampled_length into a bin
        added, idx_per_class, bins, bin_sums = add_class_to_any_bin(
            sampled_class, idx_per_class, bins, bin_sums, lengths, max_len
        )
        if added:
            rejected_sample_count = 0
        else:
            rejected_sample_count += 1
            continue

    return bins


def add_class_to_bin(c, bin_idx, idx_per_class, bins, bin_sums, lengths, max_len):
    "pop last index of class c, adding to bin, add length to bin_sums"
    if len(idx_per_class[c]) == 0:
        raise ValueError(f"no more items of class {c} to pack")
    length = lengths[idx_per_class[c][-1]]
    if bin_sums[bin_idx] + length > max_len:
        # can't fit this item, stop packing bins
        return False, idx_per_class, bins, bin_sums
    bin_sums[bin_idx] += lengths[idx_per_class[c][-1]]
    bins[bin_idx].append(idx_per_class[c].pop())
    return True, idx_per_class, bins, bin_sums


def test_add_class_to_bin():
    bins = [[]]
    idx_per_class = {"A": [0, 1, 2, 3], "B": []}
    bin_sums = [0]
    lengths = [4, 1, 4, 4]
    max_len = 5

    # Normal case
    success, idx_per_class, bins, bin_sums = add_class_to_bin(
        "A", 0, idx_per_class, bins, bin_sums, lengths, max_len
    )
    assert success
    assert bins == [[3]]

    # Exceeding max length
    success, idx_per_class, bins, bin_sums = add_class_to_bin(
        "A", 0, idx_per_class, bins, bin_sums, lengths, max_len
    )
    assert not success

    # Empty class list
    try:
        add_class_to_bin("B", 0, idx_per_class, bins, bin_sums, lengths, max_len)
    except ValueError as e:
        assert str(e) == "no more items of class B to pack"


def add_class_to_any_bin(c, idx_per_class, bins, bin_sums, lengths, max_len):
    "try to pack class into a bin"
    success = False  # needed for no bins case
    for i in range(len(bins)):
        success, idx_per_class, bins, bin_sums = add_class_to_bin(
            c, i, idx_per_class, bins, bin_sums, lengths, max_len
        )
        if success:
            break
    return success, idx_per_class, bins, bin_sums


def test_add_class_to_any_bin():
    lengths = [3, 5, 2, 2, 1]
    max_len = 5
    idx_per_class = {"A": [0, 1], "B": [2, 3, 4]}
    bins = [[], []]
    bin_sums = [0, 0]

    # Test where addition is successful
    success, idx_per_class, bins, bin_sums = add_class_to_any_bin(
        "A", idx_per_class, bins, bin_sums, lengths, max_len
    )
    assert success

    success, idx_per_class, bins, bin_sums = add_class_to_any_bin(
        "B", idx_per_class, bins, bin_sums, lengths, max_len
    )
    assert success

    success, idx_per_class, bins, bin_sums = add_class_to_any_bin(
        "A", idx_per_class, bins, bin_sums, lengths, max_len
    )
    assert success

    assert bins[0] == [1]
    assert bin_sums[0] == 5
    assert bins[1] == [4, 0]
    assert bin_sums[1] == 4

    # Test where addition is not possible due to size constraints
    success, idx_per_class, bins, bin_sums = add_class_to_any_bin(
        "B", idx_per_class, bins, bin_sums, lengths, max_len
    )
    assert not success

    assert bins[0] == [1]
    assert bin_sums[0] == 5
    assert bins[1] == [4, 0]
    assert bin_sums[1] == 4


def seeded_shuffle(x, generator):
    new = x.copy()
    if type(x) is list:
        is_list = True
        new = np.array(new)
    new = new[torch.randperm(len(x), generator=generator).tolist()]
    if is_list:
        new = new.tolist()
    return new


def seeded_random_choice(x, p, generator):
    "Using torch generator, sample from x with probabilities p"
    return x[torch.multinomial(torch.tensor(p), 1, generator=generator)[0].item()]


def pack_items(
    lengths: List[float],
    classes: List[int],
    max_len: int,
    class_proportion: Tuple[float],
    always_include: List[int],
    shuffle: bool = True,
    seed: int = 20240119,
):
    """
    Greedily and randomly packs each of the N items into a bin (list) of
    max_len, returning a list of bins containing the index of the pertinent
    items. Stops when no more items of a class are left to pack.

    If a class in always_include is low weighting in class_proportion, and
    max_len is small,
    """
    # drop any index that exceeds max_len
    assert (
        type(classes[0]) is int or type(classes[0]) is np.int64
    ), f"{type(classes[0])=}"
    items = []
    new_classes = []
    for i, l in enumerate(lengths):
        if l <= max_len:
            items.append(i)
            new_classes.append(classes[i])

    # group indices by class
    idx_per_class = defaultdict(list)
    for i, c in zip(items, new_classes):
        idx_per_class[c].append(i)

    # shuffle indices within each class
    g = torch.Generator()
    g.manual_seed(seed)
    if shuffle:
        for c in idx_per_class:
            idx_per_class[c] = seeded_shuffle(idx_per_class[c], g)

    valid_classes = np.array(list(set(idx_per_class.keys())))
    assert len(valid_classes) > 0, "no items to pack under max_len"
    bins = []
    bin_sums = []

    # when creating a new bin, since we are always including some classes,
    # we need to compensate for the imbalance in class_proportion
    # by ignoring the random choice until the debt is paid
    class_debt = {c: 0 for c in valid_classes}

    while True:
        # print("DEBUG: start while loop")
        # print(f"DEBUG: {idx_per_class=}, {bins=}")
        stop_loop = False
        for v in idx_per_class.values():
            if len(v) == 0:
                stop_loop = True
        if stop_loop:
            break
        sampled_class = seeded_random_choice(valid_classes, class_proportion, g)

        # check if we can pay off debt
        if class_debt[sampled_class] > 0:
            class_debt[sampled_class] -= 1
            continue

        # try to pack sampled_length into a bin
        added, idx_per_class, bins, bin_sums = add_class_to_any_bin(
            sampled_class, idx_per_class, bins, bin_sums, lengths, max_len
        )

        if not added:
            # create a new bin
            bins.append([])
            bin_sums.append(0)
            # add required classes to bin
            # print(f"DEBUG: {idx_per_class=}")
            for c in always_include:
                # print(f"DEBUG: {c=}, {bins=}")
                success, idx_per_class, bins, bin_sums = add_class_to_bin(
                    c, -1, idx_per_class, bins, bin_sums, lengths, max_len
                )
                if not success:
                    # print(f"DEBUG: {idx_per_class=}, {bins=}")
                    raise ValueError(
                        "hit the unusual edge case that is slightly annoying but solvable"
                    )
                    # unable to add any more valid bins with required classes
                    # TODO
                    # EDGE CASE: if we fail to make a valid bin because of bad luck
                    # eg if always_include = [1, 2] and 1 & 2 don't fit together
                    return fill_remaining_bins(
                        bins[:-1],
                        idx_per_class,
                        lengths,
                        max_len,
                        class_proportion,
                        class_debt,
                        g,
                    )
                if c == sampled_class:
                    # no debt accrues
                    added = True
                    continue
                else:
                    # debt accrues
                    class_debt[c] += 1

        # if the sampled class is not part of always_include, we need to fit it somewhere
        if not added:
            assert (
                sampled_class not in always_include
            ), f"{sampled_class} in always_include"
            # try to pack sampled_length into new bin
            success, idx_per_class, bins, bin_sums = add_class_to_bin(
                sampled_class, -1, idx_per_class, bins, bin_sums, lengths, max_len
            )
            if not success:
                # either this item is annoyingly long, or we got an unlucky
                # draw of new bin. can't pack this item, so push to back
                index = idx_per_class[sampled_class].pop()
                idx_per_class[sampled_class].insert(0, index)

    return fill_remaining_bins(
        bins, bin_sums, idx_per_class, lengths, max_len, class_proportion, class_debt, g
    )


def test_complex_pack_items(
    lengths,
    classes,
    class_proportion,
    always_include,
    max_len,
    max_proportion_error=0.05,
):
    # complex example that mimics gaddy dataset
    N = len(lengths)
    bins = pack_items(lengths, classes, max_len, class_proportion, always_include)
    # check class proportions
    for i in range(3):
        sampled_proportion = np.mean(
            [classes[item] == i for bin in bins for item in bin]
        )
        print(f"{sampled_proportion=}, {class_proportion[i]=}")
        assert (
            abs(sampled_proportion - class_proportion[i]) < max_proportion_error
        ), f"{sampled_proportion=}, {class_proportion[i]=}"

    for bin in bins:
        assert 0 in [classes[item] for item in bin]
        assert 1 in [classes[item] for item in bin]
        assert sum(lengths[item] for item in bin) <= max_len

    bin_classes = [[classes[item] for item in bin] for bin in bins]
    bin_lengths = [sum(lengths[item] for item in bin) for bin in bins]
    N_packed = sum(len(bin) for bin in bins)
    print(f"{bin_classes=}")
    print(f"{bin_lengths=}")
    print(f"{N_packed=}, {N=}, {min(bin_lengths)=}")
    assert (
        min(bin_lengths) >= max_len * 0.85
    ), f"{min(bin_lengths)=}, is too small vs {max_len=}"


def test_pack_items():
    lengths = [2, 3, 1, 4]
    classes = [0, 0, 1, 1]
    max_len = 4
    class_proportion = (0.5, 0.5)
    always_include = [1]

    # Normal case
    bins = pack_items(lengths, classes, max_len, class_proportion, always_include)
    assert all(
        sum(lengths[item] for item in bin) <= max_len for bin in bins
    )  # All bins must respect max_len
    for bin in bins:
        classes_in_bin = [classes[item] for item in bin]
        assert 1 in classes_in_bin, f"All bins must include class 1 {bins=}"

    # I think we may be done..? maybe need a couple more tests

    # Test with items larger than max_len
    lengths = [5, 6, 7]
    classes = [1, 1, 1]
    bins = pack_items(lengths, classes, 5, [1.0], always_include)
    assert len(bins) == 1  # one items should be packed

    # Test with empty class list
    lengths = []
    classes = []
    try:
        bins = pack_items(lengths, classes, max_len, class_proportion, always_include)
    except AssertionError as e:
        assert str(e) == "no items to pack under max_len"

    N = 1000
    test_complex_pack_items(
        lengths=np.random.randint(1, 20, N),
        classes=np.random.randint(0, 3, N),
        class_proportion=[0.08, 0.42, 0.5],
        always_include=[0, 1],
        max_len=100,
    )

    test_complex_pack_items(
        lengths=np.random.randint(1, 20, N),
        # due to these probabilities, we may not pack many bins since we run out of class 0
        classes=np.random.choice([0, 1, 2], N, p=[0.01, 0.09, 0.9]),
        class_proportion=[0.08, 0.42, 0.5],
        always_include=[0, 1],
        max_len=100,
        max_proportion_error=0.1,  # we don't always nail the proportions, but are close enough
    )


class DistributedBatchSampler(torch.utils.data.Sampler):
    def __init__(
        self,
        num_replicas: int,
        constant_num_batches: bool = True,
        num_epochs: int = 200,
    ):
        self.num_replicas = num_replicas if type(num_replicas) is int else 1
        self.constant_num_batches = constant_num_batches
        self.epoch = 0
        self.num_epochs = num_epochs

        self.constant_num_batches = False
        if constant_num_batches:
            self.hardcode_len = self.min_len(num_epochs)
            self.constant_num_batches = True
            logging.warning(
                f"Hard coding len to {self.hardcode_len} as hack to get pytorch lightning to work, assuming 200 epochs"
            )

        rank_key = "RANK" if "RANK" in os.environ else "LOCAL_RANK"
        self.rank = int(os.environ[rank_key]) if rank_key in os.environ else 0

    def min_len(self, num_epochs: int):
        """Minimum number of batches in dataset on any GPU."""
        logging.debug("Calling min_len")
        cur_epoch = self.epoch
        min_length = np.inf

        def batch_length(epoch, rank):
            self.set_epoch(epoch)
            return len(list(self.iter_batches(rank)))

        # Calculate batch lengths in parallel
        results = Parallel(n_jobs=-1)(
            delayed(batch_length)(epoch, rank)
            for epoch in range(num_epochs)
            for rank in range(self.num_replicas)
        )

        min_length = min(results)

        self.set_epoch(cur_epoch)
        return min_length

    def iter_batches(self, rank):
        raise NotImplementedError

    def __iter__(self):
        logging.debug("Initializing DistributedSizeAwareStratifiedBatchSampler")
        return self.iter_batches(self.rank)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self):
        "Return approximate number of batches per epoch"
        # https://github.com/Lightning-AI/lightning/issues/18023
        if self.constant_num_batches:
            return self.hardcode_len
        else:
            return len(iter(self))


class BalancedBinPackingBatchSampler(DistributedBatchSampler):
    """ "Given the class of each example, sample batches without replacement
    with desired proportions of each class using heuristic bin packing.

    classes in always_include are always included in each batch.

    If we run out of examples of a given class, we stop yielding batches.
    """

    def __init__(
        self,
        classes: np.ndarray,
        lengths: np.ndarray,
        class_proportion: np.ndarray,
        max_len: int,
        shuffle: bool = True,
        seed: int = 61923,
        num_replicas: int = None,
        constant_num_batches: bool = True,
        always_include_class: List[int] = [],
        num_epochs: int = 200,
        **kwargs,  # ignore extra arguments
    ):
        assert np.allclose(
            np.sum(class_proportion), 1
        ), f"does not sum to 1: {class_proportion=}"
        assert np.all(class_proportion >= 0)
        assert np.all(class_proportion <= 1)
        assert np.all(np.unique(classes) == np.arange(class_proportion.shape[0]))
        if num_replicas > 1:
            # need to thread seed / torch generator, and also split batches per gpu
            raise NotImplementedError("multiple GPUs not yet supported")

        self.classes = classes
        self.lengths = lengths
        self.max_len = max_len
        self.always_include_class = always_include_class
        self.class_proportion = class_proportion
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        super().__init__(num_replicas, constant_num_batches, num_epochs)

    def iter_batches(self, rank):
        batches = pack_items(
            self.lengths,
            self.classes,
            self.max_len,
            self.class_proportion,
            self.always_include_class,
            shuffle=self.shuffle,
            seed=self.seed + self.epoch,
        )
        avg_num_ex = np.mean([len(x) for x in batches])
        logging.debug(
            f"Average number of examples per batch: {avg_num_ex} for epoch {self.epoch} and rank {rank}"
        )
        if self.constant_num_batches:
            return iter(batches[: self.hardcode_len])
        else:
            return iter(batches)
