##
2
##
# %load_ext autoreload
# %autoreload 2
##
import pytorch_lightning as pl
import os, pickle
import sys
import numpy as np
import logging
import subprocess, torchmetrics
import random
from tqdm.auto import tqdm
from typing import List
from dataclasses import dataclass
import torch
from torch import nn
from torch.utils.data import DistributedSampler
import torch.nn.functional as F

# horrible hack to get around this repo not being a proper python package
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(SCRIPT_DIR)

from read_emg import EMGDataset, SizeAwareSampler, PreprocessedEMGDataset, \
    PreprocessedSizeAwareSampler, EMGDataModule, ensure_folder_on_scratch
from architecture import Model, S4Model, H3Model, ResBlock
from data_utils import combine_fixed_length, decollate_tensor
from transformer import TransformerEncoderLayer
from pytorch_lightning.loggers import NeptuneLogger
# import neptune, shutil
import neptune.new as neptune, shutil
from datetime import datetime
from pytorch_lightning.callbacks import ModelCheckpoint, GradientAccumulationScheduler
from pytorch_lightning.profilers import SimpleProfiler, AdvancedProfiler, PyTorchProfiler, PassThroughProfiler
from pytorch_lightning.strategies import DDPStrategy
from data_utils import TextTransform
from typing import List
from collections import defaultdict
from enum import Enum
from magneto.preprocessing import ensure_data_on_scratch
from dataloaders import LibrispeechDataset, EMGAndSpeechModule, \
    DistributedStratifiedBatchSampler, StratifiedBatchSampler, CachedDataset, \
    split_batch_into_emg_audio, DistributedSizeAwareStratifiedBatchSampler, \
    SizeAwareStratifiedBatchSampler
from functools import partial
from contrastive import cross_contrastive_loss, var_length_cross_contrastive_loss, \
    nobatch_cross_contrastive_loss, supervised_contrastive_loss

DEBUG = False
# DEBUG = True

per_index_cache = True # read each index from disk separately
# per_index_cache = False # read entire dataset from disk


isotime = datetime.now().isoformat()
hostname = subprocess.run("hostname", capture_output=True)
ON_SHERLOCK = hostname.stdout[:2] == b"sh"

# out of date..
# When using 4 GPUs, bz=128, grad_accum=1,
# one epoch takes 4:57 and validation takes 2:51
# unfortunately there is a ton of downtime between epoch so total time is 8:30
# also, we got OOM on GPU 2 at the end of epoch 6

# 1 GPU, bz=24, grad_accum=4, 1 epoch takes 7:40, validation takes 1:50

if DEBUG:
    NUM_GPUS = 1
    limit_train_batches = 2
    limit_val_batches = 2 # will not run on_validation_epoch_end
    # NUM_GPUS = 2
    # limit_train_batches = None
    # limit_val_batches = None
    log_neptune = False
    n_epochs = 2
    # precision = "32"
    precision = "16-mixed"
    num_sanity_val_steps = 2
    grad_accum = 1
    logger_level = logging.DEBUG
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    
else:
    NUM_GPUS = 2
    grad_accum = 3
    if ON_SHERLOCK:
        NUM_GPUS = 4
        grad_accum = 2
    # variable length batches are destroying pytorch lightning
    # limit_train_batches = 900 # validation loop doesn't run at 900 ?! wtf
    # limit_train_batches = 100 # validation loop runs at 100
    # limit_train_batches = 500
    limit_train_batches = None
    limit_val_batches = None
    log_neptune = True
    # log_neptune = False
    n_epochs = 200
    precision = "16-mixed"
    num_sanity_val_steps = 0 # may prevent crashing of distributed training
    # grad_accum = 2 # NaN loss at epoch 67 with BatchNorm, two gpu, grad_accum=2, base_bz=16
    
    # if BatchNorm still causes issues can try RunningBatchNorm (need to implement for distributed)
    # https://youtu.be/HR0lt1hlR6U?t=7543
    logger_level = logging.WARNING


assert os.environ["NEPTUNE_ALLOW_SELF_SIGNED_CERTIFICATE"] == 'TRUE', "run this in shell: export NEPTUNE_ALLOW_SELF_SIGNED_CERTIFICATE='TRUE'"

# load our data file paths and metadata:

if per_index_cache:
    cache_suffix = "_per_index"
else:
    cache_suffix = ""
if ON_SHERLOCK:
    sessions_dir = '/oak/stanford/projects/babelfish/magneto/'
    # TODO: bechmark SCRATCH vs LOCAL_SCRATCH ...?
    # scratch_directory = os.environ["SCRATCH"]
    scratch_directory = os.environ["LOCAL_SCRATCH"]
    gaddy_dir = '/oak/stanford/projects/babelfish/magneto/GaddyPaper/'
else:
    # on my local machine
    sessions_dir = '/data/magneto/'
    scratch_directory = "/scratch"
    gaddy_dir = '/scratch/GaddyPaper/'
    
librispeech_train_cache = os.path.join(scratch_directory, "librispeech",
    "librispeech_train_phoneme_cache")
librispeech_val_cache = os.path.join(scratch_directory, "librispeech",
    "librispeech_val_phoneme_cache")
librispeech_test_cache = os.path.join(scratch_directory, "librispeech",
    "librispeech_test_phoneme_cache")
data_dir = os.path.join(scratch_directory, 'gaddy/')

lm_directory = os.path.join(gaddy_dir, 'pretrained_models/librispeech_lm/')
normalizers_file = os.path.join(SCRIPT_DIR, "normalizers.pkl")
togglePhones = False

if ON_SHERLOCK:
    lm_directory = ensure_folder_on_scratch(lm_directory, scratch_directory)
    lm_directory = ensure_folder_on_scratch(lm_directory, scratch_directory)
    
# bz = 96 # OOM after 25 steps
# bz = 128
# bz = 96
# bz = 64

# OOM w/ 4 GPUs
# bz = 48 # memory usage is massive on GPU 0 (32GB), but not on GPU 1 (13GB) or 2 (12GB) or 3 (11GB)
# bz = 32
# bz = 48  # OOM at epoch 36
# bz = 32 # ~15:30 for epoch 1 (1 GPUs w/ num_workers=0 )
# bz = 32 # 7:30 for epoch 1 (1 GPUs w/ num_workers=32)
# bz = 128 # 6:20 per epoch (0 workers)
# bz = 128 # 11:14 epoch 0, 4:47 epoch 1 (8 workers)
# TODO: validation is really slow with distributed, maybe we should just do it on one GPU, somehow..?
# OR better, let's actually use all 4 GPUs with not-shitty batch size ;)
# num_workers=0 # 11:42 epoch 0, ~10:14 epoch 1
# num_workers=8 # 7:42 epoch 0, 7:24 epoch 1
# num_workers=8 # I think that's 8 per GPU..?
# TODO: try prefetch_factor=4 for dataloader

# 2022/06/25: 2 GPUs ddp num_workers=0,8 is same w/ cached Librispeech
# about 4:52 per epoch, 30s validation, <5:30 total
# TODO: figure out what Gaddy batch size is by averaging dataloader
# I think bz=20.6 with grad_accum=2 on average assuming 4.5s per example
gpu_ram = torch.cuda.get_device_properties(0).total_memory / 1024**3

if gpu_ram < 24:
    # Titan RTX
    base_bz = 12
    # TODO: need to fix by using size-aware sampling for dataloader
    # base_bz = 4
    # base_bz = 16 # OOM epoch 9 with Titan RTX for batch-level infoNCE
    # val_bz = base_bz
    val_bz = 8
    max_len = 48000 # works for supNCE on Titan RTX
    assert NUM_GPUS == 2
    hardcode_len = 904 # 2 GPUs
elif gpu_ram > 30:
    # V100
    base_bz = 24
    val_bz = base_bz
    max_len = 64000 # try on V100
    assert NUM_GPUS == 4
    hardcode_len = 375 # 4 GPUs
else:
    raise ValueError("Unknown GPU")

# needed for using CachedDataset
# TODO: is this creating problems...?
# data_dir = '/scratch/GaddyPaper/cached/' # temporarily hack for hardcoded paths
emg_datamodule = EMGDataModule(data_dir, togglePhones, normalizers_file, max_len=max_len,
    pin_memory=(not DEBUG), batch_size=val_bz)
emg_train = emg_datamodule.train

mfcc_norm, emg_norm = pickle.load(open(normalizers_file,'rb'))

# TODO: CachedDataset for LibrispeechDataset should construct cache if it doesn't exist
# right now actually need to run 2023-06-21_cache_librispeech.py to create the cache
##
# after loading this + EMG, using 100GB of RAM
speech_train = CachedDataset(LibrispeechDataset, librispeech_train_cache,
    per_index_cache=per_index_cache)
speech_val = CachedDataset(LibrispeechDataset, librispeech_val_cache,
    per_index_cache=per_index_cache)
speech_test =  CachedDataset(LibrispeechDataset, librispeech_test_cache,
    per_index_cache=per_index_cache)

num_emg_train = len(emg_train)
num_speech_train = len(speech_train)

num_emg_train, num_speech_train
emg_speech_train = torch.utils.data.ConcatDataset([
    emg_train, speech_train
])
len(emg_speech_train)

emg_speech_train[num_emg_train-1]
emg_speech_train[num_emg_train]


if ON_SHERLOCK:
    # TODO: should we just use the scratch directory over LOCAL_SCRATCH?
    output_directory = os.path.join(os.environ["SCRATCH"], f"{isotime}_gaddy")
else:
    output_directory = os.path.join(scratch_directory, f"{isotime}_gaddy")

os.makedirs(output_directory, exist_ok=True)
    
logging.basicConfig(handlers=[
        logging.FileHandler(os.path.join(output_directory, 'log.txt'), 'w'),
        logging.StreamHandler()
        ], level=logger_level, format="%(message)s")

logging.debug("DEBUG mode")
if not log_neptune:
    logging.warning("not logging to neptune")


##
auto_lr_find = False

# precision = 32
learning_rate = 3e-4
# 3e-3 leads to NaNs, prob need to have slower warmup in this case
togglePhones = False
text_transform = TextTransform(togglePhones = togglePhones)
##

n_chars = len(text_transform.chars)

if NUM_GPUS > 1:
    # num_workers=0 # nccl backend doesn't support num_workers>0
    # num_workers=8
    num_workers=0
    rank_key = "RANK" if "RANK" in os.environ else "LOCAL_RANK"
    bz = base_bz * NUM_GPUS
    if rank_key not in os.environ:
        rank = 0
    else:
        rank = int(os.environ[rank_key])
    logging.info(f"SETTING CUDA DEVICE ON RANK: {rank}")

    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()
    # we cannot call DistributedSampler before pytorch lightning trainer.fit() is called,
    # or we get this error:
    # RuntimeError: Default process group has not been initialized, please make sure to call init_process_group.
    # always include at least one example of class 1 (EMG & Audio) in batch
    # TrainBatchSampler = partial(DistributedSizeAwareStratifiedBatchSampler,
    #     num_replicas=NUM_GPUS, max_len=max_len//8, always_include_class=1)
    # TrainBatchSampler = partial(DistributedStratifiedBatchSampler,
    #     num_replicas=NUM_GPUS)
    TrainBatchSampler = partial(DistributedSizeAwareStratifiedBatchSampler,
        num_replicas=NUM_GPUS, max_len=max_len//8, always_include_class=1,
        hardcode_len=hardcode_len)
    ValSampler = lambda: DistributedSampler(emg_datamodule.val,
        shuffle=False, num_replicas=NUM_GPUS)
    TestSampler = lambda: DistributedSampler(emg_datamodule.test,
        shuffle=False, num_replicas=NUM_GPUS)
else:
    # TrainBatchSampler = SizeAwareStratifiedBatchSampler
    TrainBatchSampler = partial(DistributedSizeAwareStratifiedBatchSampler,
        num_replicas=NUM_GPUS, max_len=max_len//8, always_include_class=1,
        hardcode_len=hardcode_len)
    # num_workers=32
    num_workers=0 # prob better now that we're caching
    bz = base_bz
    ValSampler = None
    TestSampler = None


datamodule =  EMGAndSpeechModule(emg_datamodule.train,
    emg_datamodule.val, emg_datamodule.test,
    speech_train, speech_val, speech_test,
    bz=bz, val_bz=val_bz, num_replicas=NUM_GPUS, pin_memory=(not DEBUG),
    num_workers=num_workers,
    TrainBatchSampler=TrainBatchSampler,
    ValSampler=ValSampler,
    TestSampler=TestSampler,
)
# steps_per_epoch = len(datamodule.train_dataloader()) # may crash if distributed
steps_per_epoch = len(datamodule.TrainBatchSampler) // grad_accum

# steps_per_epoch = len(datamodule.train_dataloader())
# print(steps_per_epoch)
# for i,b in enumerate(datamodule.train):
#     print(b["silent"])
#     if i>10: break
##

# to Include
# steps_per_epoch, epochs, lm_directory, lr=3e-4,
#                 learning_rate_warmup = 1000, 
# model_size, dropout, num_layers, num_outs, 

@dataclass
class SpeechOrEMGToTextConfig:
    input_channels:int = 8
    steps_per_epoch:int = steps_per_epoch
    # learning_rate:float = 0.00025
    # learning_rate:float = 5e-4
    # learning_rate:float = 2e-5
    learning_rate:float = 3e-4 # also sets initial s4 lr
    # learning_rate:float = 5e-6
    weight_decay:float = 0.1
    adam_epsilon:float = 1e-8
    # warmup_steps:int = 1000
    warmup_steps:int = 1000 // grad_accum # warmup is effectilely by batch count now
    # batch_size:int = 8
    batch_size:int = bz
    # batch_size:int = 24
    # batch_size:int = 32
    # batch_size:int = 2
    num_workers:int = num_workers
    num_train_epochs:int = n_epochs
    gradient_accumulation_steps:int = grad_accum
    sample_rate:int = 16000
    precision:str = precision
    seqlen:int = 600
    attn_layers:int = 6
    # d_model:int = 256
    d_model:int = 768 # original Gaddy

    # https://iclr-blog-track.github.io/2022/03/25/unnormalized-resnets/#balduzzi17shattered
    beta:float = 1 / np.sqrt(2) # adjust resnet initialization
    
    cross_nce_lambda:float = 1.0 # how much to weight the latent loss
    audio_lambda:float = 1.0 # how much to weight the audio->text loss
    sup_nce_lambda:float = 1.0

    # d_inner:int = 1024
    d_inner:int = 3072 # original Gaddy
    prenorm:bool = False
    dropout:float = 0.2
    in_channels:int = 8
    out_channels:int = 80
    resid_dropout:float = 0.0
    num_outs:int = n_chars+1
    max_len:int = max_len
    num_heads:int = 8
    lm_directory:str = lm_directory

Task = Enum('Task', ['EMG', 'AUDIO', 'AUDIO_EMG'])

class SpeechOrEMGToText(Model):
    
    def __init__(self, cfg:SpeechOrEMGToTextConfig, text_transform:TextTransform,
                 profiler = None):
        pl.LightningModule.__init__(self)
        self.profiler = profiler or PassThroughProfiler()
        self.emg_conv_blocks = nn.Sequential(
            ResBlock(cfg.input_channels, cfg.d_model, 2, pre_activation=False,
                beta=cfg.beta),
            ResBlock(cfg.d_model, cfg.d_model, 2, pre_activation=False,
                beta=cfg.beta**2),
            ResBlock(cfg.d_model, cfg.d_model, 2, pre_activation=False,
                beta=cfg.beta**3)
        )
        self.audio_conv_blocks = nn.Sequential(
            ResBlock(80, cfg.d_model, beta=cfg.beta), # 80 mel freq cepstrum coefficients
            ResBlock(cfg.d_model, cfg.d_model, beta=cfg.beta**2),
            ResBlock(cfg.d_model, cfg.d_model, beta=cfg.beta**3)
        )
        # equivalent to w_raw_in in Gaddy's model
        self.emg_latent_linear = nn.Linear(cfg.d_model, cfg.d_model)
        self.emg_latent_norm = nn.BatchNorm1d(cfg.d_model)
        self.audio_latent_norm = nn.BatchNorm1d(cfg.d_model)
        self.audio_latent_linear = nn.Linear(cfg.d_model, cfg.d_model)
        encoder_layer = TransformerEncoderLayer(d_model=cfg.d_model,
            nhead=cfg.num_heads, relative_positional=True,
            relative_positional_distance=100, dim_feedforward=cfg.d_inner,
            dropout=cfg.dropout,
            # beta=1/np.sqrt(2)
            )
        self.transformer = nn.TransformerEncoder(encoder_layer, cfg.attn_layers)
        self.w_out       = nn.Linear(cfg.d_model, cfg.num_outs)
            
        self.seqlen = cfg.seqlen
        self.lr = cfg.learning_rate
        self.target_lr = cfg.learning_rate # will not mutate
        self.learning_rate_warmup = cfg.warmup_steps
        self.epochs = cfg.num_train_epochs
        
        # val/test procedure...
        self.text_transform = text_transform
        self.n_chars = len(text_transform.chars)
        self.lm_directory = cfg.lm_directory
        self.lexicon_file = os.path.join(cfg.lm_directory, 'lexicon_graphemes_noApostrophe.txt')
        self._init_ctc_decoder()
        self.cross_nce_lambda = cfg.cross_nce_lambda
        self.audio_lambda = cfg.audio_lambda
        self.steps_per_epoch = cfg.steps_per_epoch
        
        self.step_target = []
        self.step_pred = []
        self.sup_nce_lambda = cfg.sup_nce_lambda
    
    def emg_encoder(self, x):
        "Encode emg (B x T x C) into a latent space (B x T/8 x C)"
        # print(f"emg_encoder: {x.shape=}")
        x = x.transpose(1,2) # put channel before time for conv
        x = self.emg_conv_blocks(x)
        x = x.transpose(1,2)
        x = self.emg_latent_linear(x)
        logging.info(f"emg_encoder pre-norm: {x.shape=}")
        x = x.transpose(1,2) # channel first for batchnorm
        x = self.emg_latent_norm(x)
        x = x.transpose(1,2) # B x T/8 x C
        return x
        
    def audio_encoder(self, x):
        "Encode emg (B x T x C) into a latent space (B x T/8 x C)"
        x = x.transpose(1,2) # put channel before time for conv
        x = self.audio_conv_blocks(x)
        x = x.transpose(1,2)
        x = self.audio_latent_linear(x)
        logging.info(f"audio_encoder pre-norm: {x.shape=}")
        x = x.transpose(1,2) # channel first for batchnorm
        x = self.audio_latent_norm(x)
        x = x.transpose(1,2) # B x T/8 x C
        return x     
        
    def decoder(self, x):
        """Predict characters from latent space (B x T/8 x C)"""
        x = x.transpose(0,1) # put time first
        # print(f"before transformer: {x.shape=}")
        x = self.transformer(x)
        x = x.transpose(0,1)
        x = self.w_out(x)
        return F.log_softmax(x,2)

    def augment_shift(self, x):
        if self.training:
            xnew = x.clone() # unclear why need this here but gaddy didn't
            r = random.randrange(8)
            if r > 0:
                xnew[:,:-r,:] = x[:,r:,:] # shift left r
                xnew[:,-r:,:] = 0
            return xnew
        else:
            return x
        
    def emg_forward(self, x):
        "Predict characters from emg (B x T x C)"
        x = self.augment_shift(x)
        z = self.emg_encoder(x) # latent space
        return self.decoder(z), z
    
    def audio_forward(self, x):
        "Predict characters from audio mfcc (B x T/8 x 80)"
        z = self.audio_encoder(x) # latent space
        return self.decoder(z), z
    
    def forward(self, emg:List[torch.Tensor], audio:List[torch.Tensor], length_emg, length_audio):
        """Group x by task and predict characters for the batch.
        
        Note that forward will call combine_fixed_length, re-splitting the batch into
        self.seqlen chunks. I believe this is done to avoid having to pad the batch to the max,
        which also may quadratically reduce memory usage due to attention. This is prob okay for
        training, but for inference we want to use the full sequence length."""
        if len(emg) > 0:
            emg = combine_fixed_length(emg, self.seqlen*8)
            # logging.debug(f"FORWARD emg shape: {emg.shape}")
            emg_pred, emg_z = self.emg_forward(emg)
            emg_bz = len(emg) # batch size not known until after combine_fixed_length
            length_emg = [l // 8 for l in length_emg]
            logging.debug(f"before decollate {len(emg_pred)=}, {emg_pred[0].shape=}")
            emg_pred = decollate_tensor(emg_pred, length_emg)
            logging.debug(f"after decollate {len(emg_pred)=}, {emg_pred[0].shape=}")
            # logging.debug(f"before decollate {len(emg_z)=}, {emg_z[0].shape=}")
            # # TODO: perhaps we shouldn't decollate z, since we need to use it cross contrastive loss
            # INFO: but we have to decollate or else we don't know which audio to pair with which emg
            emg_z = decollate_tensor(emg_z, length_emg)
            # logging.debug(f"after decollate {len(emg_z)=}, {emg_z[0].shape=}")
        else:
            emg_pred, emg_z, emg_bz = None, None, 0

        if len(audio) > 0:
            audio = combine_fixed_length(audio, self.seqlen)
            # logging.debug(f"FORWARD audio shape: {audio.shape}")
            audio_pred, audio_z = self.audio_forward(audio)
            audio_bz = len(audio)
            audio_pred = decollate_tensor(audio_pred, length_audio)
            audio_z = decollate_tensor(audio_z, length_audio)
        else:
            audio_pred, audio_z, audio_bz = None, None, 0
        
        # logging.debug("finished FORWARD")
        return (emg_pred, audio_pred), (emg_z, audio_z), (emg_bz, audio_bz)
        
    def ctc_loss(self, pred, target, pred_len, target_len):
        # INFO: Gaddy passes emg length, but shouldn't this actually be divided by 8?
        # TODO: try padding length / 8. must be integers though...
        # print(f"ctc_loss: {pred_len=}, {target_len=}")
        
        # TODO FIXME
        # ctc_loss: [p.shape for p in pred]=[torch.Size([600, 38]), torch.Size([600, 38]), torch.Size([600, 38]), torch.Size([600, 38])], [t.shape for t in target]=[torch.Size([306])]
        # print(f"{pred.shape=}, {target[0].shape=}, {pred_len=}, {target_len=}")
        pred = nn.utils.rnn.pad_sequence(pred, batch_first=False) 
        # pred = nn.utils.rnn.pad_sequence(decollate_tensor(pred, pred_len), batch_first=False) 
        # pred = nn.utils.rnn.pad_sequence(pred, batch_first=False) 
        target    = nn.utils.rnn.pad_sequence(target, batch_first=True)
        # print(f"{pred.shape=}, {target[0].shape=}, {pred_len=}, {target_len=}")
        # print(f"ctc_loss: {[p.shape for p in pred]=}, {[t.shape for t in target]=}")
        loss = F.ctc_loss(pred, target, pred_len, target_len, blank=self.n_chars)
        return loss

    def calc_loss(self, batch):
        emg_tup, audio_tup, idxs = split_batch_into_emg_audio(batch)
        emg, length_emg, emg_phonemes, y_length_emg, y_emg = emg_tup
        audio, length_audio, audio_phonemes, y_length_audio, y_audio = audio_tup
        paired_emg_idx, paired_audio_idx = idxs


        
        (emg_pred, audio_pred), (emg_z, audio_z), (emg_bz, audio_bz) = self(emg, audio, length_emg, length_audio)
        
        if emg_pred is not None:
            length_emg = [l//8 for l in length_emg] # Gaddy doesn't do this but I think it's necessary
            emg_ctc_loss = self.ctc_loss(emg_pred, y_emg, length_emg, y_length_emg)
        else:
            logging.info("emg_pred is None")
            emg_ctc_loss = 0.
        
        if audio_pred is not None:
            audio_ctc_loss = self.ctc_loss(audio_pred, y_audio, length_audio, y_length_audio)
        else:
            logging.info("audio_pred is None")
            audio_ctc_loss = 0.
        
        if emg_z is not None and audio_z is not None:

            # InfoNCE contrastive loss with emg_t, audio_t as positive pairs
            
            paired_e_z = [emg_z[i] for i in paired_emg_idx]
            paired_a_z = [audio_z[i] for i in paired_audio_idx]
            paired_e_phonemes = [emg_phonemes[i] for i in paired_emg_idx]

            # per-utterance only
            # emg_audio_contrastive_loss = var_length_cross_contrastive_loss(
            #     paired_e_z, paired_a_z,
            #     device=self.device)
            
            # across batch
            try:
                paired_e_z = torch.concatenate(paired_e_z)
            except Exception as e:
                logging.error(f"paired_e_z: {paired_e_z=}")
                raise e
            paired_a_z = torch.concatenate(paired_a_z)
            emg_audio_contrastive_loss = nobatch_cross_contrastive_loss(paired_e_z, paired_a_z,
                                                                device=self.device)

            # only use vocalized emg for supervised contrastive loss as we have
            # frame-aligned phoneme labels for those
            z = torch.concatenate([paired_e_z, *audio_z])
            z_class = torch.concatenate([*paired_e_phonemes, *audio_phonemes])
            sup_nce_loss = supervised_contrastive_loss(z, z_class, device=self.device)
            # sup_nce_loss = 0.
        elif emg_z is not None:
            # INFO: phoneme labels aren't frame-aligned with emg, so we can't use them
            # TODO: try DTW with parallel audio/emg to align phonemes with silent emg
            # z = torch.concatenate(emg_z)
            # z_class = torch.concatenate(emg_phonemes)
            emg_audio_contrastive_loss = 0.
            sup_nce_loss = 0.
        elif audio_z is not None:
            raise NotImplementedError("audio only is not expected")
            z = torch.concatenate(audio_z)
            z_class = torch.concatenate(audio_phonemes)
        else:
            emg_audio_contrastive_loss = 0.
            sup_nce_loss = 0.
        
        # logging.debug(f"{z_class=}")
        
        # assert audio_pred is None, f'Audio only not implemented, got {audio_pred=}'
        logging.debug(f"emg_ctc_loss: {emg_ctc_loss}, audio_ctc_loss: {audio_ctc_loss}, " \
                        f"emg_audio_contrastive_loss: {emg_audio_contrastive_loss}, " \
                        f"sup_nce_loss: {sup_nce_loss}")
        loss = emg_ctc_loss + \
            self.audio_lambda * audio_ctc_loss + \
            self.cross_nce_lambda * emg_audio_contrastive_loss + \
            self.sup_nce_lambda * sup_nce_loss
        
        if torch.isnan(loss):
            logging.warning(f"Loss is NaN.")
            # emg_isnan = torch.any(torch.tensor([torch.isnan(e) for e in emg_pred]))
            # audio_isnan = torch.any(torch.tensor([torch.isnan(a) for a in audio_pred]))
            # logging.warning(f"Loss is NaN. EMG isnan output: {emg_isnan}. " \
            #       f"Audio isnan output: {audio_isnan}")
        if torch.isinf(loss):
            logging.warning(f"Loss is Inf.")
            # emg_isinf = torch.any(torch.tensor([torch.isinf(e) for e in emg_pred]))
            # audio_isinf = torch.any(torch.tensor([torch.isinf(a) for a in audio_pred]))
            # logging.warning(f"Loss is Inf. EMG isinf output: {emg_isinf}. " \
            #       f"Audio isinf output: {audio_isinf}")
        
        paired_bz = len(paired_emg_idx)
        # paired_bz <= min(emg_bz, audio_bz)
        bz = np.array([emg_bz, audio_bz, paired_bz])
        if not emg_z is None:
            emg_z_mean = torch.concatenate([e.reshape(-1).abs() for e in emg_z]).mean()
        else:
            emg_z_mean = None
        if not audio_z is None:
            audio_z_mean = torch.concatenate([a.reshape(-1).abs() for a in audio_z]).mean()
        else:
            audio_z_mean = None
        return {
            'loss': loss,
            'emg_ctc_loss': emg_ctc_loss,
            'audio_ctc_loss': audio_ctc_loss,
            'cross_contrastive_loss': emg_audio_contrastive_loss,
            'supervised_contrastive_loss': sup_nce_loss,
            'emg_z_mean': emg_z_mean,
            'audio_z_mean': audio_z_mean,
            'bz': bz
        }
    
    def _beam_search_batch(self, batch):
        "Repeatedly called by validation_step & test_step."
        X = nn.utils.rnn.pad_sequence(batch['raw_emg'], batch_first=True)
        # X = batch['raw_emg'][0].unsqueeze(0) 

        logging.debug(f"calling emg_forward with {X.shape=}")
        pred  = self.emg_forward(X)[0].cpu()

        beam_results = self.ctc_decoder(pred)
        # print(f"{beam_results=}")
        # pred_text    = ' '.join(beam_results[0][0].words).strip().lower()
        # use top hypothesis from beam search
        pred_text = [' '.join(b[0].words).strip().lower() for b in beam_results]
        
        # target_text  = self.text_transform.clean_2(batch['text'][0])
        target_text  = [self.text_transform.clean_2(b) for b in batch['text']]
        
        # print(f"{target_text=}, {pred_text=}")
        return target_text, pred_text
    
    def training_step(self, batch, batch_idx):
        c = self.calc_loss(batch)
        loss = c['loss']
        emg_ctc_loss = c['emg_ctc_loss']
        audio_ctc_loss = c['audio_ctc_loss']
        cross_contrastive_loss = c['cross_contrastive_loss']
        sup_contrastive_loss = c['supervised_contrastive_loss']
        bz = c['bz']
        avg_emg_latent = c['emg_z_mean']
        avg_audio_latent = c['audio_z_mean']
        
        
        self.log("train/loss", loss,
                 on_step=False, on_epoch=True, logger=True, prog_bar=True, batch_size=bz.sum(), sync_dist=True)
        self.log("train/emg_ctc_loss", emg_ctc_loss,
            on_step=False, on_epoch=True, logger=True, prog_bar=False, batch_size=bz[0], sync_dist=True)
        self.log("train/audio_ctc_loss", audio_ctc_loss,
            on_step=False, on_epoch=True, logger=True, prog_bar=False, batch_size=bz[0], sync_dist=True)
        self.log("train/cross_contrastive_loss", cross_contrastive_loss,
                 on_step=False, on_epoch=True, logger=True, prog_bar=False, batch_size=bz[2], sync_dist=True)
        self.log("train/supervised_contrastive_loss", sup_contrastive_loss,
                 on_step=False, on_epoch=True, logger=True, prog_bar=False, batch_size=bz[2], sync_dist=True)
        self.log("train/avg_emg_latent", avg_emg_latent,
                 on_step=False, on_epoch=True, logger=True, prog_bar=False, batch_size=bz[0], sync_dist=True)
        self.log("train/avg_audio_latent", avg_audio_latent,
                 on_step=False, on_epoch=True, logger=True, prog_bar=False, batch_size=bz[1], sync_dist=True)
        torch.cuda.empty_cache()
        return loss

    def validation_step(self, batch, batch_idx, task="val"):
        c = self.calc_loss(batch)
        loss = c['loss']
        emg_ctc_loss = c['emg_ctc_loss']
        audio_ctc_loss = c['audio_ctc_loss']
        bz = c['bz']

        logging.debug(f"validation_step: {batch_idx=}, {loss=}, {emg_ctc_loss=}, {audio_ctc_loss=}, {bz=}")

        target_texts, pred_texts = self._beam_search_batch(batch) # TODO: also validate on audio
        # print(f"text: {batch['text']}; target_text: {target_text}; pred_text: {pred_text}")
        for i, (target_text, pred_text) in enumerate(zip(target_texts, pred_texts)):
            if len(target_text) > 0:
                self.step_target.append(target_text)
                self.step_pred.append(pred_text)
                if i % 16 == 0 and type(self.logger) == NeptuneLogger:
                    # log approx 10 examples
                    self.logger.experiment[f"training/{task}/sentence_target"].append(target_text)
                    self.logger.experiment[f"training/{task}/sentence_pred"].append(pred_text)
            
        self.log(f"{task}/loss", loss, prog_bar=True, batch_size=bz.sum(), sync_dist=True)
        self.log(f"{task}/emg_ctc_loss", emg_ctc_loss, prog_bar=False, batch_size=bz[0], sync_dist=True)
        self.log(f"{task}/audio_ctc_loss", audio_ctc_loss, prog_bar=False, batch_size=bz[0], sync_dist=True)
        

        return loss
    
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx, task="test")

class BoringModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.rand(1))

    def forward(self, x):
        return x * self.param
    
    def calc_loss(self,batch):
        return self(batch['audio_features'][0]).sum()

    def training_step(self, batch, batch_idx):
        loss = self.calc_loss(batch)
        self.log("train_loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        logging.warning(f"validation_step: {batch_idx=}")
        loss = self.calc_loss(batch)
        self.log("valid_loss", loss)

    def test_step(self, batch, batch_idx):
        loss = self.calc_loss(batch)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)

config = SpeechOrEMGToTextConfig()

model = SpeechOrEMGToText(config, text_transform)
# model = BoringModel()

# why is this sooo slow?? slash freezes..? are we hitting oak?
# TODO: benchmark with cProfiler. CPU & GPU are near 100% during however
# not always slamming CPU/GPU...
logging.info('made model')

callbacks = [
    # starting at epoch 0, accumulate 2 batches of grads
    GradientAccumulationScheduler(scheduling={0: config.gradient_accumulation_steps})
]

if log_neptune:
    neptune_logger = NeptuneLogger(
        # need to store credentials in your shell env
        api_key=os.environ["NEPTUNE_API_TOKEN"],
        project="neuro/Gaddy",
        # name=magneto.fullname(model), # from lib
        name=model.__class__.__name__,
        tags=[model.__class__.__name__,
                "EMGonly",
                "preactivation",
                "AdamW",
                f"fp{config.precision}",
                ],
        log_model_checkpoints=False,
    )
    neptune_logger.log_hyperparams(vars(config))

    checkpoint_callback = ModelCheckpoint(
        monitor="val/wer",
        mode="min",
        dirpath=output_directory,
        filename=model.__class__.__name__+"-{epoch:02d}-{val/wer:.3f}",
    )
    callbacks.extend([
        checkpoint_callback,
        pl.callbacks.LearningRateMonitor(logging_interval="epoch"),
        # pl.callbacks.LearningRateMonitor(logging_interval="step"), # good for troubleshooting warmup
    ])
else:
    neptune_logger = None

# QUESTION: why does validation loop become massively slower as training goes on?
# perhaps this line will resolve..?
# export NEPTUNE_ALLOW_SELF_SIGNED_CERTIFICATE='TRUE'
##
if NUM_GPUS > 1:
    devices = 'auto'
    strategy=DDPStrategy(gradient_as_bucket_view=True, find_unused_parameters=True)
elif NUM_GPUS == 1:
    devices = [0]
    strategy = "auto"
else:
    devices = 'auto'
    strategy = "auto"
trainer = pl.Trainer(
    max_epochs=config.num_train_epochs,
    devices=devices,
    accelerator="gpu",
    # accelerator="cpu",
    gradient_clip_val=0.5,
    logger=neptune_logger,
    default_root_dir=output_directory,
    callbacks=callbacks,
    precision=config.precision,
    limit_train_batches=limit_train_batches,
    limit_val_batches=limit_val_batches,
    # strategy=strategy,
    use_distributed_sampler=False, # we need to make a custom distributed sampler
    num_sanity_val_steps=num_sanity_val_steps,
    sync_batchnorm=True, # TODO: we should pass Audio & EMG together, not by task
    strategy=strategy,
    # strategy='fsdp', # errors on CTC loss being used on half-precision.
    # also model only ~250MB of params, so fsdp may be overkill
    # check_val_every_n_epoch=10 # should give speedup of ~30% since validation is bz=1
)

if auto_lr_find:
    # TODO: might be deprecated
    # https://lightning.ai/docs/pytorch/stable/upgrade/from_1_9.html
    # https://lightning.ai/docs/pytorch/stable/advanced/training_tricks.html#learning-rate-finder
    tuner = pl.tuner.Tuner(trainer)
    tuner.lr_find(model, datamodule)
        
logging.info('about to fit')
# epoch of 242 if only train...
# trainer.fit(model, datamodule.train_dataloader(),
#             datamodule.val_dataloader())
# trainer.fit(model, train_dataloaders=datamodule.train_dataloader()) 
# note: datamodule.train_dataloader() can sometimes be slow depending on Oak filesystem
# we should prob transfer this data to $LOCAL_SCRATCH first...
trainer.fit(model, datamodule=datamodule) 
# trainer.fit(model, train_dataloaders=datamodule.train_dataloader(),
#             val_dataloaders=datamodule.val_dataloader()) 

if log_neptune:
    ckpt_path = os.path.join(output_directory,f"finished-training_epoch={config.num_train_epochs}.ckpt")
    trainer.save_checkpoint(ckpt_path)
    print(f"saved checkpoint to {ckpt_path}")
##
# TODO: run again now that we fixed num_replicas in DistributedStratifiedBatchSampler

# ##
# for i,b in enumerate(datamodule.train_dataloader()):
#     N = len(b['audio_features'])
#     for j in range(N):
#         if b['silent'][j]:
#             aus = b['audio_features'][j].shape[0]
#             es = b['raw_emg'][j].shape[0] // 8
#             ps = b['phonemes'][j].shape[0]
#         # assert aus == es == ps, f"batch {i} sample {j} size mismatch: audio={aus}, emg={es}, phonemes={ps}"
#         if not aus == es == ps:
#             print(f"batch {i} sample {j} size mismatch: audio={aus}, emg={es}, phonemes={ps}")
# ##
# for i,b in enumerate(datamodule.train):
#     if b['silent']:
#         aus = b['audio_features'].shape[0]
#         es = b['raw_emg'].shape[0] // 8
#         ps = b['phonemes'].shape[0]
#         if not aus == es == ps:
#             print(f"sample {i} size mismatch: audio={aus}, emg={es}, phonemes={ps}")
# ##
# td = EMGDataset(togglePhones=togglePhones, normalizers_file=normalizers_file)
# ##
# # TODO: phoneme length is wrong when silent... 
# # a) we can use pretrained model to align phonemes
# # b) we can use DTW to align phonemes during training
# # TODO: align phonemes for silent speech using pretrained model & DTW
# i = 2
# td[i]['silent'], td[i]['phonemes'].shape[0], td[i]['audio_features'].shape[0], td[i]['raw_emg'].shape[0] //8

# ##
# max_phonemes = 0
# phone_count = {}
# for i,b in enumerate(datamodule.train):
#     max_phonemes = max(max_phonemes, b['phonemes'].max())
#     for p in b['phonemes']:
#         p = int(p)
#         if p not in phone_count:
#             phone_count[p] = 1
#         phone_count[p] += 1
# max_phonemes, phone_count
# ##
# for i in range(max_phonemes+1):
#     if i not in phone_count:
#         print(i)
# ##
