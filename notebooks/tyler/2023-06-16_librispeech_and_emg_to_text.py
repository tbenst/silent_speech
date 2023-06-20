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
import subprocess
import jiwer
import random
from tqdm.auto import tqdm
from typing import List
from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F
from torchaudio.models.decoder import ctc_decoder

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
from data_utils import TextTransform
from typing import List
from collections import defaultdict
from enum import Enum
from magneto.preprocessing import ensure_data_on_scratch
from dataloaders import LibrispeechDataset, EMGAndSpeechModule
from datasets import load_dataset

isotime = datetime.now().isoformat()
hostname = subprocess.run("hostname", capture_output=True)
ON_SHERLOCK = hostname.stdout[:2] == b"sh"

assert os.environ["NEPTUNE_ALLOW_SELF_SIGNED_CERTIFICATE"] == 'TRUE', "run this in shell: export NEPTUNE_ALLOW_SELF_SIGNED_CERTIFICATE='TRUE'"

# load our data file paths and metadata:
if ON_SHERLOCK:
    sessions_dir = '/oak/stanford/projects/babelfish/magneto/'
    scratch_directory = os.environ["LOCAL_SCRATCH"]
    gaddy_dir = '/oak/stanford/projects/babelfish/magneto/GaddyPaper/'
else:
    sessions_dir = '/data/magneto/'
    scratch_directory = "/scratch"
    gaddy_dir = '/scratch/GaddyPaper/'

# This approach is massively slow, at least 1+ hours.
# we should figure out how to better cache...
# (may need to expand the $LOCAL_SCRATCH variable first)
# ln -s $LOCAL_SCRATCH/huggingface ~/.cache/huggingface
# rsync -avP /oak/stanford/projects/babelfish/magneto/huggingface $LOCAL_SCRATCH/
librispeech_datasets = load_dataset("librispeech_asr")
librispeech_clean_train = torch.utils.data.ConcatDataset([librispeech_datasets['train.clean.100'],
                                                    librispeech_datasets['train.clean.360']])
                                                    # librispeech_datasets['train.other.500']])
librispeech_clean_val = librispeech_datasets['validation.clean']
librispeech_clean_test = librispeech_datasets['test.clean']

max_len = 128000 * 2
data_dir = os.path.join(gaddy_dir, 'processed_data/')
emg_dir = os.path.join(gaddy_dir, 'emg_data/')
lm_directory = os.path.join(gaddy_dir, 'pretrained_models/librispeech_lm/')
normalizers_file = os.path.join(SCRIPT_DIR, "normalizers.pkl")
togglePhones = False


# copy_metadata_command = f"rsync -am --include='*.json' --include='*/' --exclude='*' {emg_dir} {scratch_directory}/"
scratch_emg = os.path.join(scratch_directory,"emg_data")
if ON_SHERLOCK:
    if not os.path.exists(scratch_emg):
        os.symlink(emg_dir, scratch_emg)
    data_dir = ensure_folder_on_scratch(data_dir, scratch_directory)
    lm_directory = ensure_folder_on_scratch(lm_directory, scratch_directory)

emg_datamodule = EMGDataModule(data_dir, togglePhones, normalizers_file, max_len=max_len)
emg_train = emg_datamodule.train

mfcc_norm, emg_norm = pickle.load(open(normalizers_file,'rb'))

speech_train = LibrispeechDataset(librispeech_clean_train, emg_train.text_transform, mfcc_norm)
speech_val = LibrispeechDataset(librispeech_clean_val, emg_train.text_transform, mfcc_norm)
speech_test = LibrispeechDataset(librispeech_clean_test, emg_train.text_transform, mfcc_norm)
num_emg_train = len(emg_train)
num_speech_train = len(speech_train)

num_emg_train, num_speech_train
emg_speech_train = torch.utils.data.ConcatDataset([
    emg_train, speech_train
])
len(emg_speech_train)

emg_speech_train[num_emg_train-1]
emg_speech_train[num_emg_train]


output_directory = os.path.join(scratch_directory, f"{isotime}_gaddy")

##
auto_lr_find = False
max_len = 128000 * 2
log_neptune = True
# log_neptune = False
# log_neptune = False
# precision = 32
learning_rate = 3e-4
# 3e-3 leads to NaNs, prob need to have slower warmup in this case
togglePhones = False

##

os.makedirs(output_directory, exist_ok=True)
logging.basicConfig(handlers=[
        logging.FileHandler(os.path.join(output_directory, 'log.txt'), 'w'),
        logging.StreamHandler()
        ], level=logging.INFO, format="%(message)s")

##
n_chars = len(emg_datamodule.val.text_transform.chars)
bz = 24
num_workers=8
datamodule =  EMGAndSpeechModule(emg_datamodule, speech_train, speech_val, speech_test,
    bz=bz,
    # num_workers=0, # 11:42 epoch 0, ~10:14 epoch 1
    num_workers=num_workers,
)
steps_per_epoch = len(datamodule.train_dataloader())
print(steps_per_epoch)
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
    warmup_steps:int = 500
    # batch_size:int = 8
    batch_size:int = bz
    # batch_size:int = 24
    # batch_size:int = 32
    # batch_size:int = 2
    num_workers:int = num_workers
    num_train_epochs:int = 200
    gradient_accumulation_steps:int = 1
    sample_rate:int = 16000
    precision:str = "16-mixed"
    seqlen:int = 600
    # precision:str = "32"
    attn_layers:int = 6
    # d_model:int = 256
    d_model:int = 768 # original Gaddy

    # https://iclr-blog-track.github.io/2022/03/25/unnormalized-resnets/#balduzzi17shattered
    beta:float = 1 / np.sqrt(2) # adjust resnet initialization
    
    latent_lambda:float = 0.1 # how much to weight the latent loss
    audio_lambda:float = 0.1 # how much to weight the audio->text loss

    # d_inner:int = 1024
    d_inner:int = 3072 # original Gaddy
    prenorm:bool = False
    dropout:float = 0.2
    in_channels:int = 8
    out_channels:int = 80
    resid_dropout:float = 0.0
    num_outs:int = n_chars+1
    max_len:int = max_len # maybe make smaller..?
    num_heads:int = 8
    lm_directory:str = '/oak/stanford/projects/babelfish/magneto/GaddyPaper/pretrained_models/librispeech_lm/'

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
        self.steps_per_epoch = cfg.steps_per_epoch
        
        # val/test procedure...
        self.text_transform = text_transform
        self.n_chars = len(text_transform.chars)
        self.lm_directory = cfg.lm_directory
        self.lexicon_file = os.path.join(cfg.lm_directory, 'lexicon_graphemes_noApostrophe.txt')
        self._init_ctc_decoder()
        self.latent_lambda = cfg.latent_lambda
        self.audio_lambda = cfg.audio_lambda
        
        self.step_target = []
        self.step_pred = []
    
    def emg_encoder(self, x):
        "Encode emg (B x T x C) into a latent space (B x T/8 x C)"
        # print(f"emg_encoder: {x.shape=}")
        x = x.transpose(1,2) # put channel before time for conv
        x = self.emg_conv_blocks(x)
        x = x.transpose(1,2)
        x = self.emg_latent_linear(x)
        return x
        
    def audio_encoder(self, x):
        "Encode emg (B x T x C) into a latent space (B x T/8 x C)"
        x = x.transpose(1,2) # put channel before time for conv
        x = self.audio_conv_blocks(x)
        x = x.transpose(1,2)
        x = self.audio_latent_linear(x)
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
        x = self.emg_encoder(x)
        return self.decoder(x)
    
    def audio_forward(self, x):
        "Predict characters from audio mfcc (B x T/8 x 80)"
        x = self.audio_encoder(x)
        return self.decoder(x)
    
    def audio_emg_forward(self, audio, emg):
        """Predict characters from audio mfcc (B x T/8 x 80) and emg (B x T x C)
        
        We addditionally return the latent space for each modality.
        """
        emg = self.augment_shift(emg)
        emg_latent = self.emg_encoder(emg)
        # audio_latent = None
        audio_latent = self.audio_encoder(audio)
        emg_pred = self.decoder(emg_latent)
        # audio_pred = None
        audio_pred = self.decoder(audio_latent)
        return emg_pred, audio_pred, emg_latent, audio_latent
    
    def forward(self, tasks:List[Task], emg:List[torch.Tensor], audio:List[torch.Tensor]):
        "Group x by task and predict characters for the batch."
        # group by task
        task_emg = []
        task_audio = []
        task_both_audio = []
        task_both_emg = []
        for task, e, a in zip(tasks, emg, audio):
            if task == Task.EMG:
                task_emg.append(e)
            elif task == Task.AUDIO:
                task_audio.append(a)
            elif task == Task.AUDIO_EMG:
                task_both_audio.append(a)
                task_both_emg.append(e)
            else:
                raise ValueError(f'Unknown task {task}')
                
        emg_pred, audio_pred, both_pred = None, None, None
        # combine to single tensor for each task
        emg_bz = 0
        audio_bz = 0
        both_bz = 0
        if len(task_emg) > 0:
            # print(f"{task_emg[0].shape=}")
            task_emg = combine_fixed_length(task_emg, self.seqlen*8)
            # print(f"{task_emg.shape=}")
            emg_pred = self.emg_forward(task_emg)
            emg_bz += len(task_emg) # batch size not known until after combine_fixed_length
        if len(task_audio) > 0:
            task_audio = combine_fixed_length(task_audio, self.seqlen)
            audio_pred = self.audio_forward(task_audio)
            audio_bz += len(task_audio)
        if len(task_both_audio) > 0:
            task_both_audio = combine_fixed_length(task_both_audio, self.seqlen)
            task_both_emg = combine_fixed_length(task_both_emg, self.seqlen*8)
            both_emg_pred, both_audio_pred, both_emg_latent, both_audio_latent = self.audio_emg_forward(task_both_audio, task_both_emg)
            both_pred = (both_emg_pred, both_audio_pred, both_emg_latent, both_audio_latent)
            both_bz += len(task_both_audio)
            
        return emg_pred, audio_pred, both_pred, (emg_bz, audio_bz, both_bz)
        
    def ctc_loss(self, pred, target, pred_len, target_len):
        # INFO: Gaddy passes emg length, but shouldn't this actually be divided by 8?
        # TODO: try padding length / 8. must be integers though...
        # print(f"ctc_loss: {pred_len=}, {target_len=}")
        
        # TODO FIXME
        # ctc_loss: [p.shape for p in pred]=[torch.Size([600, 38]), torch.Size([600, 38]), torch.Size([600, 38]), torch.Size([600, 38])], [t.shape for t in target]=[torch.Size([306])]
        # print(f"{pred.shape=}, {target[0].shape=}, {pred_len=}, {target_len=}")
        pred = nn.utils.rnn.pad_sequence(decollate_tensor(pred, pred_len), batch_first=False) 
        # pred = nn.utils.rnn.pad_sequence(pred, batch_first=False) 
        target    = nn.utils.rnn.pad_sequence(target, batch_first=True)
        # print(f"{pred.shape=}, {target[0].shape=}, {pred_len=}, {target_len=}")
        # print(f"ctc_loss: {[p.shape for p in pred]=}, {[t.shape for t in target]=}")
        loss = F.ctc_loss(pred, target, pred_len, target_len, blank=self.n_chars)
        return loss


    def calc_loss(self, batch):
        tasks = []
        emg = []
        audio = []
        length_emg = []
        y_length_emg = []
        length_audio = []
        y_length_audio = []
        length_both = []
        y_length_both = []
        y_emg = []
        y_audio = []
        y_both = []
        for i, (s,a) in enumerate(zip(batch['silent'], batch['audio_only'])):
            if s:
                tasks.append(Task.EMG)
                emg.append(batch['raw_emg'][i])
                audio.append(None)
                length_emg.append(batch['raw_emg_lengths'][i])
                y_length_emg.append(batch['text_int_lengths'][i])
                y_emg.append(batch['text_int'][i])
            elif a:
                tasks.append(Task.AUDIO)
                emg.append(None)
                audio.append(batch['audio_features'][i])
                length_audio.append(batch['audio_feature_lengths'][i])
                y_length_audio.append(batch['text_int_lengths'][i])
                y_audio.append(batch['text_int'][i])
            else:
                tasks.append(Task.AUDIO_EMG)
                emg.append(batch['raw_emg'][i])
                audio.append(batch['audio_features'][i])
                length_both.append(batch['raw_emg_lengths'][i])
                y_length_both.append(batch['text_int_lengths'][i])
                y_both.append(batch['text_int'][i])
    
        emg_pred, audio_pred, both_pred, (emg_bz, audio_bz, both_bz) = self(tasks, emg, audio)
        # print(f"{emg_pred.shape=}")
        
        # TODO: finish this!! need to implement loss funcion for each task
        if emg_pred is not None:
            length_emg = [l//8 for l in length_emg] # Gaddy doesn't do this but I think it's necessary
            emg_ctc_loss = self.ctc_loss(emg_pred, y_emg, length_emg, y_length_emg)
        else:
            logging.warn("emg_pred is None")
            emg_ctc_loss = 0
        
        if audio_pred is not None:
            audio_ctc_loss = self.ctc_loss(audio_pred, y_audio, length_audio, y_length_audio)
        else:
            logging.warn("audio_pred is None")
            audio_ctc_loss = 0
            
        if both_pred is not None:
            both_emg_pred, both_audio_pred, both_emg_latent, both_audio_latent = both_pred
            # audio mfccs should be length / 8 ...?
            length_both = [l//8 for l in length_both]
            both_ctc_loss = self.ctc_loss(both_emg_pred, y_both, length_both, y_length_both) + \
                self.ctc_loss(both_audio_pred, y_both, length_both, y_length_both) * self.audio_lambda
            both_latent_match_loss = F.mse_loss(both_emg_latent, both_audio_latent) * self.latent_lambda
            
            # no audio loss for now to compare with Gaddy
            # both_ctc_loss = self.ctc_loss(both_emg_pred, y_both, length_both, y_length_both)
            # both_latent_match_loss = 0
            
        else:
            logging.warn("both_pred is None")
            both_ctc_loss = 0
            both_latent_match_loss = 0
        # assert audio_pred is None, f'Audio only not implemented, got {audio_pred=}'

        loss = emg_ctc_loss + audio_ctc_loss + both_ctc_loss + both_latent_match_loss
        
        if torch.isnan(loss):
            print(f"Loss is NaN. Isnan output: {torch.any(torch.isnan(emg_pred))}")
        if torch.isinf(loss):
            print(f"Loss is Inf. Isinf output: {torch.any(torch.isinf(emg_pred))}")
            
        bz = np.array([emg_bz, audio_bz, both_bz])
        return loss, (emg_ctc_loss, audio_ctc_loss, both_ctc_loss, both_latent_match_loss), bz
    
    def _beam_search_step(self, batch):
        "Repeatedly called by validation_step & test_step. Impure function!"
        X = batch['raw_emg'][0].unsqueeze(0)
        pred  = self.emg_forward(X).cpu()

        beam_results = self.ctc_decoder(pred)
        pred_int     = beam_results[0][0].tokens
        pred_text    = ' '.join(beam_results[0][0].words).strip().lower()
        target_text  = self.text_transform.clean_2(batch['text'][0][0])

        return target_text, pred_text
    
    def training_step(self, batch, batch_idx):
        loss, (emg_ctc_loss, audio_ctc_loss, both_ctc_loss, both_latent_match_loss), bz = self.calc_loss(batch)
        self.log("train/loss", loss,
                 on_step=False, on_epoch=True, logger=True, prog_bar=True, batch_size=bz.sum())
        self.log("train/emg_ctc_loss", emg_ctc_loss,
            on_step=False, on_epoch=True, logger=True, prog_bar=False, batch_size=bz[0])
        self.log("train/audio_ctc_loss", audio_ctc_loss,
            on_step=False, on_epoch=True, logger=True, prog_bar=False, batch_size=bz[0])
        self.log("train/both_ctc_loss", both_ctc_loss,
            on_step=False, on_epoch=True, logger=True, prog_bar=False, batch_size=bz[2])
        self.log("train/both_latent_match_loss", both_latent_match_loss,
                 on_step=False, on_epoch=True, logger=True, prog_bar=False, batch_size=bz[2])
        return loss

    def validation_step(self, batch, batch_idx):
        loss, (emg_ctc_loss, audio_ctc_loss, both_ctc_loss, both_latent_match_loss), bz = self.calc_loss(batch)
        target_text, pred_text = self._beam_search_step(batch) # TODO: also validate on audio
        assert len(batch['raw_emg']) == 1, "Currently only support batch size of 1 for validation"
        if len(target_text) > 0:
            self.step_target.append(target_text)
            self.step_pred.append(pred_text)
            
        self.log("val/loss", loss, prog_bar=True, batch_size=bz.sum())
        self.log("val/emg_ctc_loss", emg_ctc_loss, prog_bar=False, batch_size=bz[0])
        self.log("val/audio_ctc_loss", audio_ctc_loss, prog_bar=False, batch_size=bz[0])
        self.log("val/both_ctc_loss", both_ctc_loss, prog_bar=False, batch_size=bz[2])
        self.log("val/both_latent_match_loss", both_latent_match_loss, prog_bar=False, batch_size=bz[2])
        return loss
    
    def test_step(self, batch, batch_idx):
        loss, (emg_ctc_loss, audio_ctc_loss, both_ctc_loss, both_latent_match_loss), bz = self.calc_loss(batch)
        target_text, pred_text = self._beam_search_step(batch) # TODO: also validate on audio
        if len(target_text) > 0:
            self.step_target.append(target_text)
            self.step_pred.append(pred_text)
        self.log("test/loss", loss, prog_bar=True, batch_size=bz.sum())
        self.log("test/emg_ctc_loss", emg_ctc_loss, prog_bar=False, batch_size=bz[0])
        self.log("testaudiog_ctc_loss", audio_ctc_loss, prog_bar=False, batch_size=bz[0])
        self.log("test/both_ctc_loss", both_ctc_loss, prog_bar=False, batch_size=bz[2])
        self.log("test/both_latent_match_loss", both_latent_match_loss, prog_bar=False, batch_size=bz[2])
        return loss
    

config = SpeechOrEMGToTextConfig()

model = SpeechOrEMGToText(config, emg_datamodule.val.text_transform)

# why is this sooo slow?? slash freezes..? are we hitting oak?
# TODO: benchmark with cProfiler. CPU & GPU are near 100% during however
# not always slamming CPU/GPU...
logging.info('made model')

callbacks = [
    # starting at epoch 0, accumulate 2 batches of grads
    GradientAccumulationScheduler(scheduling={0: 2})
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

# TODO: at epoch 22 validation seems to massively slow down...?
# may be due to neptune...? (saw freeze on two models at same time...)
trainer = pl.Trainer(
    max_epochs=config.num_train_epochs,
    devices=[0],
    # devices=[1],
    accelerator="gpu",
    # QUESTION: Gaddy accumulates grads from two batches, then does clip_grad_norm_
    # are we clipping first then addiing? (prob doesn't matter...)
    gradient_clip_val=0.5,
    logger=neptune_logger,
    default_root_dir=output_directory,
    callbacks=callbacks,
    precision=config.precision,
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
trainer.fit(model, train_dataloaders=datamodule.train_dataloader(),
            val_dataloaders=datamodule.val_dataloader()) 
# trainer.save_checkpoint(os.path.join(output_directory,f"finished-training_epoch={config.epochs}.ckpt"))
##
