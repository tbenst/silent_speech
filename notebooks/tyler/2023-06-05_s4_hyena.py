##
2
##
%load_ext autoreload
%autoreload 2
##
import os
nep_key = "NEPTUNE_ALLOW_SELF_SIGNED_CERTIFICATE"
if not nep_key in os.environ or os.environ["NEPTUNE_ALLOW_SELF_SIGNED_CERTIFICATE"] != 'TRUE':
    os.environ["NEPTUNE_ALLOW_SELF_SIGNED_CERTIFICATE"] = 'TRUE'
    
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import whisper
import pytorch_lightning as pl
import sys
from functools import lru_cache, partial
from magneto.models.s4d import S4D
from magneto.models.s4 import S4
# from magneto.models.hyena import HyenaOperator
from safari.models.sequence.hyena import HyenaOperator
from safari.models.sequence.long_conv_lm import create_block
from magneto.models.waveword import S4Params
import numpy as np
import logging
import subprocess, re
import jiwer, scipy.signal, evaluate
import random
from transformers import get_linear_schedule_with_warmup
from tqdm.auto import tqdm
import torch
from torch import nn
import torch.nn.functional as F
from torchaudio.models.decoder import ctc_decoder
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import LearningRateFinder
from torch.distributed.fsdp.wrap import wrap
from torchvision.ops import StochasticDepth
from flash_attn.modules.mha import MHA

from dataclasses import dataclass

# horrible hack to get around this repo not being a proper python package
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(SCRIPT_DIR)

from read_emg import EMGDataset, SizeAwareSampler, PreprocessedEMGDataset, PreprocessedSizeAwareSampler, EMGDataModule, ensure_folder_on_scratch
from architecture import Model
from data_utils import combine_fixed_length, decollate_tensor
from transformer import TransformerEncoderLayer
from pytorch_lightning.loggers import NeptuneLogger
import neptune, shutil
from datetime import datetime
from pytorch_lightning.callbacks import ModelCheckpoint, GradientAccumulationScheduler
from pytorch_lightning.profilers import SimpleProfiler, AdvancedProfiler, PyTorchProfiler
from data_utils import TextTransform
import random
from flash_attn.flash_attention import FlashMHA

random.seed(9148)
torch.manual_seed(5611)
np.random.seed(1122)
##
isotime = datetime.now().isoformat()
hostname = subprocess.run("hostname", capture_output=True)
ON_SHERLOCK = hostname.stdout[:2] == b"sh"

# load our data file paths and metadata:
if ON_SHERLOCK:
    sessions_dir = '/oak/stanford/projects/babelfish/magneto/'
    scratch_directory = os.environ["LOCAL_SCRATCH"]
    data_dir = '/oak/stanford/projects/babelfish/magneto/GaddyPaper/processed_data/'
else:
    sessions_dir = '/data/magneto/'
    scratch_directory = "/scratch"
    data_dir = '/scratch/GaddyPaper/processed_data'
output_directory = os.path.join(scratch_directory, f"{isotime}_gaddy")
togglePhones = False
max_len = 128000 * 2 # max length for SizeAwareSampler
normalizers_file = os.path.join(SCRIPT_DIR, "normalizers.pkl")
# defaults to transcribe english

datamodule = EMGDataModule(data_dir, togglePhones, normalizers_file, max_len=max_len)
n_chars = len(datamodule.val.text_transform.chars)
num_outs = n_chars+1
steps_per_epoch = len(datamodule.train_dataloader())

if ON_SHERLOCK:
    lm_directory = '/oak/stanford/projects/babelfish/magneto/GaddyPaper/pretrained_models/librispeech_lm/'
else:
    raise NotImplementedError

# lm_directory = ensure_folder_on_scratch(lm_directory, scratch_directory)
# data_dir = ensure_folder_on_scratch(data_dir, data_dir)


##

@dataclass
class S4HyenaConfig:
    input_channels:int = 8
    steps_per_epoch:int = steps_per_epoch
    # learning_rate:float = 0.00025
    # learning_rate:float = 5e-4
    learning_rate:float = 2e-5
    # learning_rate:float = 5e-3
    # learning_rate:float = 5e-6
    weight_decay:float = 0.1
    adam_epsilon:float = 1e-8
    warmup_steps:int = 500
    # batch_size:int = 8
    batch_size:int = 16
    # batch_size:int = 24
    # batch_size:int = 32
    # batch_size:int = 2
    num_worker:int = 0
    num_train_epochs:int = 200
    gradient_accumulation_steps:int = 1
    sample_rate:int = 16000
    # precision:str = "16-mixed"
    precision:str = 32
    hyena_layers:int = 2
    s4_layers:int = 4
    hyena_dim:int = 64
    hyena_seq_len:int = 2**18
    hyena_order:int = 2
    hyena_filter_order:int = 64
    d_model:int = 512
    d_inner:int = 2048
    prenorm:bool = False
    dropout:float = 0.0
    in_channels:int = 8
    out_channels:int = 80
    resid_dropout:float = 0.0
    num_outs:int = num_outs
    max_len:int = max_len # maybe make smaller..?
    num_heads:int = 8
    
class Block(nn.Module):
    def __init__(self, layer, seqlen=600*8, d_model=512, h_model=1024, num_heads=8, dropout=0.1,
                 drop_path=0.0):
        super().__init__()
        # The standard block is: LN -> MHA -> Dropout -> Add -> LN -> MLP -> Dropout -> Add.
        # [Ref: https://arxiv.org/abs/2002.04745]
        self.ln1 = nn.LayerNorm([seqlen, d_model])
        self.layer = layer
        self.dropout1 = nn.Dropout(dropout)
        self.drop_path1 = StochasticDepth(drop_path, mode='row')
        self.drop_path2 = StochasticDepth(drop_path, mode='row')
        self.dropout2 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(d_model, h_model)
        self.activation = F.gelu
        self.fc2 = nn.Linear(h_model, d_model)
        self.ln2 = nn.LayerNorm([seqlen, d_model]) # TODO new dims
        
        
    def forward(self, x):
        z = self.ln1(x)
        z = self.layer(z)
        z = self.drop_path1(self.dropout1(z))
        # residual connection
        x = x + z
        x = self.ln2(x)
        z = x
        # MLP
        z = self.fc1(z)
        z = self.activation(z)
        z = self.fc2(z)
        z = self.drop_path2(self.dropout2(z))
        # residual connection
        x = x + z
        return x


class S4HyenaModule(Model):
    _init_ctc_decoder = Model._init_ctc_decoder
    calc_loss = Model.calc_loss
    # __beam_search_step = Model.__beam_search_step
    training_step = Model.training_step
    on_validation_epoch_start = Model.on_validation_epoch_start
    validation_step = Model.validation_step
    on_validation_epoch_end = Model.on_validation_epoch_end
    test_step = Model.test_step
    on_test_epoch_end = Model.on_test_epoch_end
    
    def __init__(self, cfg:S4HyenaConfig, text_transform: TextTransform, lm_directory,
                 train_dataset=[], eval_dataset=[]) -> None:
        pl.LightningModule.__init__(self)
        # super().__init__()
        self.steps_per_epoch = cfg.steps_per_epoch
        self.lr = cfg.learning_rate

        self.hparams.update(vars(cfg))
        
        self.seqlen = 600
        # accumulate text over epoch for validation so we can caclulate WER
        self.step_target = []
        self.step_pred = []
        
        self.text_transform = text_transform
        self.n_chars = len(text_transform.chars)
        self.lm_directory = lm_directory
        self.lexicon_file = os.path.join(lm_directory, 'lexicon_graphemes_noApostrophe.txt')
        self._init_ctc_decoder()

        
        ################ S4Model emg -> (audio) mel spectrogram ###############
        self.encoder = nn.Linear(cfg.in_channels, cfg.hyena_dim)
        
        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.hyena_layers = nn.ModuleList()
        self.norms     = nn.ModuleList()
        self.dropouts  = nn.ModuleList()
        attn_layer_idx = [1,8]
        attn_cfg = {"num_heads": 8, "use_flash_attn": True, "fused_bias_fc": False, "dropout": 0.1}
        
        s4_layer_dict = {
            "_name_": "s4d",
            "d_state": 64,
            "transposed": False,
            "return_state": False
        }
        hyena_dict = {
            "_name_": "hyena",
            "emb_dim": 33,
            "filter_order": 64,
            "local_order": 3,
            "l_max": cfg.max_len,
            "fused_fft_conv": True,
            "modulate": True,
            "w": 14,
            "lr": self.lr,
            "lr_pos_emb": self.lr
        }
        
        # TODO: doesn't use lr finder for now
        for i in range(cfg.s4_layers):
            self.s4_layers.append(S4D(cfg.d_model, transposed=True,
                    lr=self.lr))


        self.nbins = 600
        for i in range(cfg.hyena_layers):
            self.hyena_layers.append(Block(
                # FlashMHA(cfg.d_model, cfg.num_heads), seqlen=self.nbins
                # torch.nn.MultiheadAttention(cfg.d_model, cfg.num_heads), seqlen=self.nbins
                torch.nn.TransformerEncoderLayer(cfg.d_model, cfg.num_heads,batch_first=True),
                seqlen=self.nbins
            ))


        # Project from d_model to num_words (80 bins for mel spectrogram)
        self.linear_encoder = nn.Conv1d(cfg.input_channels, cfg.d_model, 1)
        # we hardcode settings such that L=262144 -> L=3000
        self.avg_pool = nn.AvgPool1d(8, 8) # influences self.nbins
        self.ln = nn.LayerNorm([self.nbins, cfg.d_model])
        
        self.w_out = nn.Linear(cfg.d_model, num_outs)
            
    def forward(self, x_feat, x, session_ids):
        # print(f"pre encoder {x.shape=}")
        x = x.transpose(-1, -2)
        x = self.linear_encoder(x)
        # print(f"post encoder {x.shape=}")
        for layer in self.s4_layers:
            x, _ = layer(x)
        # print(f"post s4_layers {x.shape=}")
        x = self.avg_pool(x)
        # print(f"post avg_pool {x.shape=}")
        x = x.transpose(-1, -2)
        # x = whisper.pad_or_trim(x, 128, axis=1)
        for layer in self.hyena_layers:
            x = layer(x)
        # print(f"post attention {x.shape=}")
        x = self.ln(x)
        x = self.w_out(x)
        # print(f"final out {x.shape=}")
        return x
    
    def configure_optimizers(self):
        """Create optimizers and schedulers."""
        optimizer = torch.optim.AdamW(self.parameters(), 
                          lr=self.hparams.learning_rate, 
                          eps=self.hparams.adam_epsilon,
                          betas=(0.9, 0.98))
        self.optimizer = optimizer

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.hparams.warmup_steps, 
            num_training_steps = self.steps_per_epoch // self.hparams.gradient_accumulation_steps * self.hparams.num_train_epochs
        )
        lr_scheduler = {'scheduler': scheduler, 'interval': 'step'}
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
    
    lr_scheduler_step = pl.LightningModule.lr_scheduler_step

config = S4HyenaConfig()

model = S4HyenaModule(config, datamodule.val.text_transform, lm_directory)

##
log_neptune = True
# log_neptune = False
auto_lr_find = True
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
                "HyenaWhisper",
                "SafariHyena"
                ],
        log_model_checkpoints=False,
        capture_hardware_metrics=True,
        capture_stderr=True,
        capture_stdout=True,
    )
    # neptune_logger.log_hyperparams(params)

    checkpoint_callback = ModelCheckpoint(
        monitor="val/wer",
        mode="min",
        dirpath=output_directory,
        filename=model.__class__.__name__+"-{epoch:02d}-{val/wer:.3f}",
        # save_on_train_epoch_end=False # run after validation
    )
    callbacks.extend([
        checkpoint_callback,
        # pl.callbacks.LearningRateMonitor(logging_interval="epoch"),
        pl.callbacks.LearningRateMonitor(logging_interval="step"), # good for troubleshooting warmup
    ])
    
    neptune_logger.log_hyperparams(vars(config))
else:
    neptune_logger = None
    callbacks = []
    
# if auto_lr_find:
#     callbacks.append(LearningRateFinder())
    
if len(callbacks) == 0:
    callbacks = None

trainer = pl.Trainer(
    max_epochs=config.num_train_epochs,
    # devices=["cuda:0", "cuda:1"],
    # devices=[1],
    devices="auto",
    accelerator="gpu",
    # strategy="fsdp",
    # QUESTION: Gaddy accumulates grads from two batches, then does clip_grad_norm_
    # are we clipping first then addiing? (prob doesn't matter...)
    gradient_clip_val=10,
    logger=neptune_logger,
    default_root_dir=output_directory,
    callbacks=callbacks,
    precision=config.precision,
    check_val_every_n_epoch=5, # should be almost twice as fast
    num_sanity_val_steps=0,
)
if auto_lr_find:
    tuner = pl.tuner.Tuner(trainer)
    tuner.lr_find(model, datamodule=datamodule)
logging.info('about to fit')
# epoch of 242 if only train...
# trainer.fit(model, datamodule.train_dataloader(),
#             datamodule.val_dataloader())
# trainer.fit(model, train_dataloaders=datamodule.train_dataloader()) 
# note: datamodule.train_dataloader() can sometimes be slow depending on Oak filesystem
# we should prob transfer this data to $LOCAL_SCRATCH first...
# trainer.validate(model, dataloaders=datamodule.val_dataloader())
trainer.fit(model, train_dataloaders=datamodule.train_dataloader(),
            val_dataloaders=datamodule.val_dataloader()) 

##
