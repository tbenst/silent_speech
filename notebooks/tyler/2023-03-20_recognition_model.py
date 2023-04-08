'''
This script has been modified to use torchaudio in place of ctcdecode. On standard benchmarks it provides a ~10-100x speed improvement.

Download the Relevant LM files and then point script toward a directory holding the files via --lm_directory flag. Files can be obtained through:

wget -c https://download.pytorch.org/torchaudio/download-assets/librispeech-3-gram/{lexicon.txt, tokens.txt, lm.bin}

'''
##
# %load_ext autoreload
# %autoreload 2
##
import pytorch_lightning as pl
import os
import sys
import numpy as np
import logging
import subprocess
import jiwer
import random

import torch
from torch import nn
import torch.nn.functional as F
from torchaudio.models.decoder import ctc_decoder

# horrible hack to get around this repo not being a proper python package
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(SCRIPT_DIR)

from read_emg import EMGDataset, SizeAwareSampler, PreprocessedEMGDataset, PreprocessedSizeAwareSampler, EMGDataModule
from architecture import Model, S4Model, H3Model
from data_utils import combine_fixed_length, decollate_tensor
from transformer import TransformerEncoderLayer
from pytorch_lightning.loggers import NeptuneLogger
import neptune
from datetime import datetime

isotime = datetime.now().isoformat()
hostname = subprocess.run("hostname", capture_output=True)
ON_SHERLOCK = hostname.stdout[:2] == b"sh"

assert os.environ["NEPTUNE_ALLOW_SELF_SIGNED_CERTIFICATE"] == 'TRUE', "run this in shell: export NEPTUNE_ALLOW_SELF_SIGNED_CERTIFICATE='TRUE'"

# load our data file paths and metadata:
if ON_SHERLOCK:
    sessions_dir = '/oak/stanford/projects/babelfish/magneto/'
else:
    sessions_dir = '/data/magneto/'

##
auto_lr_find = False
debug = False
max_len = 128000 * 2
log_neptune = True
output_directory = os.path.join(os.environ["LOCAL_SCRATCH"], f"{isotime}_gaddy")
S4 = 0
# batch_size = 32
precision = 16
precision = 32
learning_rate = 3e-4
epochs = 200
# TODO: lr should not jump
 # account for accum gradient on two batches (gaddy counts iterations not backprop steps)
# learning_rate_warmup = 16
learning_rate_warmup = 500
learning_rate_patience = 5
start_training_from = None
model_size = 768 # number of hidden dimensions
num_layers = 6 # number of layers
dropout = .2 # dropout
l2 = 0
evaluate_saved = None
lm_directory = '/oak/stanford/projects/babelfish/magneto/GaddyPaper/pretrained_models/librispeech_lm/'
base_dir = '/oak/stanford/projects/babelfish/magneto/GaddyPaper/processed_data/'
normalizers_file = os.path.join(SCRIPT_DIR, "normalizers.pkl")
seqlen       = 600
togglePhones = False


##

os.makedirs(output_directory, exist_ok=True)
logging.basicConfig(handlers=[
        logging.FileHandler(os.path.join(output_directory, 'log.txt'), 'w'),
        logging.StreamHandler()
        ], level=logging.INFO, format="%(message)s")


datamodule = EMGDataModule(base_dir, togglePhones, normalizers_file, max_len=max_len)

logging.info('output example: %s', datamodule.val.example_indices[0])
logging.info('train / dev split: %d %d',len(datamodule.train),len(datamodule.val))
##
n_chars = len(datamodule.val.text_transform.chars)
num_outs = n_chars+1
steps_per_epoch = len(datamodule.train_dataloader()) # todo: double check this is 242
model = Model(datamodule.val.num_features, model_size, dropout, num_layers,
              num_outs, datamodule.val.text_transform,
              steps_per_epoch=steps_per_epoch, epochs=epochs, lr=learning_rate,
              learning_rate_warmup=learning_rate_warmup)
##
params = {
    "num_features": datamodule.val.num_features, "model_size": model_size,
    "dropout": dropout, "num_layers": num_layers,
    "num_outs": num_outs, "lr": learning_rate
}

# neptune_logger = None

neptune_logger = NeptuneLogger(
    # need to store credentials in your shell env
    api_key=os.environ["NEPTUNE_API_TOKEN"],
    project="neuro/Gaddy",
    # name=magneto.fullname(model), # from lib
    name=model.__class__.__name__,
    tags=[model.__class__.__name__,
            "MultiStepLR",
            "AdamW",
            f"fp{precision}",
            "MultiStepLR",
            "800Hz",
            "8xDownsampling",
            "FCN_embedding",
            ],
    log_model_checkpoints=False,
)
neptune_logger.log_hyperparams(params)

if ON_SHERLOCK:
    # pl_root_dir = os.environ["SCRATCH"]
    pl_root_dir = os.environ["LOCAL_SCRATCH"]
else:
    pl_root_dir = "/scratch"
    
callbacks = [
    pl.callbacks.LearningRateMonitor(logging_interval="step"),
]

# QUESTION: why does validation loop become massively slower as training goes on?
# perhaps this line will resolve..?
# export NEPTUNE_ALLOW_SELF_SIGNED_CERTIFICATE='TRUE'
trainer = pl.Trainer(
    max_epochs=epochs,
    devices=[0],
    # devices=[1],
    accelerator="gpu",
    # QUESTION: Gaddy accumulates grads from two batches, then does clip_grad_norm_
    # are we clipping first then addiing? (prob doesn't matter...)
    gradient_clip_val=10,
    accumulate_grad_batches=2,
    logger=neptune_logger,
    default_root_dir=pl_root_dir,
    callbacks=callbacks,
    precision=precision,
    enable_checkpointing=False
)

if auto_lr_find:
    tuner = pl.tuner.Tuner(trainer)
    tuner.lr_find(model, datamodule)
        
trainer.fit(model, datamodule.train_dataloader(),
            datamodule.val_dataloader())

trainer.validate(model, dataloaders=datamodule.val_dataloader(), ckpt_path='best')
# trainer.test(model, dataloaders=dataloader, ckpt_path='best')
neptune_logger.experiment.stop()
##
