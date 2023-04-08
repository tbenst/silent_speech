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
from pytorch_lightning.callbacks import ModelCheckpoint, GradientAccumulationScheduler

isotime = datetime.now().isoformat()
hostname = subprocess.run("hostname", capture_output=True)
ON_SHERLOCK = hostname.stdout[:2] == b"sh"

assert os.environ["NEPTUNE_ALLOW_SELF_SIGNED_CERTIFICATE"] == 'TRUE', "run this in shell: export NEPTUNE_ALLOW_SELF_SIGNED_CERTIFICATE='TRUE'"

# load our data file paths and metadata:
if ON_SHERLOCK:
    sessions_dir = '/oak/stanford/projects/babelfish/magneto/'
    output_directory = os.environ["LOCAL_SCRATCH"]
else:
    sessions_dir = '/data/magneto/'
    output_directory = "/scratch"
output_directory = os.path.join(output_directory, f"{isotime}_gaddy")
##
auto_lr_find = False
debug = False
max_len = 128000 * 2
log_neptune = True
S4 = 0
batch_size = 32
precision = "16-mixed"
# precision = 32
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
logging.info('made model') # why is this sooo slow?? slash freezes..?
##
params = {
    "num_features": datamodule.val.num_features, "model_size": model_size,
    "dropout": dropout, "num_layers": num_layers,
    "num_outs": num_outs, "lr": learning_rate
}

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

    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss",
        mode="min",
        dirpath=output_directory,
        filename=model.__class__.__name__+"-{epoch:02d}-{val/loss:.3f}",
    )
    callbacks.extend([
        checkpoint_callback,
        pl.callbacks.LearningRateMonitor(logging_interval="epoch"),
        # pl.callbacks.LearningRateMonitor(logging_interval="step"), # good for troubleshooting warmup
    ])
else:
    neptune_logger = None
    callbacks = None

# QUESTION: why does validation loop become massively slower as training goes on?
# perhaps this line will resolve..?
# export NEPTUNE_ALLOW_SELF_SIGNED_CERTIFICATE='TRUE'

# TODO: at epoch 22 validation seems to massively slow down...?
# may be due to neptune...? (saw freeze on two models at same time...)
trainer = pl.Trainer(
    max_epochs=epochs,
    devices=[0],
    # devices=[1],
    accelerator="gpu",
    # QUESTION: Gaddy accumulates grads from two batches, then does clip_grad_norm_
    # are we clipping first then addiing? (prob doesn't matter...)
    gradient_clip_val=10,
    logger=neptune_logger,
    default_root_dir=output_directory,
    callbacks=callbacks,
    precision=precision,
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
trainer.fit(model, train_dataloaders=datamodule.train_dataloader(),
            val_dataloaders=datamodule.val_dataloader()) 

trainer.validate(model, dataloaders=datamodule.val_dataloader(), ckpt_path='best')
# trainer.test(model, dataloaders=dataloader, ckpt_path='best')
neptune_logger.experiment.stop()
##
