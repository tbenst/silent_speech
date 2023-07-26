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
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
import numpy as np
import logging
import subprocess
import jiwer
import random
from tqdm.auto import tqdm

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
# import neptune, shutil
import neptune.new as neptune, shutil
from datetime import datetime
from pytorch_lightning.callbacks import ModelCheckpoint, GradientAccumulationScheduler
from pytorch_lightning.profilers import SimpleProfiler, AdvancedProfiler, PyTorchProfiler

def ensure_folder_on_scratch(src, dst):
    "Check if folder exists on scratch, otherwise copy. Return new path."
    assert os.path.isdir(src)
    split_path = lm_directory.split(os.sep)
    name = split_path[-1] if split_path[-1] != '' else split_path[-2]
    out = os.path.join(dst,name)
    if not os.path.isdir(out):
        shutil.copytree(src, out)
    return out

from magneto.preprocessing import ensure_data_on_scratch

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
output_directory = os.path.join(scratch_directory, f"{isotime}_gaddy")
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
# 3e-3 leads to NaNs, prob need to have slower warmup in this case
epochs = 200
# epochs = 8
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
lm_directory = os.path.join(gaddy_dir, 'pretrained_models/librispeech_lm/')
data_dir = os.path.join(gaddy_dir, 'processed_data/')
normalizers_file = os.path.join(SCRIPT_DIR, "normalizers.pkl")
seqlen       = 600
togglePhones = False

# lm_directory = ensure_folder_on_scratch(lm_directory, scratch_directory)
# much faster if we intend to load data more than once during job;
# otherwise slightly slower as we first copy to local nvme then load to RAM
# data_dir = ensure_folder_on_scratch(data_dir, data_dir)
##

os.makedirs(output_directory, exist_ok=True)
logging.basicConfig(handlers=[
        logging.FileHandler(os.path.join(output_directory, 'log.txt'), 'w'),
        logging.StreamHandler()
        ], level=logging.INFO, format="%(message)s")


datamodule = EMGDataModule(data_dir, togglePhones, normalizers_file, max_len=max_len, batch_size=1)

logging.info('output example: %s', datamodule.val.example_indices[0])
logging.info('train / dev split: %d %d',len(datamodule.train),len(datamodule.val))
##
n_chars = len(datamodule.val.text_transform.chars)
num_outs = n_chars+1
steps_per_epoch = len(datamodule.train_dataloader()) # todo: double check this is 242
# profiler = None
# saves to `.neptune/fit-simple-profile.txt.txt`
# profiler = SimpleProfiler(filename="simple-profile")
profiler = AdvancedProfiler(dirpath=output_directory, filename="AdvancedProfiler")
# profiler = PyTorchProfiler(filename="profile")

profile_create_model = True
if profile_create_model:
    import cProfile
    my_profiler = cProfile.Profile()
    my_profiler.enable()
##
model = Model(model_size, dropout, num_layers,
              num_outs, datamodule.val.text_transform, lm_directory=lm_directory,
              steps_per_epoch=steps_per_epoch, epochs=epochs, lr=learning_rate,
              learning_rate_warmup=learning_rate_warmup, profiler=profiler)

if profile_create_model:
    my_profiler.disable()
    my_profiler.dump_stats(os.path.join(output_directory,"create_model.stats"))

# why is this sooo slow?? slash freezes..? are we hitting oak?
# TODO: benchmark with cProfiler. CPU & GPU are near 100% during however
# not always slamming CPU/GPU...
logging.info('made model')
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
                "GeLU",
                "LayerNorm",
                "MultiStepLR",
                "AdamW",
                f"fp{precision}",
                ],
        log_model_checkpoints=False,
    )
    neptune_logger.log_hyperparams(params)

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
# note: datamodule.train_dataloader() can sometimes be slow depending on Oak filesystem
# we should prob transfer this data to $LOCAL_SCRATCH first...
trainer.fit(model, train_dataloaders=datamodule.train_dataloader(),
            val_dataloaders=datamodule.val_dataloader()) 

trainer.save_checkpoint(os.path.join(output_directory,f"finished-training_epoch={epochs}.ckpt"))

# trainer.validate(model, dataloaders=datamodule.val_dataloader(), ckpt_path='best')
# trainer.test(model, dataloaders=dataloader, ckpt_path='best')
neptune_logger.experiment.stop()
##
trainer.logger = False
trainer.validate(model, dataloaders=datamodule.val_dataloader())
##
datamodule.val_test_batch_sampler = False
val1_dl = datamodule.val_dataloader()

datamodule.val_test_batch_sampler = True
valbs_dl = datamodule.val_dataloader()
##
def on_validation_epoch_start(self):
    self._init_ctc_decoder()

def on_validation_epoch_end(self):
    step_target = []
    step_pred = []
    for t,p in zip(self.step_target, self.step_pred):
        if len(t) > 0:
            step_target.append(t)
            step_pred.append(p)
        else:
            print("WARN: got target length of zero during validation.")
        if len(p) == 0:
            print("WARN: got prediction length of zero during validation.")
    wer = jiwer.wer(step_target, step_pred)
    self.step_target.clear()
    self.step_pred.clear()
    return wer

def validation_step(self, batch, batch_idx):
    loss, bz = self.calc_loss(batch)
    target_text, pred_text = _beam_search_step(batch)
    # target_text, pred_text = self._beam_search_step(batch)
    if len(target_text) > 0:
        self.step_target.append(target_text)
        self.step_pred.append(pred_text)
        # self.step_target.extend(target_text)
        # self.step_pred.extend(pred_text)
    self.log("val/loss", loss, prog_bar=True, batch_size=bz)
    return loss


def _beam_search_step(self, batch):
    "Repeatedly called by validation_step & test_step. Impure function!"
    X     = batch['emg'][0].unsqueeze(0)
    X_raw = batch['raw_emg'][0].unsqueeze(0)
    sess  = batch['session_ids'][0]

    pred  = F.log_softmax(self(X, X_raw, sess), -1).cpu()

    beam_results = self.ctc_decoder(pred)
    pred_int     = beam_results[0][0].tokens
    pred_text    = ' '.join(beam_results[0][0].words).strip().lower()
    target_text  = self.text_transform.clean_2(batch['text'][0][0])

    return target_text, pred_text
##
with torch.no_grad():
    on_validation_epoch_start(model)
    for b,batch in enumerate(tqdm(val1_dl)):
        model.validation_step(batch,b)
    on_validation_epoch_end(model)
##
with torch.no_grad():
    on_validation_epoch_start(model)
    for b,batch in enumerate(tqdm(valbs_dl)):
        model.validation_step(batch,b)
    on_validation_epoch_end(model)
##
class D:
    def __init__(self) -> None:
        pass
    
    x = 3
    
D.x
##
