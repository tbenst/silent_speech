##
2
# TODO:
# run Frank's tensorflow code
# - per-block z-score normalization
# - check if my phonemes are correct (compare to frank's)
##
# %load_ext autoreload
# %autoreload 2
##
import os, subprocess
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "backend:cudaMallocAsync" # no OOM
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
hostname = subprocess.run("hostname", capture_output=True)
ON_SHERLOCK = hostname.stdout[:2] == b"sh"

import pytorch_lightning as pl, pickle
import sys, warnings
import numpy as np
import logging
import torchmetrics
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

from read_emg import EMGDataset, PreprocessedEMGDataset, \
    PreprocessedSizeAwareSampler, EMGDataModule, ensure_folder_on_scratch
from architecture import Model, S4Model, H3Model, ResBlock, MONAConfig, MONA, LinearDispatch, XtoTextConfig, XtoText
from data_utils import combine_fixed_length, decollate_tensor
from transformer import TransformerEncoderLayer
from pytorch_lightning.loggers import NeptuneLogger
# import neptune, shutil
import neptune.new as neptune, shutil
import typer
from datetime import datetime
from pytorch_lightning.callbacks import ModelCheckpoint, GradientAccumulationScheduler
from pytorch_lightning.profilers import SimpleProfiler, AdvancedProfiler, PyTorchProfiler, PassThroughProfiler
from pytorch_lightning.strategies import DDPStrategy
from data_utils import TextTransform, in_notebook
from typing import List
from collections import defaultdict
from enum import Enum
from magneto.preprocessing import ensure_data_on_scratch
from dataloaders import LibrispeechDataset, EMGAndSpeechModule, \
    DistributedStratifiedBatchSampler, StratifiedBatchSampler, cache_dataset, \
    split_batch_into_emg_neural_audio, DistributedSizeAwareStratifiedBatchSampler, \
    SizeAwareStratifiedBatchSampler, collate_gaddy_or_speech, \
    collate_gaddy_speech_or_neural, DistributedSizeAwareSampler, \
    T12DataModule, T12Dataset, NeuralDataset, T12CompDataModule
from functools import partial
from contrastive import cross_contrastive_loss, var_length_cross_contrastive_loss, \
    nobatch_cross_contrastive_loss, supervised_contrastive_loss
import glob, scipy
from helpers import load_npz_to_memory
from typing_extensions import Annotated

DEBUG = False
DEBUG = True
RESUME = False
# RESUME = True


constant_offset_sd = 0.2
white_noise_sd = 0.8 # might be 1.0 in frank's code (see speech_release_baseline.yaml)
# constant_offset_sd = 0
# white_noise_sd = 0
seqlen = 600
auto_lr_find = False

# see https://github.com/fwillett/speechBCI/blob/main/NeuralDecoder/neuralDecoder/configs/config.yaml
# learning_rate = 1e-3 # frank used 1e-2. but we saw lar spike from 3 to 8 in validation...
learning_rate = 2e-4
# learning_rate = 1.5e-4
togglePhones = True

if RESUME:
    # TODO: make an auto-resume feature...? or at least find ckpt_path from run_id
    # to think about: can we do this automatically on gaia/sherlock if OOM..? (maybe we don't care / can do manually)
    # INFO: when resuming logging to Neptune, we might repeat some steps,
    # e.g. if epoch 29 was lowest WER, but we resume at epoch 31, we will
    # log epoch 30 & 31 twice. mainly an issue for publication plots
    ckpt_path = '/scratch/2023-'
    run_id = 'GAD-493'
    

per_index_cache = True # read each index from disk separately
# per_index_cache = False # read entire dataset from disk


isotime = datetime.now().isoformat()
precision = 32
grad_accum = 2
NUM_GPUS = 2
limit_train_batches = None
limit_val_batches = None
log_neptune = True
# log_neptune = False
n_epochs = 200
num_sanity_val_steps = 0 # may prevent crashing of distributed training
# grad_accum = 2 # NaN loss at epoch 67 with BatchNorm, two gpu, grad_accum=2, base_bz=16

# if BatchNorm still causes issues can try RunningBatchNorm (need to implement for distributed)
# https://youtu.be/HR0lt1hlR6U?t=7543
logger_level = logging.WARNING
devices = 'auto'

assert os.environ["NEPTUNE_ALLOW_SELF_SIGNED_CERTIFICATE"] == 'TRUE', "run this in shell: export NEPTUNE_ALLOW_SELF_SIGNED_CERTIFICATE='TRUE'"

# load our data file paths and metadata:

if per_index_cache:
    cache_suffix = "_per_index"
else:
    cache_suffix = ""
if ON_SHERLOCK:
    sessions_dir = '/oak/stanford/projects/babelfish/magneto/'
    # TODO: bechmark SCRATCH vs LOCAL_SCRATCH ...?
    scratch_directory = os.environ["SCRATCH"]
    # scratch_directory = os.environ["LOCAL_SCRATCH"]
    gaddy_dir = '/oak/stanford/projects/babelfish/magneto/GaddyPaper/'
    scratch_lengths_pkl = os.path.join(scratch_directory, "2023-07-25_emg_speech_dset_lengths.pkl")
    tmp_lengths_pkl = os.path.join("/tmp", "2023-07-25_emg_speech_dset_lengths.pkl")
    if os.path.exists(scratch_lengths_pkl) and not os.path.exists(tmp_lengths_pkl):
        shutil.copy(scratch_lengths_pkl, tmp_lengths_pkl)
    T12_dir = os.path.join(scratch_directory, "T12_data")
    if len(os.sched_getaffinity(0)) > 16:
        print("WARNING: if you are running more than one script, you may want to use `taskset -c 0-16` or similar")
else:
    # on my local machine
    sessions_dir = '/data/magneto/'
    scratch_directory = "/scratch"
    gaddy_dir = '/scratch/GaddyPaper/'
    T12_dir = "/data/data/T12_data_v4/"

print(f"CPU affinity: {os.sched_getaffinity(0)}")

data_dir = os.path.join(gaddy_dir, 'processed_data/')
lm_directory = os.path.join(gaddy_dir, 'pretrained_models/librispeech_lm/')
normalizers_file = os.path.join(SCRIPT_DIR, "normalizers.pkl")

if ON_SHERLOCK:
    lm_directory = ensure_folder_on_scratch(lm_directory, scratch_directory)
    
gpu_ram = torch.cuda.get_device_properties(0).total_memory / 1024**3
# print("TODO FIXME: hardcoded gpu_ram")
# gpu_ram = 31

if gpu_ram < 24:
    # Titan RTX
    # val_bz = 16 # OOM
    # base_bz = 32
    # base_bz = 24
    # base_bz = 16
    base_bz = 32 # for WilletModel (bz=64)
    val_bz = 16
    # max_len = 24000 # OOM
    max_len = 12000 # approx 11000 / 143 = 77 bz. 75 * 2 GPU = 150 bz. still high..?
    # max_len = 18000 # no OOM, approx 110 bz (frank used 64)
    # assert NUM_GPUS == 2
elif gpu_ram > 30:
    # V100
    # base_bz = 16
    base_bz = 32 # for WilletModel (bz=64)
    # base_bz = 48
    val_bz = 16
    max_len = 48000
    # assert NUM_GPUS == 4
else:
    raise ValueError("Unknown GPU")

app = typer.Typer()

@app.command()
def update_configs(
    constant_offset_sd_cli: float = typer.Option(constant_offset_sd, "--constant-offset-sd"),
    white_noise_sd_cli: float = typer.Option(white_noise_sd, "--white-noise-sd"),
    learning_rate_cli: float = typer.Option(learning_rate, "--learning-rate"),
    debug_cli: bool = typer.Option(False, "--debug"),
    phonemes:Annotated[
        bool,
        typer.Option(
            help="Use Phonemes.", rich_help_panel="Train CTC on phoneme labels."
        ),
    ] = True,
    resume_cli: bool = typer.Option(RESUME, "--resume"),
    grad_accum_cli: int = typer.Option(grad_accum, "--grad-accum"),
    precision_cli: str = typer.Option(precision, "--precision"),
    logger_level_cli: str = typer.Option("WARNING", "--logger-level"),
    base_bz_cli: int = typer.Option(base_bz, "--base-bz"),
    val_bz_cli: int = typer.Option(val_bz, "--val-bz"),
    max_len_cli: int = typer.Option(max_len, "--max-len"),
    seqlen_cli: int = typer.Option(seqlen, "--seqlen"),
    # devices_cli: str = typer.Option(devices, "--devices"),
):
    """Update configurations with command-line values."""
    global constant_offset_sd, white_noise_sd, DEBUG, RESUME, grad_accum
    global precision, logger_level, base_bz, val_bz, max_len, seqlen
    global learning_rate, devices, togglePhones
    # devices = int(NUM_GPUS) # TODO maybe need "auto" ..? idk
    # try:
    #     devices = int(devices_cli) # eg "2" -> 2
    # except:
    #     pass
    togglePhones = phonemes
    learning_rate = learning_rate_cli
    constant_offset_sd = constant_offset_sd_cli
    white_noise_sd = white_noise_sd_cli
    DEBUG = debug_cli
    RESUME = resume_cli
    grad_accum = grad_accum_cli
    precision = precision_cli
    logger_level = getattr(logging, logger_level_cli.upper())
    base_bz = base_bz_cli
    val_bz = val_bz_cli
    max_len = max_len_cli
    seqlen = seqlen_cli
    print("Updated configurations using command-line arguments.")

if __name__ == "__main__" and not in_notebook():
    try:
        app()
    except SystemExit as e:
        pass

if DEBUG:
    NUM_GPUS = 1
    limit_train_batches = 2
    limit_val_batches = 2 # will not run on_validation_epoch_end
    # NUM_GPUS = 2
    # limit_train_batches = None
    # limit_val_batches = None
    log_neptune = False
    n_epochs = 2
    # precision = "16-mixed"
    num_sanity_val_steps = 2
    grad_accum = 1
    logger_level = logging.DEBUG
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


if NUM_GPUS > 1:
    strategy=DDPStrategy(gradient_as_bucket_view=True, find_unused_parameters=True)
elif NUM_GPUS == 1:
    strategy = "auto"
else:
    strategy = "auto"
    
devices = NUM_GPUS # TODO maybe need "auto" ..? idk

##

if ON_SHERLOCK:
    # TODO: should we just use the scratch directory over LOCAL_SCRATCH?
    output_directory = os.path.join(os.environ["SCRATCH"], f"{isotime}_gaddy")
else:
    output_directory = os.path.join(scratch_directory, f"{isotime}_gaddy")

logging.basicConfig(handlers=[
        logging.StreamHandler()
        ], level=logger_level, format="%(message)s",
        force=True)

logging.debug("DEBUG mode")
if not log_neptune:
    logging.warning("not logging to neptune")
    
# TODO: comment out
##
    
datamodule = T12CompDataModule(os.path.join(T12_dir, 'competitionData'),
    train_bz=base_bz, val_bz=val_bz,
    white_noise_sd=white_noise_sd, constant_offset_sd=constant_offset_sd,
    togglePhones=togglePhones, smoothing_sigma=2)

text_transform = TextTransform(togglePhones = togglePhones)
os.makedirs(output_directory, exist_ok=True)

steps_per_epoch = len(datamodule.train) // base_bz // NUM_GPUS // grad_accum

##

n_chars = len(text_transform.chars)
num_outs = n_chars + 1 # +1 for CTC blank token ( i think? )

@dataclass
class WillettConfig(XtoTextConfig):
    num_outs: int = 41
    rnn_stride:int = 4
    rnn_kernel_size:int = 14
    rnn_dropout:float = 0.4
    d_model: int = 512
    neural_reduced_features: int = 256
    num_layers: int = 5
    input_dropout: float = 0.2
    neural_input_features:int = datamodule.train.n_features
            
class WillettModel(XtoText):
    def __init__(self, cfg:WillettConfig, text_transform:TextTransform, sessions:List[str]):
        super().__init__(cfg, text_transform)
        self.session_input_encoder = LinearDispatch(sessions,
            cfg.neural_input_features, cfg.neural_reduced_features)
        # https://github.com/fwillett/speechBCI/blob/ba3440432893e75d9413e55ed15e8a6d31034f9b/NeuralDecoder/neuralDecoder/configs/model/gru_stack_inputNet.yaml#L4
        self.neural_input_dropout = nn.Dropout(cfg.input_dropout)
        self.neural_input_act = nn.Softsign()
        # input, hidden, num_layers
        self.rnn = nn.GRU(cfg.neural_reduced_features * cfg.rnn_kernel_size, cfg.d_model, cfg.num_layers,
                          batch_first=True, dropout=cfg.rnn_dropout)
        # Willett only had learnable initial state for first layer, but that's hard to do in pytorch
        # https://github.com/fwillett/speechBCI/blob/ba3440432893e75d9413e55ed15e8a6d31034f9b/NeuralDecoder/neuralDecoder/models.py#L80
        self.rnn_initial_state = nn.Parameter(torch.randn(cfg.num_layers, 1, cfg.d_model))
        
        self.char_out = nn.Linear(cfg.d_model, cfg.num_outs)
        
        self.learning_rate = cfg.learning_rate
        self.weight_decay = cfg.weight_decay
        self.rnn_stride = cfg.rnn_stride
        self.rnn_kernel_size = cfg.rnn_kernel_size
        
    def forward(self, batch):
        sessions = batch['sessions']
        emg_tup, neural_tup, audio_tup, idxs = split_batch_into_emg_neural_audio(batch)
        neural, length_neural, neural_phonemes, y_length_neural, y_neural = neural_tup
        x = nn.utils.rnn.pad_sequence(neural, batch_first=True) # B x T x C
        x = self.session_input_encoder(x, sessions)
        x = self.neural_input_dropout(x)
        x = self.neural_input_act(x)
        # print(f"neural_forward: {x.shape}") # 32 859 256
        #   kernel_size: 14
        #  strides: 4
        x = x.unfold(1, self.rnn_kernel_size, self.rnn_stride) # 32 212 256 14
        x = x.flatten(2) # 32, 212, 2968
        # print(f"flatten: {x.shape}")
        x, _ = self.rnn(x, self.rnn_initial_state.repeat(1, x.shape[0], 1))
        # print(f"rnn: {x.shape}")
        pred = F.log_softmax(self.char_out(x),2)
        # print(f"char_out: {pred.shape}")
        return {"pred": pred, "y_neural": y_neural,
                "length_neural": length_neural, "y_length_neural": y_length_neural}
    
    def calc_loss(self, pred, y_neural, length_neural, y_length_neural):
        # logic may be wrong for rnn_stride != 4
        ln = [ int(np.floor((l - self.rnn_kernel_size) / self.rnn_stride)) for l in length_neural]
        ctc_loss = self.ctc_loss(pred, y_neural, ln, y_length_neural)
        return {
            'loss': ctc_loss,
            'neural_ctc_loss': ctc_loss,
            'neural_bz': len(y_neural)
        }
    
    def training_step(self, batch, batch_idx):
        c = self.calc_loss(**self(batch))
        loss = c['loss']
        neural_ctc_loss = c['neural_ctc_loss']
        neural_bz = c['neural_bz']
        
        
        self.log("train/loss", loss,
                 on_step=False, on_epoch=True, logger=True, prog_bar=True, batch_size=neural_bz, sync_dist=True)
        self.log("train/neural_ctc_loss", neural_ctc_loss,
            on_step=False, on_epoch=True, logger=True, prog_bar=False, batch_size=neural_bz, sync_dist=True)
        torch.cuda.empty_cache()
        return loss
        

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.trainer.model.parameters(), lr=self.learning_rate,
                                      weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0,
                                                      total_iters=100000)
        lr_scheduler = {'scheduler': scheduler, 'interval': 'step'}
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
        
config = WillettConfig(steps_per_epoch=steps_per_epoch, lm_directory=lm_directory, num_outs=num_outs,
                       learning_rate=1e-2, weight_decay=1e-5)
model = WillettModel(config, text_transform, datamodule.train.unique_sessions)

logging.info('made model')
##
callbacks = [
    # starting at epoch 0, accumulate this many batches of gradients
    GradientAccumulationScheduler(scheduling={0: config.gradient_accumulation_steps})
]

if log_neptune:
    # need to store credentials in your shell env
    nep_key = os.environ["NEPTUNE_API_TOKEN"]
    neptune_kwargs = {
        "project": "neuro/Gaddy",
        "name": model.__class__.__name__,
        "tags": [model.__class__.__name__,
                isotime,
                f"fp{config.precision}",
                ],
    }
    if RESUME:
        neptune_logger = NeptuneLogger(
            run = neptune.init_run(with_id=run_id,
                api_token=os.environ["NEPTUNE_API_TOKEN"],
                **neptune_kwargs),
            log_model_checkpoints=False
        )
    else:
        neptune_logger = NeptuneLogger(api_key=nep_key,
            **neptune_kwargs,
            log_model_checkpoints=False
        )
        neptune_logger.log_hyperparams(vars(config))
        neptune_logger.experiment["isotime"] = isotime
        neptune_logger.experiment["hostname"] = hostname.stdout.decode().strip()
        neptune_logger.experiment["output_directory"] = output_directory
        if "SLURM_JOB_ID" in os.environ:
            neptune_logger.experiment["SLURM_JOB_ID"] = os.environ["SLURM_JOB_ID"]


    checkpoint_callback = ModelCheckpoint(
        monitor="val/wer",
        mode="min",
        dirpath=output_directory,
        save_top_k=10, # TODO: try averaging weights afterwards to see if improve WER..?
        filename=model.__class__.__name__+"-{epoch:02d}-{val/wer:.3f}",
    )
    callbacks.extend([
        checkpoint_callback,
        pl.callbacks.LearningRateMonitor(logging_interval="epoch"),
        # pl.callbacks.LearningRateMonitor(logging_interval="step"), # good for troubleshooting warmup
    ])
else:
    neptune_logger = None

trainer = pl.Trainer(
    max_epochs=config.num_train_epochs,
    devices=devices,
    accelerator="gpu",
    # accelerator="cpu",
    gradient_clip_val=10, # was 0.5 for best 26.x% run, gaddy used 10, llama 2 uses 1.0, frank used 10
    logger=neptune_logger,
    default_root_dir=output_directory,
    callbacks=callbacks,
    precision=config.precision,
    limit_train_batches=limit_train_batches,
    limit_val_batches=limit_val_batches,
    # strategy=strategy,
    # use_distributed_sampler=True,
    # use_distributed_sampler=False, # we need to make a custom distributed sampler
    # num_sanity_val_steps=num_sanity_val_steps,
    sync_batchnorm=True,
    strategy=strategy,
    # strategy='fsdp', # errors on CTC loss being used on half-precision.
    # also model only ~250MB of params, so fsdp may be overkill
    # check_val_every_n_epoch=10 # should give speedup of ~30% since validation is bz=1
    num_sanity_val_steps=0,
    # https://lightning.ai/docs/pytorch/stable/debug/debugging_intermediate.html#detect-autograd-anomalies
    # detect_anomaly=True # slooooow
)

if auto_lr_find:
    # TODO: might be deprecated
    # https://lightning.ai/docs/pytorch/stable/upgrade/from_1_9.html
    # https://lightning.ai/docs/pytorch/stable/advanced/training_tricks.html#learning-rate-finder
    tuner = pl.tuner.Tuner(trainer)
    tuner.lr_find(model, datamodule)
        
logging.info('about to fit')
print(f"Sanity check: {len(datamodule.train)} training samples")
print(f"Sanity check: {len(datamodule.train_dataloader())} training batches")
# epoch of 242 if only train...
if RESUME:
    trainer.fit(model, datamodule=datamodule,
        ckpt_path=ckpt_path)
else:
    trainer.fit(model, datamodule=datamodule)
    
if log_neptune:
    ckpt_path = os.path.join(output_directory,f"finished-training_epoch={model.current_epoch}.ckpt")
    trainer.save_checkpoint(ckpt_path)
    print(f"saved checkpoint to {ckpt_path}")

##
