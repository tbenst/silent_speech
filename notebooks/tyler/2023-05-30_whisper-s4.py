##
# %load_ext autoreload
# %autoreload 2
##
import os
nep_key = "NEPTUNE_ALLOW_SELF_SIGNED_CERTIFICATE"
if not nep_key in os.environ or os.environ["NEPTUNE_ALLOW_SELF_SIGNED_CERTIFICATE"] != 'TRUE':
    os.environ["NEPTUNE_ALLOW_SELF_SIGNED_CERTIFICATE"] = 'TRUE'
    
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# https://stackoverflow.com/questions/73747731/runtimeerror-cuda-out-of-memory-how-setting-max-split-size-mb
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

import whisper
import pytorch_lightning as pl
import sys
from functools import lru_cache, partial
from magneto.models.s4d import S4D
from magneto.models.s4 import S4
# from magneto.models.hyena import HyenaOperator
from safari.models.sequence.hyena import HyenaOperator
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
from dataclasses import dataclass
from pytorch_lightning.callbacks import LearningRateFinder
from torch.distributed.fsdp.wrap import wrap

# horrible hack to get around this repo not being a proper python package
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(SCRIPT_DIR)

from read_emg import EMGDataset, SizeAwareSampler, PreprocessedEMGDataset, PreprocessedSizeAwareSampler, EMGDataModule
from data_utils import combine_fixed_length, decollate_tensor
from transformer import TransformerEncoderLayer
from pytorch_lightning.loggers import NeptuneLogger
import neptune, shutil
from datetime import datetime
from pytorch_lightning.callbacks import ModelCheckpoint, GradientAccumulationScheduler
from pytorch_lightning.profilers import SimpleProfiler, AdvancedProfiler, PyTorchProfiler
import random
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
wtokenizer = whisper.tokenizer.get_tokenizer(multilingual=True)

##

class PadAudioDataset(PreprocessedEMGDataset):
    def __init__(self, tokenizer, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.example_indices)
    
    @lru_cache(maxsize=None)
    def __getitem__(self, i):
        x = super().__getitem__(i)
        emg = x["raw_emg"] # samples x channels
        emg = emg / 38 # approximate normalization to [-1,1]
        emg = emg.swapaxes(0, 1) # channels x samples
        assert len(x["text"]) == 1
        text = x["text"][0]
        
        # gaddy 99% is length is 235390, longest is 299200, so pad to 2**18=262144
        # padded_emg = whisper.pad_or_trim(emg, 2**18, axis=-1)
        # whisper needs spectrogram length of 3000
        # padded_emg = whisper.pad_or_trim(emg, 2**18, axis=-1)
        # TODO: this is too small...
        # padded_emg = whisper.pad_or_trim(emg, 2**17, axis=-1)
        # padded_emg = whisper.pad_or_trim(emg, 2**16, axis=-1)

        text = [*self.tokenizer.sot_sequence_including_notimestamps] + self.tokenizer.encode(text)
        target_tokens = text[1:] + [self.tokenizer.eot]

        return {
            # "emg": padded_emg,
            "emg": emg,
            "target_tokens": target_tokens,
            "decoder_input_tokens": text
        }

##
_re_special = re.compile(r"\<\|.+?\|\>")
def strip_special_tokens(string):
    return re.sub(_re_special, "", string)

def filter_special_tokens(tokens, special_tokens=wtokenizer.encoding._special_tokens.values()):
    return [t for t in tokens if t not in special_tokens]

def whisper_data_collator_with_padding(features, eot_token_id=wtokenizer.eot):
        emgs, target_tokens, decoder_input_tokens = [], [], []
        for f in features:
            emg = f["emg"]
            padded_emg = whisper.pad_or_trim(emg, 2**18, axis=-1)
            emgs.append(padded_emg)
            target_tokens.append(f["target_tokens"])
            decoder_input_tokens.append(f["decoder_input_tokens"])

        emgs = torch.concat([emg[None, :] for emg in emgs])
        
        target_lengths = [len(lab) for lab in target_tokens]
        decoder_input_tokens_length = [len(e) for e in decoder_input_tokens]
        max_label_len = max(target_lengths+decoder_input_tokens_length)

        # tyler: I think -100 is arbitrary not-used token (never predicted), but not positive
        # idea from: https://github.com/openai/whisper/discussions/64 (see notebook)
        # or see Team Reflex google drive backup:
        # https://colab.research.google.com/drive/1Vu3cuAtM1Un56PwTOKyP0Xlua5MYs_-R?usp=sharing
        # 
        target_tokens = [np.pad(lab, (0, max_label_len - lab_len), 'constant', constant_values=-100) for lab, lab_len in zip(target_tokens, target_lengths)]
        decoder_input_tokens = [np.pad(e, (0, max_label_len - e_len), 'constant', constant_values=eot_token_id) for e, e_len in zip(decoder_input_tokens, decoder_input_tokens_length)]

        batch = {
            "target_tokens": target_tokens,
            "decoder_input_tokens": decoder_input_tokens
        }

        # TODO: is this really necessary
        batch = {k: torch.tensor(np.array(v), requires_grad=False) for k, v in batch.items()}
        batch["emg"] = emgs
        batch["lengths"] = [ex.shape[1] for ex in emgs]
        batch["target_lengths"] = [ex.shape[0] for ex in target_tokens]

        return batch
    
@dataclass
class WhisperConfig:
    model_name:str = "tiny"
    # "medium"
    # "small"
    # "base"
    # "tiny"
    lang:str = "en"
    steps_per_epoch:int = -1
    # learning_rate:float = 0.00025
    # learning_rate:float = 5e-4
    learning_rate:float = 5e-3
    # learning_rate:float = 5e-6
    weight_decay:float = 0.1
    adam_epsilon:float = 1e-8
    warmup_steps:int = 500
    # batch_size:int = 8 # 3:02 per epoch
    batch_size:int = 16 # 2:56 per epoch
    # batch_size:int = 24 # 4:15 per epoch
    # batch_size:int = 32 # 4:20 per epoch
    # batch_size:int = 2
    num_worker:int = 0
    num_train_epochs:int = 200
    gradient_accumulation_steps:int = 1
    sample_rate:int = 16000
    precision:str = "16-mixed"
    hyena_layers:int = 2
    hyena_dim:int = 64
    hyena_seq_len:int = 2**18
    hyena_order:int = 2
    hyena_filter_order:int = 64
    prenorm:bool = False
    dropout:float = 0.0
    in_channels:int = 8
    out_channels:int = 80

class WhisperModelModule(pl.LightningModule):
    def __init__(self, cfg:WhisperConfig,
                 train_dataset=[], eval_dataset=[]) -> None:
        super().__init__()
        self.options = whisper.DecodingOptions(language=cfg.lang, without_timestamps=True)
        # TODO: should we load a CPU first..?
        # eg whisper.load_model(model_name, 'cpu')
        self.whisper = whisper.load_model(cfg.model_name)
        self.tokenizer = whisper.tokenizer.get_tokenizer(True, language=cfg.lang, task=self.options.task)
        self.steps_per_epoch = cfg.steps_per_epoch

        # freeze encoder training
        # for p in self.whisper.encoder.parameters():
        #     p.requires_grad = False
        
        # freeze decoder
        for p in self.whisper.decoder.parameters():
            p.requires_grad = False
        
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        self.metrics_wer = evaluate.load("wer")
        self.metrics_cer = evaluate.load("cer")

        self.hparams.update(vars(config))
        self.__train_dataset = train_dataset
        self.__eval_dataset = eval_dataset
        
        # accumulate text over epoch for validation so we can caclulate WER
        self.step_target = []
        self.step_pred = []
        
        ################ S4Model emg -> (audio) mel spectrogram ###############
        self.encoder = nn.Linear(cfg.in_channels, cfg.hyena_dim)
        
        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms     = nn.ModuleList()
        self.dropouts  = nn.ModuleList()
        for _ in range(cfg.hyena_layers):
            self.s4_layers.append(
                # 5218MiB of VRAM
                # S4D(s4_cfg.d_model, dropout=s4_cfg.dropout, transposed=True,
                #     lr=s4_cfg.lr)
                # 4964MiB of VRAM
                # S4(s4_cfg.d_model, dropout=s4_cfg.dropout, bidirectional = True, transposed=True, 
                #                lr=s4_cfg.lr, mode = 'diag', measure = 'diag-inv', disc='zoh', real_type='exp')  
                HyenaOperator(
                    d_model=cfg.hyena_dim,
                    l_max=cfg.hyena_seq_len,
                    order=cfg.hyena_order,
                    filter_order=cfg.hyena_filter_order
                )
            )
            # self.norms.append(nn.LayerNorm(cfg.hyena_dim))
            self.dropouts.append(nn.Dropout1d(cfg.dropout))

        # Project from d_model to num_words (80 bins for mel spectrogram)
        self.linear_encoder = nn.Conv1d(cfg.hyena_dim, cfg.out_channels, 1)
        # we hardcode settings such that L=262144 -> L=3000
        self.spectrogram_pool = nn.AvgPool1d(87, 87)
        
        self.prenorm = cfg.prenorm
        
        self.input_dropout = nn.Dropout1d(0)
            
        #######################################################################

    def encode(self, x):
        """
        Use S4D to encode EMG signal into a mel spectrogam
        Input x is shape (B, d_input, L) where d_input is the number of EMG channels
        """
        # print("x.shape", x.shape)
        x = self.input_dropout(x)
        # print("x.shape (input_dropout)", x.shape)
        x = x.swapaxes(1,2) # (B, d_input, L) -> (B, L, d_input)
        x = self.encoder(x) # (B, L, d_input) -> (B, L, d_model)
        
        # x = nn.functional.softsign(x)

        # x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)

            z = x
            # Apply S4 block: we ignore the state input and output
            # z, _ = layer(z)
            
            # Apply hyena block
            z = layer(z)
            # print(f"post-layer {z=}")

            # Dropout on the output of the S4 block
            z = dropout(z)
            # print(f"post-dropout {z=}")

            # Residual connection
            x = z + x
            # print(f"post-residual {x=}")

        x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
        x = self.linear_encoder(x) # (B, d_model, L) -> (B, num_words, L)
        x = self.spectrogram_pool(x) # (B, num_words, L) -> (B, num_words, L')
        # print(x.shape)
        # whisper needs 80 x 3000 spectrogram
        # so we throw away last 13 bins
        # unfortunately, we need sample length that is power of 2,
        # and whisper needs divisible by 3000
        x = F.relu(x[...,:3000])
        # print(f"post Relu {x=}")
        # raise Exception("stop")
        # x = torch.complex(x, torch.zeros_like(x))
        # whisper expects input of batch x 80 x 3000
        return x
    
    def forward(self, x):
        return self.whisper(self.encode(x))

    def configure_sharded_model(self):
        self.encoder = wrap(self.encoder)
        self.s4_layers = wrap(self.s4_layers)
        self.norms = wrap(self.norms)
        self.dropouts = wrap(self.dropouts)
        self.linear_encoder = wrap(self.linear_encoder)
        

    def training_step(self, batch, batch_id):
        emg = batch["emg"]
        target_tokens = batch["target_tokens"].long()
        decoder_input_tokens = batch["decoder_input_tokens"].long()

        # no encoder training
        # with torch.no_grad():
        mel = self.encode(emg)
        audio_features = self.whisper.encoder(mel)

        out = self.whisper.decoder(decoder_input_tokens, audio_features)
        # pred = F.log_softmax(out, dim=-1)
        loss = self.loss_fn(out.view(-1, out.size(-1)), target_tokens.view(-1))
        # if torch.isnan(loss):
        #     raise Exception("Nan loss")
        # TODO: do we need to have a blank arg here?
        # print(f"{out.shape=}")
        # loss = F.ctc_loss(out, target_tokens, batch['lengths'], batch['target_lengths'])
        # loss = F.ctc_loss(out, target_tokens, pred.shape[1], target_tokens.shape[0])
        self.log("train/loss", loss, on_step=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_id):
        emg = batch["emg"]
        target_tokens = batch["target_tokens"].long()
        decoder_input_tokens = batch["decoder_input_tokens"].long()

        mel = self.encode(emg)
        audio_features = self.whisper.encoder(mel)
        out = self.whisper.decoder(decoder_input_tokens, audio_features)

        # pred = nn.utils.rnn.pad_sequence(F.log_softmax(out, dim=-1), batch_first=False)
        # print(f"\n{pred.shape=}, {target_tokens.shape=}, {batch['lengths']=}, {batch['target_lengths']=}")
        loss = self.loss_fn(out.view(-1, out.size(-1)), target_tokens.view(-1))
        # loss = F.ctc_loss(pred, target_tokens, batch['lengths'], batch['target_lengths'])
        # loss = F.ctc_loss(pred, target_tokens, pred.shape[1], target_tokens.shape[0])

        # TODO: should this be earlier? should we replace -100 with eot?
        out[out == -100] = self.tokenizer.eot
        target_tokens[target_tokens == -100] = self.tokenizer.eot

        o_list, l_list = [], []
        for o, l in zip(out, target_tokens):
            o = torch.argmax(o, dim=1)
            o_list.append(self.tokenizer.decode(filter_special_tokens(o,wtokenizer.encoding._special_tokens.values())))
            l_list.append(self.tokenizer.decode(filter_special_tokens(l,wtokenizer.encoding._special_tokens.values())))
            
        self.step_pred.extend(o_list)
        self.step_target.extend(l_list)

        self.log("val/loss", loss, prog_bar=True)

        return {
            "loss": loss
        }
        
    def on_validation_epoch_end(self):
        cer = self.metrics_cer.compute(references=self.step_target, predictions=self.step_pred)
        wer = self.metrics_wer.compute(references=self.step_target, predictions=self.step_pred)
        self.step_target.clear()
        self.step_pred.clear()
        self.log("val/cer", cer, prog_bar=True)
        self.log("val/wer", wer, prog_bar=True)
        return {
            "cer": cer,
            "wer": wer,
        }

    def configure_optimizers(self):
        """Create optimizers and schedulers."""
        # for whisper...
        model = self.whisper
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() 
                            if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() 
                            if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        # optimizer = torch.optim.AdamW(optimizer_grouped_parameters, 
        optimizer = torch.optim.AdamW(self.parameters(), 
                          lr=self.hparams.learning_rate, 
                          eps=self.hparams.adam_epsilon,
                          betas=(0.9, 0.98))
        self.optimizer = optimizer

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.hparams.warmup_steps, 
            num_training_steps = self.steps_per_epoch // self.hparams.gradient_accumulation_steps * self.hparams.num_train_epochs
        )
        # self.scheduler = scheduler
        lr_scheduler = {'scheduler': scheduler, 'interval': 'step'}
        # return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
    

SD = partial(PadAudioDataset, tokenizer=wtokenizer)
config = WhisperConfig(gradient_accumulation_steps=2)
# config = WhisperConfig(precision="32")
datamodule = EMGDataModule(data_dir, togglePhones, normalizers_file, max_len=max_len,
                           batch_size = config.batch_size, num_workers=config.num_worker,
                           drop_last=True, shuffle=True,
                           batch_sampler=False, collate_fn=whisper_data_collator_with_padding,
                           DatasetClass=SD)
# TODO: why are there only 503 steps per epoch?
config.steps_per_epoch = len(datamodule.train_dataloader()) # 503
##
# verify dataloader is working
td = datamodule.train_dataloader()
for b in tqdm(td):
    print(b.keys())
    print(b["target_tokens"].shape)
    print(b["emg"].shape)
    print(b["decoder_input_tokens"].shape)
    print(len(b["lengths"]), b["lengths"][0])
    print(len(b["target_lengths"]), b["target_lengths"][0])
    print(f"max emg: {b['emg'].max()}")
    break
##    
# wmodel = whisper.load_model("large")
n_mel_bins = 80
s4_config = S4Params(in_channels=8, lr=1e-4, fs=800,
        d_model=32,
        n_layers=2, bandpass=None,
        steps_per_epoch=None, batch_size=None, samples=None, trial_duration=None,
        num_words = n_mel_bins, num_examples = None,
        epochs=None)
whisper_model = WhisperModelModule(config, s4_config)

##
model = whisper_model
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
    devices=[1],
    # devices="auto",
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
layer = HyenaOperator(
    d_model=512, 
    l_max=1024, 
    order=2, 
    filter_order=64
)
x = torch.randn(1, 1024, 512)
y = layer(x)
    
print(x.shape, y.shape)