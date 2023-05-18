##
# %load_ext autoreload
# %autoreload 2
##
import os
nep_key = "NEPTUNE_ALLOW_SELF_SIGNED_CERTIFICATE"
if not nep_key in os.environ or os.environ["NEPTUNE_ALLOW_SELF_SIGNED_CERTIFICATE"] != 'TRUE':
    os.environ["NEPTUNE_ALLOW_SELF_SIGNED_CERTIFICATE"] = 'TRUE'
    
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import whisper
import pytorch_lightning as pl
import sys
from functools import lru_cache, partial
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

class SpectrogramDataset(PreprocessedEMGDataset):
    def __init__(self, tokenizer, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.example_indices)
    
    @lru_cache(maxsize=None)
    def __getitem__(self, i):
        x = super().__getitem__(i)
        emg = x["raw_emg"] # samples x channels
        # resample from 1kHz to 2kHz using torch
        # TODO: test training on scipy..?
        emg = torch.nn.functional.interpolate(emg.transpose(0,1).unsqueeze(0), scale_factor=2, mode="linear").squeeze(0).transpose(0,1)
        # guessing scipy.signal.resample is better
        # emg = torch.tensor(scipy.signal.resample(emg.numpy(), 2 * emg.shape[0], axis=0))
        
        flat_emg = emg.reshape(-1) # flatten to look more like audio
        assert len(x["text"]) == 1
        text = x["text"][0]
        audio = whisper.pad_or_trim(flat_emg) # whisper expects up to 480k samples; 16kHz * 30s
        # Gaddy paper uses 1kHz, and average is approx 4.5s of data, 99-percentile length is 29.7s
        mel = whisper.log_mel_spectrogram(audio)

        text = [*self.tokenizer.sot_sequence_including_notimestamps] + self.tokenizer.encode(text)
        target_tokens = text[1:] + [self.tokenizer.eot]

        return {
            "mel": mel,
            "target_tokens": target_tokens,
            "decoder_input_tokens": text
        }

##
# datamodule = EMGDataModule(data_dir, togglePhones, normalizers_file, max_len=max_len,
#                            DatasetClass=SpectrogramDataset)
# td = datamodule.train_dataloader()
# # longest_emg = 0
# longest_emg = []
# for bat in tqdm(td):
#     # longest_emg = max(longest_emg, np.quantile([e.shape[0] for e in bat['raw_emg']], 0.99))
# #     longest_emg = max(longest_emg, np.max([e.shape[0] for e in bat['raw_emg']]))
#     longest_emg.append(np.quantile([e.shape[0] for e in bat['raw_emg']], 0.5))
    
# # assert longest_emg * 8 == 299200
# # print(longest_emg * 8) # approx 235390, so 99-percentile length is 29s
# print(np.mean(longest_emg) * 8) # approx 235390, so 99-percentile length is 29s
##
def whisper_data_collator_with_padding(features, eot_token_id=wtokenizer.eot):
        mels, target_tokens, decoder_input_tokens = [], [], []
        for f in features:
            mels.append(f["mel"])
            target_tokens.append(f["target_tokens"])
            decoder_input_tokens.append(f["decoder_input_tokens"])

        mels = torch.concat([mel[None, :] for mel in mels])
        
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

        batch = {k: torch.tensor(np.array(v), requires_grad=False) for k, v in batch.items()}
        batch["mel"] = mels

        return batch
    
class WhisperConfig:
    learning_rate = 0.0005
    weight_decay = 0.01
    adam_epsilon = 1e-8
    warmup_steps = 2
    batch_size = 16
    num_worker = 0
    num_train_epochs = 10
    gradient_accumulation_steps = 1
    sample_rate = 16000

class WhisperModelModule(pl.LightningModule):
    def __init__(self, cfg:WhisperConfig, model_name="base", lang="en", train_dataset=[], eval_dataset=[]) -> None:
        super().__init__()
        self.options = whisper.DecodingOptions(language=lang, without_timestamps=True)
        self.model = whisper.load_model(model_name)
        self.tokenizer = whisper.tokenizer.get_tokenizer(True, language=lang, task=self.options.task)

        # only decoder training
        # for p in self.model.encoder.parameters():
        #     p.requires_grad = False
        
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.metrics_wer = evaluate.load("wer")
        self.metrics_cer = evaluate.load("cer")

        self.cfg = cfg
        self.__train_dataset = train_dataset
        self.__eval_dataset = eval_dataset
    
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_id):
        mels = batch["mels"]
        target_tokens = batch["target_tokens"].long()
        decoder_input_tokens = batch["decoder_input_tokens"].long()

        with torch.no_grad():
            audio_features = self.model.encoder(mels)

        out = self.model.decoder(decoder_input_tokens, audio_features)
        loss = self.loss_fn(out.view(-1, out.size(-1)), target_tokens.view(-1))
        self.log("train/loss", loss, on_step=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_id):
        mels = batch["mels"]
        target_tokens = batch["target_tokens"].long()
        decoder_input_tokens = batch["decoder_input_tokens"].long()


        audio_features = self.model.encoder(mels)
        out = self.model.decoder(decoder_input_tokens, audio_features)

        loss = self.loss_fn(out.view(-1, out.size(-1)), target_tokens.view(-1))

        out[out == -100] = self.tokenizer.eot
        target_tokens[target_tokens == -100] = self.tokenizer.eot

        o_list, l_list = [], []
        for o, l in zip(out, target_tokens):
            o = torch.argmax(o, dim=1)
            o_list.append(self.tokenizer.decode(o))
            l_list.append(self.tokenizer.decode(l))
        cer = self.metrics_cer.compute(references=l_list, predictions=o_list)
        wer = self.metrics_wer.compute(references=l_list, predictions=o_list)

        self.log("val/loss", loss, on_step=True, prog_bar=True, logger=True)
        self.log("val/cer", cer, on_step=True, prog_bar=True, logger=True)
        self.log("val/wer", wer, on_step=True, prog_bar=True, logger=True)

        return {
            "cer": cer,
            "wer": wer,
            "loss": loss
        }

    def configure_optimizers(self):
        """Create optimizers and schedulers."""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() 
                            if not any(nd in n for nd in no_decay)],
                "weight_decay": self.cfg.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() 
                            if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, 
                          lr=self.cfg.learning_rate, 
                          eps=self.cfg.adam_epsilon)
        self.optimizer = optimizer

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.cfg.warmup_steps, 
            num_training_steps=self.t_total
        )
        self.scheduler = scheduler

        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]
    
    def setup(self, stage=None):
        """Adjust linear scheduler according to number of training steps.
        
        not sure if setup is right place for this; copied code from Japanese notebook (discussion 64)
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#setup"""

        if stage == 'fit' or stage is None:
            self.t_total = (
                (len(self.__train_dataset) // (self.cfg.batch_size))
                // self.cfg.gradient_accumulation_steps
                * float(self.cfg.num_train_epochs)
            )

SD = partial(SpectrogramDataset, tokenizer=wtokenizer)
config = WhisperConfig()
datamodule = EMGDataModule(data_dir, togglePhones, normalizers_file, max_len=max_len,
                           batch_size = config.batch_size, num_workers=config.num_worker,
                           drop_last=True, shuffle=True,
                           batch_sampler=False, collate_fn=whisper_data_collator_with_padding,
                           DatasetClass=SD)
##
# verify dataloader is working
td = datamodule.train_dataloader()
for b in tqdm(td):
    print(b.keys())
    print(b["target_tokens"].shape)
    print(b["mel"].shape)
    print(b["decoder_input_tokens"].shape)

    for token, dec in zip(b["target_tokens"], b["decoder_input_tokens"]):
        token[token == -100] = wtokenizer.eot
        # text = wtokenizer.decode(token)
        text = wtokenizer.decode(token)
        print(text)

        dec[dec == -100] = wtokenizer.eot
        # text = wtokenizer.decode(dec)
        text = wtokenizer.decode(dec)
        print(text)
    
    break
##
_re_special = re.compile(r"\<\|.+?\|\>")
def strip_special_tokens(string):
    return re.sub(_re_special, "", string)
    
# wmodel = whisper.load_model("large")
whisper_model = WhisperModelModule(config,
                                #    model_name="base")
                                   model_name="medium")
                                #    model_name="large")
with torch.no_grad():
    audio_features = whisper_model.model.encoder(b["mel"].cuda())
    mel = b["mel"]
    target_tokens = b["target_tokens"].long()
    decoder_input_tokens = b["decoder_input_tokens"].long()

        
    audio_features = whisper_model.model.encoder(mel.cuda())
    print(decoder_input_tokens)
    print(mel.shape, decoder_input_tokens.shape, audio_features.shape)
    print(audio_features.shape)
    out = whisper_model.model.decoder(decoder_input_tokens.cuda(), audio_features)
    pred_tokens = torch.argmax(out, dim=2)
    for pred,true in zip(pred_tokens,target_tokens):
        pred[pred == -100] = wtokenizer.eot
        pred_text = wtokenizer.decode(pred)
        true_text = wtokenizer.decode(true)
        print(f"=============================")
        print("Pred: ", pred_text)
        print("Actual: ", true_text)
##

o_list, l_list = [], []
for o, l in zip(out, target_tokens):
    o = torch.argmax(o, dim=1)
    o_list.append(strip_special_tokens(wtokenizer.decode(o)))
    l_list.append(strip_special_tokens(wtokenizer.decode(l)))
    
wer = whisper_model.metrics_wer.compute(references=l_list, predictions=o_list)
wer

##
o_list, l_list = [], []
n = 0
with torch.no_grad():
    for b in tqdm(td):
        mel = b["mel"]
        target_tokens = b["target_tokens"].long()
        decoder_input_tokens = b["decoder_input_tokens"].long()
            
        audio_features = whisper_model.model.encoder(mel.cuda())
        out = whisper_model.model.decoder(decoder_input_tokens.cuda(), audio_features)
        
        target_tokens[target_tokens == -100] = wtokenizer.eot
        out[out == -100] = wtokenizer.eot
        
        for o, l in zip(out, target_tokens):
            o = torch.argmax(o, dim=1)
            o_list.append(strip_special_tokens(wtokenizer.decode(o)))
            l_list.append(strip_special_tokens(wtokenizer.decode(l)))
        n+=1
        if n > 50:
            break

wer = whisper_model.metrics_wer.compute(references=l_list, predictions=o_list)
wer

##
