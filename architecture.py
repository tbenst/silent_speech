import random
import torch, numpy as np
from torch import nn
import torch.nn.functional as F
from transformer import TransformerEncoderLayer
from data_utils import combine_fixed_length, decollate_tensor

import sys, os, jiwer
import pytorch_lightning as pl, torchmetrics
from torchaudio.models.decoder import ctc_decoder
from torchaudio.functional import edit_distance
# from s4 import S4
from data_utils import TextTransform, token_error_rate
from magneto.models.hyena import HyenaOperator
from flash_attn.modules.block import Block
from magneto.models.s4d import S4D

from pytorch_lightning.profilers import PassThroughProfiler
from dataclasses import dataclass
from typing import Tuple, List, Union
from dataloaders import split_batch_into_emg_neural_audio
from contrastive import (
    nobatch_cross_contrastive_loss,
    supervised_contrastive_loss,
    SupConLoss,
)
from typing import Tuple
from pytorch_lightning.loggers import NeptuneLogger
from align import align_from_distances
from torch.optim.lr_scheduler import LambdaLR

from collections import defaultdict
from warnings import warn

import gc
import logging

MODEL_SIZE = 768  # number of hidden dimensions
NUM_LAYERS = 6  # number of layers
DROPOUT = 0.2  # dropout


def layer_norm(
    x: torch.Tensor, dim: Tuple[int] = None, eps: float = 0.00001
) -> torch.Tensor:
    """
    Layer normalization as described in https://arxiv.org/pdf/1607.06450.pdf.

    Supports inputs of any shape, where first dimension is the batch. Does not
    apply elementwise affine transformation.

    https://stackoverflow.com/questions/59830168/layer-normalization-in-pytorch
    """
    if dim is None:
        # all except batch
        dim = tuple(range(1, len(x.shape)))
    mean = torch.mean(x, dim=dim, keepdim=True)
    var = torch.var(x, dim=dim, keepdim=True, correction=0)
    return (x - mean) / torch.sqrt(var + eps)


class LayerNorm(nn.Module):
    def __init__(self, dim: Tuple[int] = None, eps: float = 0.00001):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        return layer_norm(x, dim=self.dim, eps=self.eps)


class ResBlock(nn.Module):
    def __init__(
        self, num_ins, num_outs, stride=1, pre_activation=False, beta: float = 1.0
    ):
        super().__init__()

        self.conv1 = nn.Conv1d(num_ins, num_outs, 3, padding=1, stride=stride)
        self.norm1 = nn.BatchNorm1d(num_outs)
        self.conv2 = nn.Conv1d(num_outs, num_outs, 3, padding=1)
        self.norm2 = nn.BatchNorm1d(num_outs)
        # self.act = nn.ReLU()
        self.act = nn.GELU()  # TODO: test which is better
        self.beta = beta

        if stride != 1 or num_ins != num_outs:
            self.residual_path = nn.Conv1d(num_ins, num_outs, 1, stride=stride)
            self.res_norm = nn.BatchNorm1d(num_outs)
            if pre_activation:
                self.skip = nn.Sequential(self.res_norm, self.residual_path)
            else:
                self.skip = nn.Sequential(self.residual_path, self.res_norm)
        else:
            self.skip = nn.Identity()

        # ResNet v2 style pre-activation https://arxiv.org/pdf/1603.05027.pdf
        self.pre_activation = pre_activation

        if pre_activation:
            self.block = nn.Sequential(
                self.norm1, self.act, self.conv1, self.norm2, self.act, self.conv2
            )
        else:
            self.block = nn.Sequential(
                self.conv1, self.norm1, self.act, self.conv2, self.norm2
            )

    def forward(self, x):
        # logging.warning(f"ResBlock forward pass. x.shape: {x.shape}")
        res = self.block(x) * self.beta
        x = self.skip(x)

        if self.pre_activation:
            return x + res
        else:
            return self.act(x + res)


@dataclass
class XtoTextConfig:
    steps_per_epoch: int
    lm_directory: str
    togglePhones: bool = True
    learning_rate_warmup: int = 500  # not used, todo refactor to MONA/Gaddy only
    weight_decay: float = 1e-5
    learning_rate: float = 0.01
    gradient_accumulation_steps: int = 1
    num_train_epochs: int = 200
    precision: Union[int, str] = 32


class XtoText(pl.LightningModule):
    "Base model for all (neural, audio, emg, X) to text models."

    def __init__(self, cfg, text_transform: TextTransform):
        super().__init__()
        self.text_transform = text_transform
        self.n_chars = len(text_transform.chars)
        self.lm_directory = cfg.lm_directory
        self.weight_decay = cfg.weight_decay
        self.lr = cfg.learning_rate
        self.target_lr = cfg.learning_rate  # will not mutate
        self.learning_rate_warmup = cfg.learning_rate_warmup
        self.steps_per_epoch = cfg.steps_per_epoch

        if cfg.togglePhones:
            self.lexicon_file = os.path.join(cfg.lm_directory, "cmudict.txt")
        else:
            self.lexicon_file = os.path.join(
                cfg.lm_directory, "lexicon_graphemes_noApostrophe.txt"
            )

        self.step_vocal_emg_text_target = []
        self.step_vocal_emg_text_pred = []
        self.step_vocal_emg_int_target = []
        self.step_vocal_emg_int_pred = []

        self.step_silent_emg_text_target = []
        self.step_silent_emg_text_pred = []
        self.step_silent_emg_int_target = []
        self.step_silent_emg_int_pred = []

    def _init_ctc_decoder(self):
        self.ctc_decoder = ctc_decoder(
            lexicon=self.lexicon_file,
            # tokens      = [x.lower() for x in self.text_transform.chars] + ['_'],
            tokens=self.text_transform.chars + ["_"],
            # lm      = os.path.join(self.lm_directory, '4gram_lm.bin'),
            lm=os.path.join(self.lm_directory, "lm.binary"),
            blank_token="_",
            sil_token="|",
            nbest=1,
            lm_weight=2,  # default is 2; Gaddy sets to 1.85
            # word_score  = -3,
            # sil_score   = -2,
            beam_size=150,  # SET TO 150 during inference
        )

    def ctc_loss(self, pred, target, pred_len, target_len):
        # this pads with 0, which corresponds to 'a', but by passing target_len
        # to CTC loss we can ignore these padded values
        pred = nn.utils.rnn.pad_sequence(
            pred, batch_first=False
        )  # B x T x C -> T x B x C, as required by ctc
        # pred = nn.utils.rnn.pad_sequence(decollate_tensor(pred, pred_len), batch_first=False)
        target = nn.utils.rnn.pad_sequence(target, batch_first=True)
        # print(f"\n ==== CTC ====\n{pred.shape=}, {target.shape=}\n{pred=}\n{target=}\n")
        # print(f"{pred.shape=}, {target[0].shape=}, {pred_len=}, {target_len=}")
        # print(f"ctc_loss: {[p.shape for p in pred]=}, {[t.shape for t in target]=}")
        loss = F.ctc_loss(
            pred, target, pred_len, target_len, blank=self.n_chars, zero_infinity=True
        )
        return loss

    def on_train_epoch_start(self):
        # bad separation of concerns / composability,
        # but this seems forced by pytorch lightning
        # maybe should use Fabric in the future..
        if self.trainer.datamodule is not None:
            try:
                self.trainer.datamodule.TrainBatchSampler.set_epoch(self.current_epoch)
                logging.debug(f"set epoch to {self.current_epoch=}")
            except:
                # not all datamodules have a TrainBatchSampler, or a set_epoch method
                pass
            
        # print(f"==== DEBUG: {self.lr_schedulers().state_dict()=} ====")

    def training_step(self, batch, batch_idx):
        c = self.calc_loss(**self.forward(batch))
        loss = c["loss"]
        emg_bz = c["emg_bz"] if "emg_bz" in c else 0
        neural_bz = c["neural_bz"] if "neural_bz" in c else 0
        audio_bz = c["audio_bz"] if "audio_bz" in c else 0
        cross_con_bz = c["cross_con_bz"] if "cross_con_bz" in c else 0
        summed_bz = emg_bz + neural_bz + audio_bz

        self.maybe_log(
            "train/loss",
            c,
            "loss",
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
            batch_size=summed_bz,
            sync_dist=True,
        )
        self.maybe_log(
            "train/emg_ctc_loss",
            c,
            "emg_ctc_loss",
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=False,
            batch_size=emg_bz,
            sync_dist=True,
        )
        self.maybe_log(
            "train/neural_ctc_loss",
            c,
            "neural_ctc_loss",
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=False,
            batch_size=neural_bz,
            sync_dist=True,
        )
        self.maybe_log(
            "train/audio_ctc_loss",
            c,
            "audio_ctc_loss",
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=False,
            batch_size=audio_bz,
            sync_dist=True,
        )
        self.maybe_log(
            "train/cross_contrastive_loss",
            c,
            "cross_contrastive_loss",
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=False,
            batch_size=cross_con_bz,
            sync_dist=True,
        )
        self.maybe_log(
            "train/supervised_contrastive_loss",
            c,
            "supervised_contrastive_loss",
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=False,
            # same batch size as cross_con_bz as implemented in MONA calc_loss
            # but not generally true
            batch_size=cross_con_bz,
            sync_dist=True,
        )
        self.maybe_log(
            "train/avg_emg_latent",
            c,
            "emg_z_mean",
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=False,
            batch_size=emg_bz,
            sync_dist=True,
        )
        self.maybe_log(
            "train/avg_audio_latent",
            c,
            "audio_z_mean",
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=False,
            batch_size=audio_bz,
            sync_dist=True,
        )
        self.maybe_log(
            "train/avg_neural_latent",
            c,
            "neural_z_mean",
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=False,
            batch_size=neural_bz,
            sync_dist=True,
        )
        torch.cuda.empty_cache()
        return loss

    def on_validation_epoch_start(self):
        # self.profiler.start(f"validation loop")
        self._init_ctc_decoder()

    def validation_step(self, batch, batch_idx, task="val"):
        ret = self.forward(batch, fixed_length=False)
        # supTcon will fail for silent-only data
        c = self.calc_loss(**ret, use_supTcon=False, use_crossCon=False, use_dtw=False)
        warn("only using EMG data for validation")
        pred = ret["emg_pred"]
        loss = c["loss"]
        emg_bz = c["emg_bz"] if "emg_bz" in c else 0
        neural_bz = c["neural_bz"] if "neural_bz" in c else 0
        audio_bz = c["audio_bz"] if "audio_bz" in c else 0
        paired_bz = c["paired_bz"] if "paired_bz" in c else 0
        summed_bz = emg_bz + neural_bz + audio_bz + paired_bz

        silent_emg_idx = ret["silent_emg_idx"]
        parallel_emg_idx = ret["parallel_emg_idx"]
        target_ints = ret["y_emg"]
        batch_text = ret["text_emg"]

        # logging.debug(f"{silent_emg_idx=}, {parallel_emg_idx=}, {emg_bz=}")
        assert (
            len(silent_emg_idx) + len(parallel_emg_idx) == emg_bz
        ), f"Expeceted all examples to be silent or parallel EMG, but got other examples, too"

        is_silent = []
        for i in range(emg_bz):
            if i in silent_emg_idx:
                is_silent.append(True)
            elif i in parallel_emg_idx:
                is_silent.append(False)
            else:
                raise ValueError(
                    "Expected all examples to be silent or parallel EMG, but got other examples, too"
                )

        # TODO: split text by emg, audio, neural
        # TODO: think through if 'text' is being matched correctly to silent emg
        pred_texts, pred_ints = self._beam_search_pred(pred.cpu())
        pred_texts = [self.text_transform.clean_text(b) for b in pred_texts]
        target_texts = [self.text_transform.clean_text(b) for b in batch_text]
        # print(f"text: {batch['text']}; target_text: {target_text}; pred_text: {pred_text}")

        # sanity check lengths
        lens = [
            len(t)
            for t in [target_texts, pred_texts, pred_ints, target_ints, is_silent]
        ]
        # check all equal
        assert len(set(lens)) == 1, f"all lengths should be equal {lens=}"

        for i, (target_text, pred_text, target_int, pred_int, is_s) in enumerate(
            zip(target_texts, pred_texts, target_ints, pred_ints, is_silent)
        ):
            if len(target_text) > 0:
                if is_s:
                    stt = self.step_silent_emg_text_target
                    stp = self.step_silent_emg_text_pred
                    sit = self.step_silent_emg_int_target
                    sip = self.step_silent_emg_int_pred
                    if i % 16 == 0 and type(self.logger) == NeptuneLogger:
                        # log approx 10 examples
                        self.logger.experiment[
                            f"training/{task}/silent_emg_sentence_target"
                        ].append(target_text)
                        self.logger.experiment[
                            f"training/{task}/silent_emg_sentence_pred"
                        ].append(pred_text)

                else:
                    stt = self.step_vocal_emg_text_target
                    stp = self.step_vocal_emg_text_pred
                    sit = self.step_vocal_emg_int_target
                    sip = self.step_vocal_emg_int_pred
                    if i % 16 == 0 and type(self.logger) == NeptuneLogger:
                        self.logger.experiment[
                            f"training/{task}/vocal_emg_sentence_target"
                        ].append(target_text)
                        self.logger.experiment[
                            f"training/{task}/vocal_emg_sentence_pred"
                        ].append(pred_text)

                stt.append(target_text)
                stp.append(pred_text)

                sit.append(target_int.cpu().numpy())
                sip.append(pred_int)

        self.maybe_log(
            f"{task}/loss",
            c,
            "loss",
            prog_bar=True,
            batch_size=summed_bz,
            sync_dist=True,
        )
        self.maybe_log(
            f"{task}/emg_ctc_loss",
            c,
            "emg_ctc_loss",
            prog_bar=False,
            batch_size=emg_bz,
            sync_dist=True,
        )
        self.maybe_log(
            f"{task}/neural_ctc_loss",
            c,
            "neural_ctc_loss",
            prog_bar=False,
            batch_size=neural_bz,
            sync_dist=True,
        )
        self.maybe_log(
            f"{task}/audio_ctc_loss",
            c,
            "audio_ctc_loss",
            prog_bar=False,
            batch_size=audio_bz,
            sync_dist=True,
        )

        return loss

    def _on_validation_epoch_end(
        self, text_target, text_pred, int_target, int_pred
    ) -> None:
        "Helper function for vocal & silent emg."
        # TODO: this may not be implemented correctly for DDP
        # raise NotImplementedError("on_validation_epoch_end not implemented neural, librispeech")
        # logging.warning(f"start on_validation_epoch_end")
        nonzero_text_target = []
        nonzero_text_pred = []
        nonzero_int_target = []
        nonzero_int_pred = []
        for t, p, i, j in zip(text_target, text_pred, int_target, int_pred):
            if len(t) > 0:
                nonzero_text_target.append(t)
                nonzero_text_pred.append(p)
                nonzero_int_target.append(i)
                nonzero_int_pred.append(j)
            else:
                print("WARN: got target length of zero during validation.")
            if len(p) == 0:
                logging.debug("WARN: got prediction length of zero during validation.")
        # logging.warning(f"on_validation_epoch_end: calc wer")
        wer = jiwer.wer(nonzero_text_target, nonzero_text_pred)
        # print(f"{nonzero_text_target=}, {nonzero_text_pred=}")
        # print(f"WER: {wer}")
        # print(f"{nonzero_int_target=}, {nonzero_int_pred=}")
        try:
            # CER not fully debugged yet
            cer = token_error_rate(
                nonzero_int_target, nonzero_int_pred, self.text_transform
            )
        except:
            cer = None
        text_target.clear()
        text_pred.clear()
        int_target.clear()
        int_pred.clear()
        return wer, cer

        # self.profiler.stop(f"validation loop")
        # self.profiler.describe()
        # logging.warning(f"on_validation_epoch_end: gc.collect()")
        gc.collect()
        torch.cuda.empty_cache()  # TODO: see if fixes occasional freeze...?

    def on_validation_epoch_end(self) -> None:
        vocal_wer, vocal_cer = self._on_validation_epoch_end(
            self.step_vocal_emg_text_target,
            self.step_vocal_emg_text_pred,
            self.step_vocal_emg_int_target,
            self.step_vocal_emg_int_pred,
        )

        self.log("val/vocal_emg_wer", vocal_wer, prog_bar=True, sync_dist=True)
        self.log("val/vocal_emg_cer", vocal_cer, prog_bar=True, sync_dist=True)

        silent_wer, silent_cer = self._on_validation_epoch_end(
            self.step_silent_emg_text_target,
            self.step_silent_emg_text_pred,
            self.step_silent_emg_int_target,
            self.step_silent_emg_int_pred,
        )
        self.log("val/silent_emg_wer", silent_wer, prog_bar=True, sync_dist=True)
        self.log("val/silent_emg_cer", silent_cer, prog_bar=True, sync_dist=True)

        # log for backwards compatibility / easy comparison with old results
        self.log("val/wer", silent_wer, prog_bar=True, sync_dist=True)
        self.log("val/cer", silent_cer, prog_bar=True, sync_dist=True)

        gc.collect()
        torch.cuda.empty_cache()  # TODO: see if fixes occasional freeze...?

    def _beam_search_pred(self, pred):
        "Repeatedly called by validation_step & test_step."
        beam_results = self.ctc_decoder(pred)
        pred_text = []
        pred_int = []
        for b in beam_results:
            if len(b) > 0:
                # I think length is zero only when there's NaNs in the output
                # we could just allow the crash here
                pred_text.append(" ".join(b[0].words).strip().lower())
                pred_int.append(b[0].tokens)
            else:
                pred_text.append("")
                pred_int.append([])

        return pred_text, pred_int

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx, task="test")

    def on_test_epoch_end(self) -> None:
        wer = jiwer.wer(self.step_text_target, self.step_text_pred)
        self.step_text_target.clear()
        self.step_text_pred.clear()
        self.log("test/wer", wer, prog_bar=True)

    def maybe_log(self, name, my_dict, key, **kwargs):
        if key in my_dict and not my_dict[key] is None:
            self.log(name, my_dict[key], **kwargs)

    def log(self, *args, **kwargs):
        try:
            isnan = np.isnan(args[0])
        except:
            isnan = False

        if "batch_size" in kwargs and kwargs["batch_size"] == 0:
            pass
        elif args[0] is None:
            pass
        elif isnan:
            logging.warning(f"got nan in log: {args=}, {kwargs=}")
            pass
        else:
            super().log(*args, **kwargs)


class GaddyBase(XtoText):
    def configure_optimizers(self):
        initial_lr = self.target_lr / self.learning_rate_warmup

        # for FSDP
        optimizer = torch.optim.AdamW(
            self.trainer.model.parameters(),
            lr=initial_lr,
            weight_decay=self.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[
                125 * self.steps_per_epoch,
                150 * self.steps_per_epoch,
                175 * self.steps_per_epoch,
            ],
            gamma=0.5,
        )
        lr_scheduler = {"scheduler": scheduler, "interval": "step"}

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def set_lr(self, new_lr):
        optimizer = self.optimizers().optimizer
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr

    def lr_scheduler_step(self, scheduler, metric):
        # warmup per Gaddy

        # print(f"lr_scheduler_step: {self.global_step=}")
        # optimizer = self.optimizers().optimizer
        # for param_group in optimizer.param_groups:
        #     print(f"lr: {param_group['lr']}")
        if metric is None:
            scheduler.step()
        else:
            scheduler.step(metric)

        # TODO:  switch to a new (proper) scheduler that supports
        # linear warmup and gamma decay

        # linear warmup
        if self.global_step <= self.learning_rate_warmup:
            new_lr = self.global_step * self.target_lr / self.learning_rate_warmup
            self.set_lr(new_lr)


class Model(GaddyBase):
    def __init__(
        self,
        model_size,
        dropout,
        num_layers,
        num_outs,
        text_transform: TextTransform,
        steps_per_epoch,
        epochs,
        lm_directory,
        num_aux_outs=None,
        lr=3e-4,
        learning_rate_warmup=1000,
        profiler=None,
        weight_decay=0.0,
    ):
        super().__init__(text_transform, lm_directory)
        self.profiler = profiler or PassThroughProfiler()
        self.conv_blocks = nn.Sequential(
            ResBlock(8, model_size, 2),
            ResBlock(model_size, model_size, 2),
            ResBlock(model_size, model_size, 2),
        )
        self.w_raw_in = nn.Linear(model_size, model_size)
        encoder_layer = TransformerEncoderLayer(
            d_model=model_size,
            nhead=8,
            relative_positional=True,
            relative_positional_distance=100,
            dim_feedforward=3072,
            dropout=dropout,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.w_out = nn.Linear(model_size, num_outs)

        self.has_aux_out = num_aux_outs is not None
        if self.has_aux_out:
            self.w_aux = nn.Linear(model_size, num_aux_outs)

        self.seqlen = 600
        self.lr = lr
        self.target_lr = lr  # will not mutate
        self.learning_rate_warmup = learning_rate_warmup
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch

        # val/test procedure...

        self._init_ctc_decoder()

        self.step_text_target = []
        self.step_text_pred = []
        self.step_int_target = []
        self.step_int_pred = []
        self.weight_decay = weight_decay

    def forward(self, x_raw):
        # x shape is (batch, time, electrode)

        if self.training:
            r = random.randrange(8)
            if r > 0:
                x_raw[:, :-r, :] = x_raw[:, r:, :]  # shift left r
                x_raw[:, -r:, :] = 0

        x_raw = x_raw.transpose(1, 2)  # put channel before time for conv
        # print(f"before conv: {x_raw.shape=}")
        x_raw = self.conv_blocks(x_raw)
        # print(f"after conv: {x_raw.shape=}")
        x_raw = x_raw.transpose(1, 2)
        x_raw = self.w_raw_in(x_raw)

        x = x_raw
        x = x.transpose(0, 1)  # put time first
        # print(f"before transformer: {x.shape=}")
        x = self.transformer(x)
        x = x.transpose(0, 1)

        if self.has_aux_out:
            aux_out = self.w_aux(x)

        x = F.log_softmax(self.w_out(x), -1)
        if self.has_aux_out:
            return x, aux_out
        else:
            return x
        # before conv: x_raw.shape=torch.Size([4, 8, 4800])
        # after conv: x_raw.shape=torch.Size([4, 768, 600])
        # before transformer: x.shape=torch.Size([600, 4, 768])
        # after w_out: x.shape=torch.Size([4, 600, 38])

        # before conv: x_raw.shape=torch.Size([1, 8, 14568])
        # after conv: x_raw.shape=torch.Size([1, 768, 1821])
        # before transformer: x.shape=torch.Size([1821, 1, 768])
        # after w_out: x.shape=torch.Size([1, 1821, 38])

        # before conv: x_raw.shape=torch.Size([1, 8, 4800])
        # after conv: x_raw.shape=torch.Size([1, 768, 600])
        # before transformer: x.shape=torch.Size([600, 1, 768])
        # after w_out: x.shape=torch.Size([1, 600, 38])

        # before conv: x_raw.shape=torch.Size([1, 8, 2776])
        # after conv: x_raw.shape=torch.Size([1, 768, 347])
        # before transformer: x.shape=torch.Size([347, 1, 768])
        # after w_out: x.shape=torch.Size([1, 347, 38])

    def calc_loss(self, batch):
        X = combine_fixed_length(batch["emg"], self.seqlen)
        X_raw = combine_fixed_length(batch["raw_emg"], self.seqlen * 8)
        bz = X.shape[0]

        pred = self(X_raw)

        # seq first, as required by ctc
        pred = nn.utils.rnn.pad_sequence(
            decollate_tensor(pred, batch["lengths"]), batch_first=False
        )
        y = nn.utils.rnn.pad_sequence(batch["text_int"], batch_first=True)
        loss = F.ctc_loss(
            pred, y, batch["lengths"], batch["text_int_lengths"], blank=self.n_chars
        )

        if torch.isnan(loss) or torch.isinf(loss):
            # print('batch:', batch_idx)
            print("Isnan output:", torch.any(torch.isnan(pred)))
            print("Isinf output:", torch.any(torch.isinf(pred)))
            # raise ValueError("NaN/Inf detected in loss")

        return loss, bz

    def _beam_search_step(self, batch):
        "Repeatedly called by validation_step & test_step. Impure function!"
        X_raw = batch["raw_emg"][0].unsqueeze(0)

        pred = self(X_raw).cpu()

        beam_results = self.ctc_decoder(pred)
        pred_text = " ".join(beam_results[0][0].words).strip().lower()
        b0 = batch["text"][0]
        if len(batch["text"][0]) == 1:
            # index twice for gaddy's collate function
            target_text = self.text_transform.clean_text(b0[0])
        else:
            # Only index once for new collate function
            target_text = self.text_transform.clean_text(b0)

        return target_text, pred_text

    def training_step(self, batch, batch_idx):
        loss, bz = self.calc_loss(batch)
        self.log(
            "train/loss",
            loss,
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
            batch_size=bz,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, bz = self.calc_loss(batch)
        target_text, pred_text = self._beam_search_step(batch)
        assert (
            len(batch["emg"]) == 1
        ), "Currently only support batch size of 1 for validation"
        if len(target_text) > 0:
            self.step_text_target.append(target_text)
            self.step_text_pred.append(pred_text)

        self.log("val/loss", loss, prog_bar=True, batch_size=bz)
        return loss

    def on_validation_epoch_end(self) -> None:
        # TODO: this may not be implemented correctly for DDP
        logging.warning(f"start on_validation_epoch_end")
        step_text_target = []
        step_text_pred = []
        for t, p in zip(self.step_text_target, self.step_text_pred):
            if len(t) > 0:
                step_text_target.append(t)
                step_text_pred.append(p)
            else:
                print("WARN: got target length of zero during validation.")
            if len(p) == 0:
                logging.debug("WARN: got prediction length of zero during validation.")
        logging.warning(f"on_validation_epoch_end: calc wer")
        wer = jiwer.wer(step_text_target, step_text_pred)
        self.step_text_target.clear()
        self.step_text_pred.clear()
        self.log("val/wer", wer, prog_bar=True, sync_dist=True)
        # self.profiler.stop(f"validation loop")
        # self.profiler.describe()
        logging.warning(f"on_validation_epoch_end: gc.collect()")
        gc.collect()
        torch.cuda.empty_cache()  # TODO: see if fixes occasional freeze...?

    def test_step(self, batch, batch_idx):
        loss, bz = self.calc_loss(batch)
        target_text, pred_text = self._beam_search_step(batch)
        if len(target_text) > 0:
            self.step_text_target.append(target_text)
            self.step_text_pred.append(pred_text)
        self.log("test/loss", loss, prog_bar=True, batch_size=bz)
        return loss


class S4Layer(nn.Module):
    """
    https://github.com/HazyResearch/state-spaces/blob/ab287c63f4938a76d06a6b6868ee4a7163b50b05/example.py

    Abstraction layer that gives more fine-grained control over S4 design.
    This module has a S4Kernel, dropout, and layer norm.
    """

    def __init__(
        self, model_size, dropout, s4_dropout=None, diagonal=False, prenorm=False
    ):
        super().__init__()

        self.model_size = model_size
        self.s4_dropout = dropout if s4_dropout is None else s4_dropout

        if diagonal:
            self.s4_layer = S4(
                model_size,
                dropout=self.s4_dropout,
                bidirectional=True,
                transposed=True,
                lr=None,
                mode="diag",
                measure="diag-inv",
                disc="zoh",
                real_type="exp",
            )
        else:
            self.s4_layer = S4(
                model_size,
                dropout=self.s4_dropout,
                bidirectional=True,
                transposed=True,
                lr=None,
            )

        self.norm = nn.LayerNorm(model_size)
        self.dropout = nn.Dropout1d(dropout)
        # self.dropout  = nn.Dropout(dropout)
        self.prenorm = prenorm

    def forward(self, x):
        """
        Input x is list of tensors with shape (B, L, d_input)
        Returns tensor of same size.
        """

        x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
        z = x

        if self.prenorm:  # Prenorm
            z = self.norm(z.transpose(-1, -2)).transpose(-1, -2)

        # Apply S4 block: we ignore the state input and output
        z, _ = self.s4_layer(z)

        # Dropout on the output of the S4 block
        z = self.dropout(z)

        # Residual connection
        x = z + x

        if not self.prenorm:
            # Postnorm
            x = self.norm(x.transpose(-1, -2)).transpose(-1, -2)

        x = x.transpose(-1, -2)

        return x


class S4Model(nn.Module):
    def __init__(self, num_features, num_outs, num_aux_outs=None):
        super().__init__()
        self.prenorm = False
        self.diagonal = False

        # Linear encoder
        self.encoder = nn.Sequential(
            nn.Linear(8, MODEL_SIZE), nn.Softsign(), nn.Linear(8, MODEL_SIZE)
        )

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.linears = nn.ModuleList()
        for i in range(NUM_LAYERS):
            if i > 2:
                s4_dropout = DROPOUT
                #   # channels = 2
                # else:
                s4_dropout = 0
            #  #  channels = 3

            s4_dropout = DROPOUT

            dropout = DROPOUT
            self.s4_layers.append(S4Layer(MODEL_SIZE, dropout, s4_dropout=s4_dropout))

        self.w_out = nn.Linear(MODEL_SIZE, num_outs)

        self.has_aux_out = num_aux_outs is not None
        if self.has_aux_out:
            self.w_aux = nn.Linear(MODEL_SIZE, num_aux_outs)

    def forward(self, x_raw):
        # x shape is (batch, time, electrode)

        if self.training:
            r = random.randrange(8)
            if r > 0:
                x_raw[:, :-r, :] = x_raw[:, r:, :]  # shift left r
                x_raw[:, -r:, :] = 0

        x = self.encoder(x_raw)

        for i, layer in enumerate(self.s4_layers):
            x = layer(x)

            # if i == 2 or i == 4 or i == 6:
            #    x = x[:, ::2, :] # 8x downsampling
            if i <= 2:
                x = x[:, ::2, :]

        if self.has_aux_out:
            return self.w_out(x), self.w_aux(x)
        else:
            return self.w_out(x)


sys.path.append("/home/users/ghwilson/repos/safari/src/models/sequence/")
sys.path.append("/home/users/ghwilson/repos/safari/")
try:
    from h3 import H3
except:
    print("Could not import H3")


class H3Model(nn.Module):
    def __init__(self, num_features, num_outs, num_aux_outs=None):
        super().__init__()
        self.prenorm = False

        # Linear encoder
        self.encoder = nn.Linear(8, MODEL_SIZE)

        # Stack S4 layers as residual blocks
        self.h3_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.linears = nn.ModuleList()
        for i in range(NUM_LAYERS):
            self.h3_layers.append(H3(d_model=MODEL_SIZE, dropout=DROPOUT, lr=None))
            self.norms.append(nn.LayerNorm(MODEL_SIZE))
            self.dropouts.append(nn.Dropout1d(DROPOUT))

        self.w_out = nn.Linear(MODEL_SIZE, num_outs)

        self.has_aux_out = num_aux_outs is not None
        if self.has_aux_out:
            self.w_aux = nn.Linear(MODEL_SIZE, num_aux_outs)

    def forward(self, x_raw):
        # x shape is (batch, time, electrode)

        if self.training:
            r = random.randrange(8)
            if r > 0:
                x_raw[:, :-r, :] = x_raw[:, r:, :]  # shift left r
                x_raw[:, -r:, :] = 0

        x = self.encoder(x_raw)
        # x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
        for i, (layer, norm, dropout) in enumerate(
            zip(self.h3_layers, self.norms, self.dropouts)
        ):
            z = x
            if self.prenorm:
                # Prenorm
                z = norm(z)

            # Apply H3 block
            # print(z.shape)
            z = layer(z)
            # print('Passed layer', i)
            # print(z.shape)

            # Dropout on the output of the S4 block
            z = dropout(z)

            # Residual connection
            x = z + x

            if not self.prenorm:
                # Postnorm
                x = norm(x)

            if i < 3:
                x = x[:, ::2, :]

        # x = x.transpose(-1, -2)

        if self.has_aux_out:
            return self.w_out(x), self.w_aux(x)
        else:
            return self.w_out(x)


def is_str(s):
    return isinstance(s, str) or isinstance(s, np.str_) or isinstance(s, np.unicode_)


# TODO: refactor batching dispatch logic in forward, and replace neual/audio/emg
# encoder with a single Module that dispatches to the appropriate encoder
# we can rid ourselves of emg_encoder, audio_encoder, neural_encoder, etc.
class LinearDispatch(nn.Module):
    """Based on a class label, dispatch to a linear layer.

    Attributes:
        classes (List[str]): The list of classes for dispatching.
        layers (nn.ModuleDict): Dictionary of linear layers for each class.
    """

    def __init__(self, classes: List[str], in_features: int, out_features: int):
        """Initializes the LinearDispatch with given classes, features and output size.

        Args:
            classes (List[str]): List of classes.
            in_features (int): Number of features in the input.
            out_features (int): Number of outputs from the linear layer.
        """
        super().__init__()
        # module name can't contain "."
        self.classes = list(map(self.sanitize_name, classes))
        self.out_features = out_features
        self.layers = nn.ModuleDict(
            {c: nn.Linear(in_features, out_features) for c in self.classes}
        )

    def sanitize_name(self, c: str) -> str:
        """Sanitizes a class string to be a valid module name.

        Args:
            c (str): Class string.

        Returns:
            str: Sanitized class string.
        """
        assert c != "", "Class string cannot be empty"
        assert is_str(c), f"Class string must be a string but got {type(c)}"
        return c.replace(".", "_")

    def forward(self, x: torch.Tensor, classes: List[str]) -> torch.Tensor:
        """Splits the batch into class_batches, then apply the appropriate linear layer,
        then concatenate the results back together in the same order.

        Args:
            x (torch.Tensor): Input tensor of shape (batch x ... x features).
            classes (List[str]): A batch of class labels.

        Returns:
            torch.Tensor: Processed tensor.


        Usage:
            x = torch.ones(5, 8)
            sessions = ['arst', 'ad', 'arst', 'ad', 'wqf']
            layer = LinearDispatch(sessions, 8, 4)
            layer(x, sessions)
        """
        assert len(x) == len(classes), "Batch size must be the same for x and classes"
        class_batches = {c: [] for c in self.classes}

        for i, c in enumerate(classes):
            new_c = self.sanitize_name(c)
            if new_c not in self.classes:
                raise ValueError(f"Unexpected class {c}")
            class_batches[new_c].append(i)

        # Initialize the return tensor and populate it with the processed class outputs.
        out = torch.zeros(
            (*x.shape[:-1], self.out_features), dtype=x.dtype, device=x.device
        )

        for c, indices in class_batches.items():
            if indices:  # Only process if there are indices for this class
                batch_for_class = x[
                    indices
                ]  # This gets the corresponding rows for this class
                out[indices] = self.layers[c](batch_for_class)

        return out


@dataclass
class MONAConfig(XtoTextConfig):
    num_outs: int = -1  # TODO: how to make required..? how do dataclass compose?
    input_channels: int = 8
    learning_rate: float = 3e-4  # also sets initial s4 lr
    weight_decay: float = 0.01
    warmup_steps: int = None  # warmup by backward steps
    batch_size: int = 12  # not necessary depending on batch sampler
    num_workers: int = 0
    num_train_epochs: int = 200
    gradient_accumulation_steps: int = 1
    sample_rate: int = 16000
    precision: str = "16-mixed"
    seqlen: int = 600
    attn_layers: int = 6
    d_model: int = 768  # original Gaddy
    # https://iclr-blog-track.github.io/2022/03/25/unnormalized-resnets/#balduzzi17shattered
    beta: float = 1 / np.sqrt(2)  # adjust resnet initialization
    neural_input_features: int = 1280
    neural_reduced_features: int = 768  # reduce 1280 down to this features

    cross_nce_lambda: float = 1.0  # how much to weight the latent loss
    audio_lambda: float = 1.0  # how much to weight the audio->text loss
    emg_lambda: float = 1.0  # how much to weight the emg->text loss
    neural_lambda: float = 1.0  # how much to weight the neural->text loss
    sup_nce_lambda: float = 0.1

    d_inner: int = 3072  # original Gaddy
    prenorm: bool = False
    dropout: float = 0.2
    in_channels: int = 8
    out_channels: int = 80
    resid_dropout: float = 0.0
    max_len: int = 480000
    num_heads: int = 8
    fixed_length: bool = False  # gaddy style fixed length of combining examples
    constant_offset_sd: float = 1.0
    white_noise_sd: float = 0.2

    togglePhones: bool = False
    use_dtw: bool = True
    use_crossCon: bool = True
    use_supTcon: bool = True
    batch_class_proportions: np.ndarray = np.array([0.16, 0.42, 0.42])
    latent_affine: bool = False  # so emg&audio latent are both unit norm

    def __post_init__(self):
        if self.warmup_steps is None:
            self.warmup_steps = 1000 // self.gradient_accumulation_steps


class MONA(GaddyBase):
    "Multimodal Orofacial Neural Audio"

    def __init__(
        self,
        cfg: MONAConfig,
        text_transform: TextTransform,
        profiler=None,
        # TODO: better if the no* and sessions are in cfg for easier loading
        no_emg=False,
        no_audio=False,
        sessions: List[str] = None,
        no_neural=False,
    ):
        super().__init__(cfg, text_transform)
        self.profiler = profiler or PassThroughProfiler()

        if sessions is not None:
            self.use_session_input_encoder = True
            self.session_input_encoder = LinearDispatch(
                sessions, cfg.neural_input_features, cfg.neural_reduced_features
            )
        else:
            self.use_session_input_encoder = False
            # self.neural_input_encoder = nn.Linear(cfg.neural_input_features,
            #                                       cfg.neural_reduced_features)

        # self.neural_input_dropout = nn.Dropout(0.4)
        # self.neural_input_act = nn.Softsign()

        if not no_emg:
            self.emg_conv_blocks = nn.Sequential(
                ResBlock(
                    cfg.input_channels,
                    cfg.d_model,
                    2,
                    pre_activation=False,
                    beta=cfg.beta,
                ),
                ResBlock(
                    cfg.d_model,
                    cfg.d_model,
                    2,
                    pre_activation=False,
                    beta=cfg.beta**2,
                ),
                ResBlock(
                    cfg.d_model,
                    cfg.d_model,
                    2,
                    pre_activation=False,
                    beta=cfg.beta**3,
                ),
            )
        if not no_audio:
            self.audio_conv_blocks = nn.Sequential(
                ResBlock(
                    80, cfg.d_model, beta=cfg.beta
                ),  # 80 mel freq cepstrum coefficients
                ResBlock(cfg.d_model, cfg.d_model, beta=cfg.beta**2),
                ResBlock(cfg.d_model, cfg.d_model, beta=cfg.beta**3),
            )
        if not no_neural:
            self.neural_conv_blocks = nn.Sequential(
                # TODO: should we do a 2D conv here..? T x C x F,
                # where C is the number of electrodes (256)
                # and F is the number of features (5)
                # could even do a 3D conv with spatial info, T x H x W x C x F
                # ResBlock(cfg.neural_reduced_features, cfg.d_model, beta=cfg.beta),
                ResBlock(cfg.neural_input_features, cfg.d_model, beta=cfg.beta),
                ResBlock(cfg.d_model, cfg.d_model, beta=cfg.beta**2),
                ResBlock(cfg.d_model, cfg.d_model, beta=cfg.beta**3),
            )

        # equivalent to w_raw_in in Gaddy's model
        # affine=False so emg&audio latent are both unit norm
        if not no_emg:
            self.emg_latent_linear = nn.Linear(cfg.d_model, cfg.d_model)
            self.emg_latent_norm = nn.BatchNorm1d(cfg.d_model, affine=cfg.latent_affine)
        if not no_neural:
            self.neural_latent_norm = nn.BatchNorm1d(
                cfg.d_model, affine=cfg.latent_affine
            )
            self.neural_latent_linear = nn.Linear(cfg.d_model, cfg.d_model)
        if not no_audio:
            self.audio_latent_norm = nn.BatchNorm1d(
                cfg.d_model, affine=cfg.latent_affine
            )
            self.audio_latent_linear = nn.Linear(cfg.d_model, cfg.d_model)

        encoder_layer = TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.num_heads,
            relative_positional=True,
            relative_positional_distance=100,
            dim_feedforward=cfg.d_inner,
            dropout=cfg.dropout,
            # beta=1/np.sqrt(2)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, cfg.attn_layers)
        self.w_out = nn.Linear(cfg.d_model, cfg.num_outs)

        self.seqlen = cfg.seqlen
        self.epochs = cfg.num_train_epochs

        # val/test procedure...
        self.text_transform = text_transform
        self.n_chars = len(text_transform.chars)
        self.lm_directory = cfg.lm_directory
        if cfg.togglePhones:
            self.lexicon_file = os.path.join(cfg.lm_directory, "cmudict.txt")
        else:
            self.lexicon_file = os.path.join(
                cfg.lm_directory, "lexicon_graphemes_noApostrophe.txt"
            )

        self._init_ctc_decoder()
        self.cross_nce_lambda = cfg.cross_nce_lambda
        self.audio_lambda = cfg.audio_lambda
        self.emg_lambda = cfg.emg_lambda
        self.neural_lambda = cfg.neural_lambda
        self.steps_per_epoch = cfg.steps_per_epoch

        self.sup_nce_lambda = cfg.sup_nce_lambda

        self.fixed_length = cfg.fixed_length
        self.use_dtw = cfg.use_dtw
        self.use_crossCon = cfg.use_crossCon
        self.use_supTcon = cfg.use_supTcon
        self.warmup_steps = cfg.warmup_steps
        # self.supervised_contrastive_loss = SupConLoss(temperature=0.1)

    def emg_encoder(self, x):
        "Encode emg (B x T x C) into a latent space (B x T/8 x D)"
        # print(f"emg_encoder: {x.shape=}")
        x = x.transpose(1, 2)  # put channel before time for conv
        x = self.emg_conv_blocks(x)
        x = x.transpose(1, 2)
        x = self.emg_latent_linear(x)
        logging.info(f"emg_encoder pre-norm: {x.shape=}")
        # TODO: unlike Gaddy, I believe we added this norm before the latent
        # for our best 26.7% WER, should uncomment these three lines...
        x = x.transpose(1, 2)  # channel first for batchnorm
        x = self.emg_latent_norm(x)
        x = x.transpose(1, 2)  # B x T/8 x C
        return x

    def neural_encoder(self, x, sessions=None):
        "Encode neural (B x T x C) into a latent space (B x Tau x D)"
        if self.use_session_input_encoder:
            x = self.session_input_encoder(x, sessions)
        else:
            # x = self.neural_input_encoder(x) # reduce number of inputs
            pass
        # x = self.neural_input_dropout(x)
        # x = self.neural_input_act(x)
        x = x.transpose(1, 2)
        x = self.neural_conv_blocks(x)
        x = x.transpose(1, 2)
        x = self.neural_latent_linear(x)
        # logging.info(f"neural_encoder pre-norm: {x.shape=}")
        x = x.transpose(1, 2)  # channel first for batchnorm
        x = self.neural_latent_norm(x)
        x = x.transpose(1, 2)  # B x T/8 x C
        return x

    def audio_encoder(self, x):
        "Encode audio (B x T x C) into a latent space (B x T/8 x D)"
        x = x.transpose(1, 2)  # put channel before time for conv
        x = self.audio_conv_blocks(x)
        x = x.transpose(1, 2)
        x = self.audio_latent_linear(x)
        logging.info(f"audio_encoder pre-norm: {x.shape=}")
        x = x.transpose(1, 2)  # channel first for batchnorm
        x = self.audio_latent_norm(x)
        x = x.transpose(1, 2)  # B x T/8 x C
        return x

    def decoder(self, x):
        """Predict characters from latent space (B x T/8 x D)"""
        x = x.transpose(0, 1)  # put time first
        # print(f"before transformer: {x.shape=}")
        x = self.transformer(x)
        x = x.transpose(0, 1)
        x = self.w_out(x)
        return F.log_softmax(x, 2)

    def augment_shift(self, x):
        if self.training:
            xnew = x.clone()  # unclear why need this here but gaddy didn't
            r = random.randrange(8)
            if r > 0:
                xnew[:, :-r, :] = x[:, r:, :]  # shift left r
                xnew[:, -r:, :] = 0
            return xnew
        else:
            return x

    def emg_forward(self, x):
        "Predict characters from emg (B x T x C)"
        x = self.augment_shift(x)
        z = self.emg_encoder(x)  # latent space
        return self.decoder(z), z

    def neural_forward(self, x, sessions=None):
        """Predict characters from neural features (B x Tau x 1280)

        20ms frames for neural

        """
        z = self.neural_encoder(x, sessions=sessions)  # latent space
        return self.decoder(z), z

    def audio_forward(self, x):
        "Predict characters from audio mel spectrogram (B x T/8 x 80)"
        z = self.audio_encoder(x)  # latent space
        return self.decoder(z), z

    def forward(self, batch, fixed_length=None):
        sessions = batch["sessions"] if "sessions" in batch else None
        emg_tup, neural_tup, audio_tup, idxs = split_batch_into_emg_neural_audio(batch)
        emg, length_emg, emg_phonemes, y_length_emg, y_emg, text_emg = emg_tup
        (
            neural,
            length_neural,
            neural_phonemes,
            y_length_neural,
            y_neural,
            text_neural,
        ) = neural_tup
        (
            audio,
            length_audio,
            audio_phonemes,
            y_length_audio,
            y_audio,
            text_audio,
        ) = audio_tup
        (
            paired_emg_idx,
            paired_audio_idx,
            silent_emg_idx,
            parallel_emg_idx,
            parallel_audio_idx,
        ) = idxs

        (
            (emg_pred, neural_pred, audio_pred),
            (emg_z, neural_z, audio_z),
            (emg_bz, neural_bz, audio_bz),
        ) = self.multi_forward(
            emg,
            neural,
            audio,
            length_emg,
            length_neural,
            length_audio,
            sessions=sessions,
            fixed_length=fixed_length,
        )
        ret = {
            "emg_pred": emg_pred,
            "neural_pred": neural_pred,
            "audio_pred": audio_pred,
            "emg_z": emg_z,
            "neural_z": neural_z,
            "audio_z": audio_z,
            "text_emg": text_emg,
            "text_neural": text_neural,
            "text_audio": text_audio,
            "y_emg": y_emg,
            "y_neural": y_neural,
            "y_audio": y_audio,
            "y_length_emg": y_length_emg,
            "y_length_neural": y_length_neural,
            "y_length_audio": y_length_audio,
            "length_emg": length_emg,
            "length_neural": length_neural,
            "length_audio": length_audio,
            "emg_phonemes": emg_phonemes,
            "neural_phonemes": neural_phonemes,
            "audio_phonemes": audio_phonemes,
            "paired_emg_idx": paired_emg_idx,
            "paired_audio_idx": paired_audio_idx,
            "silent_emg_idx": silent_emg_idx,
            "parallel_emg_idx": parallel_emg_idx,
            "parallel_audio_idx": parallel_audio_idx,
            "emg_bz": emg_bz,
            "neural_bz": neural_bz,
            "audio_bz": audio_bz,
        }
        return ret

    def multi_forward(
        self,
        emg: List[torch.Tensor],
        neural: List[torch.Tensor],
        audio: List[torch.Tensor],
        length_emg,
        length_neural,
        length_audio,
        sessions=None,
        fixed_length=None,
    ):
        """Group x by task and predict characters for the batch.

        Note that forward will call combine_fixed_length, re-splitting the batch into
        self.seqlen chunks. I believe this is done to avoid having to pad the batch to the max,
        which also may quadratically reduce memory usage due to attention. This is prob okay for
        training, but for inference we want to use the full sequence length."""
        if fixed_length is None:
            fixed_length = self.fixed_length
        if len(emg) > 0:
            # print(f"FORWARD emg shape: {[e.shape for e in emg]=}")
            if fixed_length:
                emg = combine_fixed_length(emg, self.seqlen * 8)
            else:
                emg = nn.utils.rnn.pad_sequence(emg, batch_first=True)
            # logging.debug(f"FORWARD emg shape: {emg.shape}")
            emg_pred, emg_z = self.emg_forward(emg)
            emg_bz = len(emg)  # batch size not known until after combine_fixed_length
            length_emg = [l // 8 for l in length_emg]
            # logging.debug(f"before decollate {len(emg_pred)=}, {emg_pred[0].shape=}")
            if fixed_length:
                emg_pred = decollate_tensor(emg_pred, length_emg)
                emg_z = decollate_tensor(emg_z, length_emg)
            # logging.debug(f"after decollate {len(emg_pred)=}, {emg_pred[0].shape=}")
            # logging.debug(f"before decollate {len(emg_z)=}, {emg_z[0].shape=}")
            # # TODO: perhaps we shouldn't decollate z, since we need to use it cross contrastive loss
            # INFO: but we have to decollate or else we don't know which audio to pair with which emg
            # logging.debug(f"after decollate {len(emg_z)=}, {emg_z[0].shape=}")
        else:
            emg_pred, emg_z, emg_bz = None, None, 0

        if len(neural) > 0:
            if fixed_length:
                neural = combine_fixed_length(neural, self.seqlen)
            else:
                neural = nn.utils.rnn.pad_sequence(neural, batch_first=True)
            # logging.debug(f"FORWARD neural shape: {neural.shape}")
            neural_pred, neural_z = self.neural_forward(neural, sessions=sessions)
            neural_bz = len(neural)
            if fixed_length:
                neural_pred = decollate_tensor(neural_pred, length_neural)
                neural_z = decollate_tensor(neural_z, length_neural)
        else:
            # raise ValueError("Expecting neural right now")
            neural_pred, neural_z, neural_bz = None, None, 0

        if len(audio) > 0:
            if fixed_length:
                audio = combine_fixed_length(audio, self.seqlen)
            else:
                audio = nn.utils.rnn.pad_sequence(audio, batch_first=True)
            # logging.debug(f"FORWARD audio shape: {audio.shape}")
            audio_pred, audio_z = self.audio_forward(audio)
            audio_bz = len(audio)
            if fixed_length:
                audio_pred = decollate_tensor(audio_pred, length_audio)
                audio_z = decollate_tensor(audio_z, length_audio)
        else:
            audio_pred, audio_z, audio_bz = None, None, 0

        # logging.debug("finished FORWARD")
        return (
            (emg_pred, neural_pred, audio_pred),
            (emg_z, neural_z, audio_z),
            (emg_bz, neural_bz, audio_bz),
        )

    # TODO: can we simplify this somehow..? 23 required args is a lot
    def calc_loss(
        self,
        emg_pred,
        neural_pred,
        audio_pred,
        emg_z,
        neural_z,
        audio_z,
        y_emg,
        y_neural,
        y_audio,
        y_length_emg,
        y_length_neural,
        y_length_audio,
        length_emg,
        length_neural,
        length_audio,
        emg_phonemes,
        neural_phonemes,
        audio_phonemes,
        paired_emg_idx,
        paired_audio_idx,
        silent_emg_idx,
        parallel_emg_idx,
        parallel_audio_idx,
        emg_bz,
        neural_bz,
        audio_bz,
        use_supTcon=None,
        use_crossCon=None,
        use_dtw=None,
        **kwargs,
    ):
        # print(f"{torch.concatenate(emg_z).shape=}, {torch.concatenate(audio_z).shape=}, {torch.concatenate(emg_phonemes).shape=}, {torch.concatenate(audio_phonemes).shape=}")
        if use_supTcon is None:
            use_supTcon = self.use_supTcon
        if use_crossCon is None:
            use_crossCon = self.use_crossCon
        if use_dtw is None:
            use_dtw = self.use_dtw

        # we assume every emg example is either silent,
        # paired (simultaneous with audio), or parallel
        assert len(emg_pred) == len(silent_emg_idx) + len(paired_emg_idx) + len(
            parallel_emg_idx
        ), f"{len(emg_pred)=}, {len(silent_emg_idx)=}, {len(paired_emg_idx)=}, {len(parallel_emg_idx)=}"

        if emg_pred is not None:
            length_emg = [
                l // 8 for l in length_emg
            ]  # Gaddy doesn't do this but I think it's necessary
            emg_ctc_loss = self.ctc_loss(emg_pred, y_emg, length_emg, y_length_emg)
        else:
            logging.info("emg_pred is None")
            emg_ctc_loss = 0.0

        if neural_pred is not None:
            neural_ctc_loss = self.ctc_loss(
                neural_pred, y_neural, length_neural, y_length_neural
            )
        else:
            logging.info("neural_pred is None")
            neural_ctc_loss = 0.0

        if audio_pred is not None:
            audio_ctc_loss = self.ctc_loss(
                audio_pred, y_audio, length_audio, y_length_audio
            )
        else:
            logging.info("audio_pred is None")
            audio_ctc_loss = 0.0

        # TODO: we should refactor into another function...
        # and also figure out how not to write code for cartesian product
        # INFO: this block is for Audio&EMG contrastive loss only
        if emg_z is not None and audio_z is not None:
            # use DTW with parallel audio/emg to align phonemes with silent emg
            silent_e_z = torch.concatenate([emg_z[i] for i in silent_emg_idx])
            parallel_e_z = torch.concatenate([emg_z[i] for i in parallel_emg_idx])
            parallel_a_z = torch.concatenate([audio_z[i] for i in parallel_audio_idx])
            parallel_a_phonemes = torch.concatenate(
                [audio_phonemes[i] for i in parallel_audio_idx]
            )

            if use_dtw:
                # euclidean distance between silent emg and parallel audio
                # costs = torch.cdist(parallel_a_z, silent_e_z).squeeze(0)
                # cosine dissimiliarity between silent emg and parallel audio
                # costs = 1 - torchmetrics.functional.pairwise_cosine_similarity(parallel_a_z, silent_e_z).squeeze(0)

                # euclidean distance between silent emg and parallel emg
                costs = torch.cdist(parallel_e_z, silent_e_z).squeeze(0)
                # cosine dissimiliarity between silent emg and parallel emg
                # costs = 1 - torchmetrics.functional.pairwise_cosine_similarity(parallel_e_z, silent_e_z).squeeze(0)

                # print(f"cdist: {costs.dtype}")
                # print(f"cos dissim: {costs.dtype}")

            if use_crossCon or use_supTcon:
                # save on compute & avoid val crashes by only computing alignment on train

                emg_to_concat = [emg_z[i] for i in paired_emg_idx] + [
                    emg_z[i] for i in parallel_emg_idx
                ]
                audio_to_concat = [audio_z[i] for i in paired_audio_idx] + [
                    audio_z[i] for i in parallel_audio_idx
                ]

                if use_dtw:
                    alignment = align_from_distances(
                        costs.T.detach().cpu().float().numpy()
                    )
                    aligned_a_z = parallel_a_z[alignment]
                    logging.debug(
                        f"{len(alignment)=}, {max(alignment)=}, {len(parallel_a_z)=}, {len(aligned_a_z)=}"
                    )
                    frames_alignment = [
                        a // 8 for a in alignment
                    ]  # downsample to phoneme frames
                    aligned_a_phonemes = parallel_a_phonemes[frames_alignment]
                    emg_to_concat.append(silent_e_z)
                    audio_to_concat.append(aligned_a_z)
                # print(f"{silent_e_z.shape=}, {parallel_a_z.shape=}, {parallel_a_phonemes.shape=}," \
                #       f"{len(alignment)=}, {aligned_a_z.shape=}, {aligned_a_phonemes.shape=}")

                matched_e_z = torch.concatenate(emg_to_concat)
                matched_a_z = torch.concatenate(audio_to_concat)

            ###### InfoNCE #####
            # contrastive loss with emg_t, audio_t as positive pairs
            # TODO: need to make a memoizing cos sim function..?
            if use_crossCon:
                emg_audio_contrastive_loss = nobatch_cross_contrastive_loss(
                    matched_e_z, matched_a_z, device=self.device
                )
            else:
                emg_audio_contrastive_loss = 0.0

            ###### Supervised NCE #######
            if use_supTcon:
                emg_phonemes_to_concat = [emg_phonemes[i] for i in paired_emg_idx] + [
                    emg_phonemes[i] for i in parallel_emg_idx
                ]
                if self.use_dtw:
                    emg_phonemes_to_concat.append(aligned_a_phonemes)
                matched_phonemes = torch.concatenate(emg_phonemes_to_concat)
                # for e,a in zip(paired_emg_idx,paired_audio_idx):
                #     print(f"{emg_phonemes[e].shape=}, {emg_z[e].shape=}, {audio_phonemes[a].shape=}, {audio_z[a].shape=}")
                # print(f"{matched_e_z.shape=}, {len(audio_z)=}, {silent_e_z.shape=}, {len(paired_e_phonemes)=}, {len(audio_phonemes)=}, {len(aligned_a_phonemes)=}")
                # # TODO: we are duplicating some audio_z here. need to fix
                # print(f"{matched_e_z.shape=}, {torch.concatenate([*paired_e_phonemes]).shape=})")
                # print(f"{torch.concatenate([*audio_z]).shape=}, {torch.concatenate([*audio_phonemes]).shape=})")
                # print(f"{silent_e_z.shape=}, {aligned_a_phonemes.shape=}")
                # logging.debug(f"{matched_e_z.shape=}, {matched_phonemes.shape=}")
                # logging.debug(f"{[a.shape for a in audio_z]=}, {[a.shape for a in audio_phonemes]=}")
                z = torch.concatenate([matched_e_z, *audio_z])
                z_class = torch.concatenate([matched_phonemes, *audio_phonemes])
                sup_nce_loss = supervised_contrastive_loss(
                    z, z_class, device=self.device
                )
            else:
                sup_nce_loss = 0.0
            # sup_nce_loss = self.supervised_contrastive_loss(z[:, None], z_class)

            ######
        elif emg_z is not None:
            # INFO: phoneme labels aren't frame-aligned with emg, so we can't use them
            # TODO: try DTW with parallel audio/emg to align phonemes with silent emg
            # z = torch.concatenate(emg_z)
            # z_class = torch.concatenate(emg_phonemes)
            emg_audio_contrastive_loss = 0.0
            sup_nce_loss = 0.0
        # elif audio_z is not None:
        #     raise NotImplementedError("audio only is not expected")
        #     z = torch.concatenate(audio_z)
        #     z_class = torch.concatenate(audio_phonemes)
        else:
            emg_audio_contrastive_loss = 0.0
            sup_nce_loss = 0.0

        # TODO: add neural sup contrastive loss
        if neural_z is not None:
            pass

        # logging.debug(f"{z_class=}")

        # assert audio_pred is None, f'Audio only not implemented, got {audio_pred=}'
        logging.debug(
            f"emg_ctc_loss: {emg_ctc_loss}, audio_ctc_loss: {audio_ctc_loss}, "
            f"emg_audio_contrastive_loss: {emg_audio_contrastive_loss}, "
            f"sup_nce_loss: {sup_nce_loss}"
        )
        loss = (
            self.emg_lambda * emg_ctc_loss
            + self.neural_lambda * neural_ctc_loss
            + self.audio_lambda * audio_ctc_loss
            + self.cross_nce_lambda * emg_audio_contrastive_loss
            + self.sup_nce_lambda * sup_nce_loss
        )

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

        cross_con_bz = len(paired_emg_idx) + len(parallel_emg_idx)
        if use_dtw:
            cross_con_bz += len(silent_emg_idx)
        if not emg_z is None:
            emg_z_mean = torch.concatenate([e.reshape(-1).abs() for e in emg_z]).mean()
        else:
            emg_z_mean = None

        if not neural_z is None:
            neural_z_mean = torch.concatenate(
                [n.reshape(-1).abs() for n in neural_z]
            ).mean()
        else:
            neural_z_mean = None

        if not audio_z is None:
            audio_z_mean = torch.concatenate(
                [a.reshape(-1).abs() for a in audio_z]
            ).mean()
        else:
            audio_z_mean = None

        return {
            "loss": loss,
            "emg_ctc_loss": emg_ctc_loss,
            "neural_ctc_loss": neural_ctc_loss,
            "audio_ctc_loss": audio_ctc_loss,
            "cross_contrastive_loss": emg_audio_contrastive_loss,
            "supervised_contrastive_loss": sup_nce_loss,
            "emg_z_mean": emg_z_mean,
            "neural_z_mean": neural_z_mean,
            "audio_z_mean": audio_z_mean,
            "emg_bz": emg_bz,
            "neural_bz": neural_bz,
            "audio_bz": audio_bz,
            "cross_con_bz": cross_con_bz,
        }

    def _beam_search_batch(self, batch):
        "Repeatedly called by validation_step & test_step."
        warn("assumes neural data right now...")
        # TODO: refactor so easy to call on nerual, emg, audio
        X = nn.utils.rnn.pad_sequence(batch["neural_features"], batch_first=True)
        # X = nn.utils.rnn.pad_sequence(batch['raw_emg'], batch_first=True)
        pred = self.neural_forward(X, sessions=batch["sessions"])[0].cpu()
        # pred  = self.emg_forward(X)[0].cpu()

        beam_results = self.ctc_decoder(pred)
        pred_text = []
        pred_int = []
        for b in beam_results:
            if len(b) > 0:
                # I think length is zero only when there's NaNs in the output
                # we could just allow the crash here
                pred_text.append(" ".join(b[0].words).strip().lower())
                pred_int.append(b[0].tokens)
            else:
                pred_text.append("")
                pred_int.append([])
        target_text = [self.text_transform.clean_text(b) for b in batch["text"]]

        target_int = batch["text_int"]

        return target_text, pred_text, target_int, pred_int

    # def on_before_optimizer_step(self, optimizer):
    #     # Compute the 2-norm for each layer
    #     # If using mixed precision, the gradients are already unscaled here
    #     norms = pl.utilities.grad_norm(self.layer, norm_type=2)
    #     self.log_dict(norms)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        # Warmup steps and epochs
        milestone_epochs = [125, 150, 175]

        # Total number of steps for training
        total_steps = self.trainer.estimated_stepping_batches
        # print(f"DEBUG: {total_steps=}")

        # Define the lambda function for learning rate schedule
        lr_lambda = (
            lambda step: min(1.0, step / self.warmup_steps)
            if step < self.warmup_steps
            else 0.5
            ** len(
                [
                    m
                    for m in milestone_epochs
                    if m * total_steps // self.trainer.max_epochs <= step
                ]
            )
        )

        # Scheduler with linear warmup and decay at specified epochs
        scheduler = LambdaLR(optimizer, lr_lambda)

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
