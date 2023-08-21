import random
import torch, numpy as np
from torch import nn
import torch.nn.functional as F
from transformer import TransformerEncoderLayer
from data_utils import combine_fixed_length, decollate_tensor

import sys, os, jiwer
import pytorch_lightning as pl, torchmetrics
from torchaudio.models.decoder import ctc_decoder
from s4 import S4
from data_utils import TextTransform
from magneto.models.hyena import HyenaOperator
from flash_attn.modules.block import Block
from magneto.models.s4d import S4D

from pytorch_lightning.profilers import PassThroughProfiler
from dataclasses import dataclass
from typing import Tuple, List
from dataloaders import split_batch_into_emg_neural_audio
from contrastive import \
    nobatch_cross_contrastive_loss, supervised_contrastive_loss, SupConLoss
from typing import Tuple
from pytorch_lightning.loggers import NeptuneLogger
from align import align_from_distances

import gc
import logging

MODEL_SIZE = 768 # number of hidden dimensions
NUM_LAYERS = 6 # number of layers
DROPOUT = .2 # dropout

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
    def __init__(self, num_ins, num_outs, stride=1, pre_activation=False,
                 beta:float=1.):
        super().__init__()

        self.conv1 = nn.Conv1d(num_ins, num_outs, 3, padding=1, stride=stride)
        self.norm1 = nn.BatchNorm1d(num_outs)
        self.conv2 = nn.Conv1d(num_outs, num_outs, 3, padding=1)
        self.norm2 = nn.BatchNorm1d(num_outs)
        # self.act = nn.ReLU()
        self.act = nn.GELU() # TODO: test which is better
        self.beta = beta

        if stride != 1 or num_ins != num_outs:
            self.residual_path = nn.Conv1d(num_ins, num_outs, 1, stride=stride)
            self.res_norm = nn.BatchNorm1d(num_outs)
            if pre_activation:
                self.skip = nn.Sequential(
                    self.res_norm, self.residual_path)
            else:
                self.skip = nn.Sequential(
                    self.residual_path, self.res_norm)
        else:
            self.skip = nn.Identity()
            
        # ResNet v2 style pre-activation https://arxiv.org/pdf/1603.05027.pdf
        self.pre_activation = pre_activation
        
        if pre_activation:
            self.block = nn.Sequential(
                self.norm1, self.act,
                self.conv1,
                self.norm2, self.act,
                self.conv2
            )
        else:
            self.block = nn.Sequential(
                self.conv1,
                self.norm1, self.act,
                self.conv2,
                self.norm2
            )

    def forward(self, x):
        # logging.warning(f"ResBlock forward pass. x.shape: {x.shape}")
        res = self.block(x) * self.beta
        x = self.skip(x)
        
        if self.pre_activation:
            return x + res
        else:
            return self.act(x + res)
    
    
class Model(pl.LightningModule):
    def __init__(self, model_size, dropout, num_layers, num_outs, text_transform: TextTransform,
                 steps_per_epoch, epochs, lm_directory, num_aux_outs=None, lr=3e-4,
                 learning_rate_warmup = 1000, profiler = None, weight_decay=0.0):
        super().__init__()
        self.profiler = profiler or PassThroughProfiler()
        self.conv_blocks = nn.Sequential(
            ResBlock(8, model_size, 2),
            ResBlock(model_size, model_size, 2),
            ResBlock(model_size, model_size, 2),
        )
        self.w_raw_in = nn.Linear(model_size, model_size)
        encoder_layer = TransformerEncoderLayer(d_model=model_size, nhead=8, relative_positional=True, 
                                                relative_positional_distance=100, dim_feedforward=3072, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.w_out       = nn.Linear(model_size, num_outs)

        self.has_aux_out = num_aux_outs is not None
        if self.has_aux_out:
            self.w_aux = nn.Linear(model_size, num_aux_outs)
            
        self.seqlen = 600
        self.lr = lr
        self.target_lr = lr # will not mutate
        self.learning_rate_warmup = learning_rate_warmup
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        
        # val/test procedure...
        self.text_transform = text_transform
        self.n_chars = len(text_transform.chars)
        self.lm_directory = lm_directory
        self.lexicon_file = os.path.join(lm_directory, 'lexicon_graphemes_noApostrophe.txt')
        self._init_ctc_decoder()
        
        self.step_target = []
        self.step_pred = []
        self.weight_decay = weight_decay

    def _init_ctc_decoder(self):
        self.ctc_decoder = ctc_decoder(
            lexicon = self.lexicon_file,
            tokens  = self.text_transform.chars + ['_'],
            lm      = os.path.join(self.lm_directory, '4gram_lm.bin'),
            blank_token = '_',
            sil_token   = '|',
            nbest       = 1,
            lm_weight   = 2, # default is 2; Gaddy sets to 1.85
            #word_score  = -3,
            #sil_score   = -2,
            beam_size   = 150  # SET TO 150 during inference
        )


    def forward(self, x_raw):
        # x shape is (batch, time, electrode)

        if self.training:
            r = random.randrange(8)
            if r > 0:
                x_raw[:,:-r,:] = x_raw[:,r:,:] # shift left r
                x_raw[:,-r:,:] = 0
        
        x_raw = x_raw.transpose(1,2) # put channel before time for conv
        # print(f"before conv: {x_raw.shape=}")
        x_raw = self.conv_blocks(x_raw)
        # print(f"after conv: {x_raw.shape=}")
        x_raw = x_raw.transpose(1,2)
        x_raw = self.w_raw_in(x_raw)

        x = x_raw
        x = x.transpose(0,1) # put time first
        # print(f"before transformer: {x.shape=}")
        x = self.transformer(x)
        x = x.transpose(0,1)

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
        X     = combine_fixed_length(batch['emg'], self.seqlen)
        X_raw = combine_fixed_length(batch['raw_emg'], self.seqlen*8)
        bz = X.shape[0]
    
        pred = self(X_raw)

        # seq first, as required by ctc
        pred = nn.utils.rnn.pad_sequence(decollate_tensor(pred, batch['lengths']), batch_first=False) 
        y    = nn.utils.rnn.pad_sequence(batch['text_int'], batch_first=True)
        loss = F.ctc_loss(pred, y, batch['lengths'], batch['text_int_lengths'], blank=self.n_chars)
        
        if torch.isnan(loss) or torch.isinf(loss):
            # print('batch:', batch_idx)
            print('Isnan output:',torch.any(torch.isnan(pred)))
            print('Isinf output:',torch.any(torch.isinf(pred)))
            # raise ValueError("NaN/Inf detected in loss")
            
        return loss, bz
    
    def _beam_search_step(self, batch):
        "Repeatedly called by validation_step & test_step. Impure function!"
        X_raw = batch['raw_emg'][0].unsqueeze(0)

        pred  = self(X_raw).cpu()

        beam_results = self.ctc_decoder(pred)
        pred_text    = ' '.join(beam_results[0][0].words).strip().lower()
        b0 = batch['text'][0]
        if len(batch['text'][0]) == 1:
            # index twice for gaddy's collate function
            target_text  = self.text_transform.clean_2(b0[0])
        else:
            # Only index once for new collate function
            target_text  = self.text_transform.clean_2(b0)

        return target_text, pred_text
    
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
    
    def training_step(self, batch, batch_idx):
        loss, bz = self.calc_loss(batch)
        self.log("train/loss", loss, on_step=False, on_epoch=True, logger=True, prog_bar=True, batch_size=bz)
        return loss
    
    def on_validation_epoch_start(self):
        # self.profiler.start(f"validation loop")
        self._init_ctc_decoder()
    
    def validation_step(self, batch, batch_idx):
        loss, bz = self.calc_loss(batch)
        target_text, pred_text = self._beam_search_step(batch)
        assert len(batch['emg']) == 1, "Currently only support batch size of 1 for validation"
        if len(target_text) > 0:
            self.step_target.append(target_text)
            self.step_pred.append(pred_text)
            
        self.log("val/loss", loss, prog_bar=True, batch_size=bz)
        return loss
    
    def on_validation_epoch_end(self) -> None:
        # TODO: this may not be implemented correctly for DDP
        logging.warning(f"start on_validation_epoch_end")
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
        logging.warning(f"on_validation_epoch_end: calc wer")
        wer = jiwer.wer(step_target, step_pred)
        self.step_target.clear()
        self.step_pred.clear()
        self.log("val/wer", wer, prog_bar=True, sync_dist=True)
        # self.profiler.stop(f"validation loop")
        # self.profiler.describe()
        logging.warning(f"on_validation_epoch_end: gc.collect()")
        gc.collect()
        torch.cuda.empty_cache() # TODO: see if fixes occasional freeze...?

    def test_step(self, batch, batch_idx):
        loss, bz = self.calc_loss(batch)
        target_text, pred_text = self._beam_search_step(batch)
        if len(target_text) > 0:
            self.step_target.append(target_text)
            self.step_pred.append(pred_text)
        self.log("test/loss", loss, prog_bar=True, batch_size=bz)
        return loss
    
    def on_test_epoch_end(self) -> None:
        wer = jiwer.wer(self.step_target, self.step_pred)
        self.step_target.clear()
        self.step_pred.clear()
        self.log("test/wer", wer, prog_bar=True)

    def configure_optimizers(self):
        initial_lr = self.target_lr/self.learning_rate_warmup
        
        # for FSDP
        optimizer = torch.optim.AdamW(self.trainer.model.parameters(), lr=initial_lr,
                                      weight_decay=self.weight_decay)

        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
            milestones=[
                125 * self.steps_per_epoch,
                150  * self.steps_per_epoch,
                175 * self.steps_per_epoch],
            gamma=.5)
        lr_scheduler = {'scheduler': scheduler, 'interval': 'step'}

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
    
    
    def set_lr(self, new_lr):
        optimizer = self.optimizers().optimizer
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
            
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
            new_lr = self.global_step*self.target_lr/self.learning_rate_warmup
            self.set_lr(new_lr)


class S4Layer(nn.Module):
    """
    https://github.com/HazyResearch/state-spaces/blob/ab287c63f4938a76d06a6b6868ee4a7163b50b05/example.py
    
    Abstraction layer that gives more fine-grained control over S4 design. 
    This module has a S4Kernel, dropout, and layer norm.
    """
    def __init__(self, model_size, dropout, s4_dropout = None, diagonal = False, prenorm = False):
        super().__init__()
                  
        self.model_size = model_size
        self.s4_dropout = dropout if s4_dropout is None else s4_dropout
        
        if diagonal:
            self.s4_layer = S4(model_size, dropout=self.s4_dropout, bidirectional = True, transposed=True, 
                               lr=None, mode = 'diag', measure = 'diag-inv', disc='zoh', real_type='exp')       
        else:
            self.s4_layer = S4(model_size, dropout=self.s4_dropout, bidirectional = True, transposed=True, lr=None)

        self.norm     = nn.LayerNorm(model_size)
        self.dropout  = nn.Dropout1d(dropout)
        #self.dropout  = nn.Dropout(dropout)
        self.prenorm  = prenorm
        
    def forward(self, x):
        """
        Input x is list of tensors with shape (B, L, d_input)
        Returns tensor of same size.
        """

        x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
        z = x
        
        if self.prenorm: # Prenorm
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
        self.prenorm  = False 
        self.diagonal = False
        
        # Linear encoder
        self.encoder = nn.Sequential(
            nn.Linear(8, MODEL_SIZE),
            nn.Softsign(),
            nn.Linear(8, MODEL_SIZE)
        )
        
        
        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms     = nn.ModuleList()
        self.dropouts  = nn.ModuleList()
        self.linears   = nn.ModuleList()
        for i in range(NUM_LAYERS):
            if i > 2:
                s4_dropout = DROPOUT
            #   # channels = 2
            #else:
                s4_dropout = 0
            #  #  channels = 3
            
            s4_dropout = DROPOUT
            
            dropout = DROPOUT
            self.s4_layers.append(S4Layer(MODEL_SIZE, dropout, s4_dropout = s4_dropout))
        
        self.w_out = nn.Linear(MODEL_SIZE, num_outs)
        
        self.has_aux_out = num_aux_outs is not None
        if self.has_aux_out:
            self.w_aux = nn.Linear(MODEL_SIZE, num_aux_outs)

            
    def forward(self, x_raw):
        # x shape is (batch, time, electrode)

        if self.training:
            r = random.randrange(8)
            if r > 0:
                x_raw[:,:-r,:] = x_raw[:,r:,:] # shift left r
                x_raw[:,-r:,:] = 0
                
        x = self.encoder(x_raw)                
                                  
        for i, layer in enumerate(self.s4_layers):
            x = layer(x)
            
            #if i == 2 or i == 4 or i == 6:
            #    x = x[:, ::2, :] # 8x downsampling
            if i <= 2:
                x = x[:, ::2, :]
                                
        if self.has_aux_out:
            return self.w_out(x), self.w_aux(x)
        else:
            return self.w_out(x)
        
sys.path.append('/home/users/ghwilson/repos/safari/src/models/sequence/')
sys.path.append('/home/users/ghwilson/repos/safari/')
try:
    from h3 import H3
except:
    print('Could not import H3')

class H3Model(nn.Module):
    def __init__(self, num_features, num_outs, num_aux_outs=None):
        super().__init__()
        self.prenorm = False 
        
        # Linear encoder
        self.encoder = nn.Linear(8, MODEL_SIZE)
        
        # Stack S4 layers as residual blocks
        self.h3_layers = nn.ModuleList()
        self.norms     = nn.ModuleList()
        self.dropouts  = nn.ModuleList()
        self.linears   = nn.ModuleList()
        for i in range(NUM_LAYERS):
            self.h3_layers.append(
                H3(d_model = MODEL_SIZE, dropout=DROPOUT, lr=None)
            )
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
                x_raw[:,:-r,:] = x_raw[:,r:,:] # shift left r
                x_raw[:,-r:,:] = 0
                
        x = self.encoder(x_raw) 
        #x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
        for i, (layer, norm, dropout) in enumerate(zip(self.h3_layers, self.norms, self.dropouts)):

            z = x
            if self.prenorm:
                # Prenorm
                z = norm(z)

            # Apply H3 block
           # print(z.shape)
            z  = layer(z)
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


@dataclass
class MONAConfig:
    steps_per_epoch:int
    lm_directory:str
    num_outs:int
    input_channels:int = 8
    learning_rate:float = 3e-4 # also sets initial s4 lr
    weight_decay:float = 0.01
    warmup_steps:int = None # warmup by backward steps
    batch_size:int = 12 # not necessary depending on batch sampler
    num_workers:int = 0
    num_train_epochs:int = 200
    gradient_accumulation_steps:int = 1
    sample_rate:int = 16000
    precision:str = "16-mixed"
    seqlen:int = 600
    attn_layers:int = 6
    d_model:int = 768 # original Gaddy
    # https://iclr-blog-track.github.io/2022/03/25/unnormalized-resnets/#balduzzi17shattered
    beta:float = 1 / np.sqrt(2) # adjust resnet initialization
    neural_input_features:int = 1280
    neural_reduced_features:int = 256 # reduce 1280 down to this features
    
    cross_nce_lambda:float = 1.0 # how much to weight the latent loss
    audio_lambda:float = 1.0 # how much to weight the audio->text loss
    neural_lambda:float = 1.0 # how much to weight the neural->text loss
    sup_nce_lambda:float = 0.1

    d_inner:int = 3072 # original Gaddy
    prenorm:bool = False
    dropout:float = 0.2
    in_channels:int = 8
    out_channels:int = 80
    resid_dropout:float = 0.0
    max_len:int = 480000
    num_heads:int = 8
    fixed_length:bool = False # gaddy style fixed length of combining examples
    
    def __post_init__(self):
        if self.warmup_steps is None:
            self.warmup_steps = 500 // self.gradient_accumulation_steps

class MONA(Model):
    "Multimodal Orofacial Neural Audio"
    
    def __init__(self, cfg:MONAConfig, text_transform:TextTransform,
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
        self.neural_conv_blocks = nn.Sequential(
            # TODO: should we do a 2D conv here..? T x C x F,
            # where C is the number of electrodes (256)
            # and F is the number of features (5)
            # could even do a 3D conv with spatial info, T x H x W x C x F
            ResBlock(cfg.neural_reduced_features, cfg.d_model, beta=cfg.beta), # 1280 neural features (5 * 256 channels)
            ResBlock(cfg.d_model, cfg.d_model, beta=cfg.beta**2),
            ResBlock(cfg.d_model, cfg.d_model, beta=cfg.beta**3)
        )
        self.neural_input_encoder = nn.Linear(cfg.neural_input_features, cfg.neural_reduced_features)
        # equivalent to w_raw_in in Gaddy's model
        self.emg_latent_linear = nn.Linear(cfg.d_model, cfg.d_model)
        # self.emg_latent_norm = nn.BatchNorm1d(cfg.d_model)
        # self.audio_latent_norm = nn.BatchNorm1d(cfg.d_model)
        # affine=False so emg&audio latent are both unit norm
        self.emg_latent_norm = nn.BatchNorm1d(cfg.d_model, affine=False)
        self.neural_pre_norm = nn.BatchNorm1d(cfg.neural_input_features)
        self.neural_latent_norm = nn.BatchNorm1d(cfg.d_model, affine=False)
        self.neural_latent_linear = nn.Linear(cfg.d_model, cfg.d_model)
        self.audio_latent_norm = nn.BatchNorm1d(cfg.d_model, affine=False)
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
        self.weight_decay = cfg.weight_decay
        
        # val/test procedure...
        self.text_transform = text_transform
        self.n_chars = len(text_transform.chars)
        self.lm_directory = cfg.lm_directory
        self.lexicon_file = os.path.join(cfg.lm_directory, 'lexicon_graphemes_noApostrophe.txt')
        self._init_ctc_decoder()
        self.cross_nce_lambda = cfg.cross_nce_lambda
        self.audio_lambda = cfg.audio_lambda
        self.neural_lambda = cfg.neural_lambda
        self.steps_per_epoch = cfg.steps_per_epoch
        
        self.step_target = []
        self.step_pred = []
        self.sup_nce_lambda = cfg.sup_nce_lambda
        
        self.fixed_length = cfg.fixed_length
        
        # self.supervised_contrastive_loss = SupConLoss(temperature=0.1)
    
    def emg_encoder(self, x):
        "Encode emg (B x T x C) into a latent space (B x T/8 x D)"
        # print(f"emg_encoder: {x.shape=}")
        x = x.transpose(1,2) # put channel before time for conv
        x = self.emg_conv_blocks(x)
        x = x.transpose(1,2)
        x = self.emg_latent_linear(x)
        logging.info(f"emg_encoder pre-norm: {x.shape=}")
        # TODO: unlike Gaddy, I believe we added this norm before the latent
        # for our best 26.7% WER, should uncomment these three lines...
        x = x.transpose(1,2) # channel first for batchnorm
        x = self.emg_latent_norm(x)
        x = x.transpose(1,2) # B x T/8 x C
        return x
    
    def neural_encoder(self, x):
        "Encode neural (B x T x C) into a latent space (B x Tau x D)"
        # x = x.transpose(1,2)
        # x = self.neural_pre_norm(x)
        # x = x.transpose(1,2)
        x = self.neural_input_encoder(x) # reduce number of inputs
        x = x.transpose(1,2)
        x = self.neural_conv_blocks(x)
        x = x.transpose(1,2)
        x = self.neural_latent_linear(x)
        # logging.info(f"neural_encoder pre-norm: {x.shape=}")
        x = x.transpose(1,2) # channel first for batchnorm
        x = self.neural_latent_norm(x)
        x = x.transpose(1,2) # B x T/8 x C
        return x     

        
    def audio_encoder(self, x):
        "Encode emg (B x T x C) into a latent space (B x T/8 x D)"
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
        """Predict characters from latent space (B x T/8 x D)"""
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
    
    def neural_forward(self, x):
        """Predict characters from neural features (B x Tau x 1280)
        
        20ms frames for neural
        
        """
        z = self.neural_encoder(x) # latent space
        return self.decoder(z), z

    def audio_forward(self, x):
        "Predict characters from audio mel spectrogram (B x T/8 x 80)"
        z = self.audio_encoder(x) # latent space
        return self.decoder(z), z
    
    def forward(self, emg:List[torch.Tensor], neural:List[torch.Tensor], audio:List[torch.Tensor],
                length_emg, length_neural, length_audio, fixed_length=None):
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
                emg = combine_fixed_length(emg, self.seqlen*8)
            else:
                emg = nn.utils.rnn.pad_sequence(emg, batch_first=True)
            # logging.debug(f"FORWARD emg shape: {emg.shape}")
            emg_pred, emg_z = self.emg_forward(emg)
            emg_bz = len(emg) # batch size not known until after combine_fixed_length
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
            neural_pred, neural_z = self.neural_forward(neural)
            neural_bz = len(neural)
            if fixed_length:
                neural_pred = decollate_tensor(neural_pred, length_neural)
                neural_z = decollate_tensor(neural_z, length_neural)
        else:
            raise ValueError("Expecting neural right now")
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
        return (emg_pred, neural_pred, audio_pred), (emg_z, neural_z, audio_z), \
               (emg_bz, neural_bz, audio_bz)
        
    def ctc_loss(self, pred, target, pred_len, target_len):
        # INFO: Gaddy passes emg length, but shouldn't this actually be divided by 8?
        # TODO: try padding length / 8. must be integers though...
        # print(f"ctc_loss: {pred_len=}, {target_len=}")
        
        # TODO FIXME
        # ctc_loss: [p.shape for p in pred]=[torch.Size([600, 38]), torch.Size([600, 38]), torch.Size([600, 38]), torch.Size([600, 38])], [t.shape for t in target]=[torch.Size([306])]
        # print(f"{pred.shape=}, {target[0].shape=}, {pred_len=}, {target_len=}")
        
        # this pads with 0, which corresponds to 'a', but by passing target_len 
        # to CTC loss we can ignore these padded values
        pred = nn.utils.rnn.pad_sequence(pred, batch_first=False) 
        # pred = nn.utils.rnn.pad_sequence(decollate_tensor(pred, pred_len), batch_first=False) 
        # pred = nn.utils.rnn.pad_sequence(pred, batch_first=False)
        target    = nn.utils.rnn.pad_sequence(target, batch_first=True)
        # print(f"\n ==== CTC ====\n{pred.shape=}, {target.shape=}\n{pred=}\n{target=}\n")
        # print(f"{pred.shape=}, {target[0].shape=}, {pred_len=}, {target_len=}")
        # print(f"ctc_loss: {[p.shape for p in pred]=}, {[t.shape for t in target]=}")
        loss = F.ctc_loss(pred, target, pred_len, target_len, blank=self.n_chars)
        return loss

    def batch_forward(self, batch):
        emg_tup, neural_tup, audio_tup, idxs = split_batch_into_emg_neural_audio(batch)
        emg, length_emg, emg_phonemes, y_length_emg, y_emg = emg_tup
        neural, length_neural, neural_phonemes, y_length_neural, y_neural = neural_tup
        audio, length_audio, audio_phonemes, y_length_audio, y_audio = audio_tup
        paired_emg_idx, paired_audio_idx, silent_emg_idx, parallel_emg_idx, parallel_audio_idx = idxs
        
        (emg_pred, neural_pred, audio_pred), \
            (emg_z, neural_z, audio_z), \
            (emg_bz, neural_bz, audio_bz) = self(emg, neural, audio, length_emg, length_neural, length_audio)
        ret = {
            'emg_pred': emg_pred, 'neural_pred': neural_pred, 'audio_pred': audio_pred,
            'emg_z': emg_z, 'neural_z': neural_z, 'audio_z': audio_z,
            'y_emg': y_emg, 'y_neural': y_neural, 'y_audio': y_audio,
            'y_length_emg': y_length_emg, 'y_length_neural': y_length_neural, 'y_length_audio': y_length_audio,
            'length_emg': length_emg, 'length_neural': length_neural, 'length_audio': length_audio,
            'emg_phonemes': emg_phonemes, 'neural_phonemes': neural_phonemes, 'audio_phonemes': audio_phonemes,
            'paired_emg_idx': paired_emg_idx, 'paired_audio_idx': paired_audio_idx,
            'silent_emg_idx': silent_emg_idx,
            'parallel_emg_idx': parallel_emg_idx, 'parallel_audio_idx': parallel_audio_idx,
            'emg_bz': emg_bz, 'neural_bz': neural_bz, 'audio_bz': audio_bz
        }
        return ret

    # TODO: can we simplify this somehow..? 23 required args is a lot
    def calc_loss(self, emg_pred, neural_pred, audio_pred,
                  emg_z, neural_z, audio_z,
                  y_emg, y_neural, y_audio,
                  y_length_emg, y_length_neural, y_length_audio,
                  length_emg, length_neural, length_audio,
                  emg_phonemes, neural_phonemes, audio_phonemes,
                  paired_emg_idx, paired_audio_idx, silent_emg_idx, parallel_emg_idx, parallel_audio_idx,
                  emg_bz, neural_bz, audio_bz):
        # print(f"{torch.concatenate(emg_z).shape=}, {torch.concatenate(audio_z).shape=}, {torch.concatenate(emg_phonemes).shape=}, {torch.concatenate(audio_phonemes).shape=}")
        
        if emg_pred is not None:
            length_emg = [l//8 for l in length_emg] # Gaddy doesn't do this but I think it's necessary
            emg_ctc_loss = self.ctc_loss(emg_pred, y_emg, length_emg, y_length_emg)
        else:
            logging.info("emg_pred is None")
            emg_ctc_loss = 0.
            
        if neural_pred is not None:
            neural_ctc_loss = self.ctc_loss(neural_pred, y_neural, length_neural, y_length_neural)
        else:
            logging.info("neural_pred is None")
            neural_ctc_loss = 0.
        
        if audio_pred is not None:
            audio_ctc_loss = self.ctc_loss(audio_pred, y_audio, length_audio, y_length_audio)
        else:
            logging.info("audio_pred is None")
            audio_ctc_loss = 0.
            
        # TODO: we should refactor into another function...
        # and also figure out how not to write code for cartesian product
        # INFO: this block is for Audio&EMG contrastive loss only
        if emg_z is not None and audio_z is not None:
            # use DTW with parallel audio/emg to align phonemes with silent emg
            silent_e_z = torch.concatenate([emg_z[i] for i in silent_emg_idx])
            parallel_e_z = torch.concatenate([emg_z[i] for i in parallel_emg_idx])
            parallel_a_z = torch.concatenate([audio_z[i] for i in parallel_audio_idx])
            parallel_a_phonemes = torch.concatenate([audio_phonemes[i] for i in parallel_audio_idx])
            
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
            
            alignment = align_from_distances(costs.T.detach().cpu().float().numpy())
            aligned_a_z = parallel_a_z[alignment]
            aligned_a_phonemes = parallel_a_phonemes[alignment]
            # print(f"{silent_e_z.shape=}, {parallel_a_z.shape=}, {parallel_a_phonemes.shape=}," \
            #       f"{len(alignment)=}, {aligned_a_z.shape=}, {aligned_a_phonemes.shape=}")

            
            ###### InfoNCE #####
            # contrastive loss with emg_t, audio_t as positive pairs
            
            matched_e_z = torch.concatenate([*[emg_z[i] for i in paired_emg_idx], silent_e_z])
            matched_a_z = torch.concatenate([*[audio_z[i] for i in paired_audio_idx], aligned_a_z])

            # TODO: need to make a memoizing cos sim function
            emg_audio_contrastive_loss = nobatch_cross_contrastive_loss(matched_e_z, matched_a_z,
                                                                device=self.device)

            ###### Supervised NCE #######
            paired_e_phonemes = [emg_phonemes[i] for i in paired_emg_idx]
            matched_phonemes = torch.concatenate([*paired_e_phonemes, aligned_a_phonemes])
            # for e,a in zip(paired_emg_idx,paired_audio_idx):
            #     print(f"{emg_phonemes[e].shape=}, {emg_z[e].shape=}, {audio_phonemes[a].shape=}, {audio_z[a].shape=}")
            # print(f"{matched_e_z.shape=}, {len(audio_z)=}, {silent_e_z.shape=}, {len(paired_e_phonemes)=}, {len(audio_phonemes)=}, {len(aligned_a_phonemes)=}")
            # # TODO: we are duplicating some audio_z here. need to fix
            # print(f"{matched_e_z.shape=}, {torch.concatenate([*paired_e_phonemes]).shape=})")
            # print(f"{torch.concatenate([*audio_z]).shape=}, {torch.concatenate([*audio_phonemes]).shape=})") 
            # print(f"{silent_e_z.shape=}, {aligned_a_phonemes.shape=}") 
            z = torch.concatenate([matched_e_z, *audio_z])
            z_class = torch.concatenate([matched_phonemes, *audio_phonemes])
            sup_nce_loss = supervised_contrastive_loss(z, z_class, device=self.device)
            # sup_nce_loss = self.supervised_contrastive_loss(z[:, None], z_class)
            # sup_nce_loss = 0.
            
            ###### 
        elif emg_z is not None:
            # INFO: phoneme labels aren't frame-aligned with emg, so we can't use them
            # TODO: try DTW with parallel audio/emg to align phonemes with silent emg
            # z = torch.concatenate(emg_z)
            # z_class = torch.concatenate(emg_phonemes)
            emg_audio_contrastive_loss = 0.
            sup_nce_loss = 0.
        # elif audio_z is not None:
        #     raise NotImplementedError("audio only is not expected")
        #     z = torch.concatenate(audio_z)
        #     z_class = torch.concatenate(audio_phonemes)
        else:
            emg_audio_contrastive_loss = 0.
            sup_nce_loss = 0.
            
        # TODO: add neural sup contrastive loss
        if neural_z is not None:
            pass
        
        # logging.debug(f"{z_class=}")
        
        # assert audio_pred is None, f'Audio only not implemented, got {audio_pred=}'
        logging.debug(f"emg_ctc_loss: {emg_ctc_loss}, audio_ctc_loss: {audio_ctc_loss}, " \
                        f"emg_audio_contrastive_loss: {emg_audio_contrastive_loss}, " \
                        f"sup_nce_loss: {sup_nce_loss}")
        loss = emg_ctc_loss + \
            self.neural_lambda * neural_ctc_loss + \
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
        if not emg_z is None:
            emg_z_mean = torch.concatenate([e.reshape(-1).abs() for e in emg_z]).mean()
        else:
            emg_z_mean = None
        
        if not neural_z is None:
            neural_z_mean = torch.concatenate([n.reshape(-1).abs() for n in neural_z]).mean()
        else:
            neural_z_mean = None
            
        if not audio_z is None:
            audio_z_mean = torch.concatenate([a.reshape(-1).abs() for a in audio_z]).mean()
        else:
            audio_z_mean = None
            
        return {
            'loss': loss,
            'emg_ctc_loss': emg_ctc_loss,
            'neural_ctc_loss': neural_ctc_loss,
            'audio_ctc_loss': audio_ctc_loss,
            'cross_contrastive_loss': emg_audio_contrastive_loss,
            'supervised_contrastive_loss': sup_nce_loss,
            'emg_z_mean': emg_z_mean,
            'neural_z_mean': neural_z_mean,
            'audio_z_mean': audio_z_mean,
            'emg_bz': emg_bz,
            'neural_bz': neural_bz,
            'audio_bz': audio_bz,
            'paired_bz': paired_bz
       }
    
    def _beam_search_batch(self, batch):
        "Repeatedly called by validation_step & test_step."
        
        # TODO: refactor so easy to call on nerual, emg, audio
        X = nn.utils.rnn.pad_sequence(batch['neural_features'], batch_first=True)
        # X = nn.utils.rnn.pad_sequence(batch['raw_emg'], batch_first=True)
        pred  = self.neural_forward(X)[0].cpu()
        # pred  = self.emg_forward(X)[0].cpu()

        beam_results = self.ctc_decoder(pred)
        pred_text = []
        for b in beam_results:
            if len(b) > 0:
                # I think length is zero only when there's NaNs in the output
                # we could just allow the crash here
                pred_text.append(' '.join(b[0].words).strip().lower())
            else:
                pred_text.append('')
        target_text  = [self.text_transform.clean_2(b) for b in batch['text']]
        return target_text, pred_text
    
    def training_step(self, batch, batch_idx):
        c = self.calc_loss(**self.batch_forward(batch))
        loss = c['loss']
        emg_ctc_loss = c['emg_ctc_loss']
        neural_ctc_loss = c['neural_ctc_loss']
        audio_ctc_loss = c['audio_ctc_loss']
        cross_contrastive_loss = c['cross_contrastive_loss']
        sup_contrastive_loss = c['supervised_contrastive_loss']
        emg_bz = c['emg_bz']
        neural_bz = c['neural_bz']
        audio_bz = c['audio_bz']
        paired_bz = c['paired_bz']
        summed_bz = emg_bz + neural_bz + audio_bz + paired_bz
        avg_emg_latent = c['emg_z_mean']
        avg_neural_latent = c['neural_z_mean']
        avg_audio_latent = c['audio_z_mean']
        
        
        self.log("train/loss", loss,
                 on_step=False, on_epoch=True, logger=True, prog_bar=True, batch_size=summed_bz, sync_dist=True)
        self.log("train/emg_ctc_loss", emg_ctc_loss,
            on_step=False, on_epoch=True, logger=True, prog_bar=False, batch_size=emg_bz, sync_dist=True)
        self.log("train/neural_ctc_loss", neural_ctc_loss,
            on_step=False, on_epoch=True, logger=True, prog_bar=False, batch_size=neural_bz, sync_dist=True)
        self.log("train/audio_ctc_loss", audio_ctc_loss,
            on_step=False, on_epoch=True, logger=True, prog_bar=False, batch_size=audio_bz, sync_dist=True)
        self.log("train/cross_contrastive_loss", cross_contrastive_loss,
                 on_step=False, on_epoch=True, logger=True, prog_bar=False, batch_size=paired_bz, sync_dist=True)
        self.log("train/supervised_contrastive_loss", sup_contrastive_loss,
                 on_step=False, on_epoch=True, logger=True, prog_bar=False, batch_size=paired_bz, sync_dist=True)
        if not avg_emg_latent is None:
            self.log("train/avg_emg_latent", avg_emg_latent,
                    on_step=False, on_epoch=True, logger=True, prog_bar=False, batch_size=emg_bz, sync_dist=True)
        if not avg_audio_latent is None:
            self.log("train/avg_audio_latent", avg_audio_latent,
                    on_step=False, on_epoch=True, logger=True, prog_bar=False, batch_size=audio_bz, sync_dist=True)
        if not avg_neural_latent is None:
            self.log("train/avg_neural_latent", avg_neural_latent,
                    on_step=False, on_epoch=True, logger=True, prog_bar=False, batch_size=neural_bz, sync_dist=True)
        torch.cuda.empty_cache()
        return loss

    def validation_step(self, batch, batch_idx, task="val"):
        ret = self.batch_forward(batch)
        c = self.calc_loss(**ret)
        loss = c['loss']
        emg_ctc_loss = c['emg_ctc_loss']
        neural_ctc_loss = c['neural_ctc_loss']
        audio_ctc_loss = c['audio_ctc_loss']
        emg_bz = c['emg_bz']
        neural_bz = c['neural_bz']
        audio_bz = c['audio_bz']
        paired_bz = c['paired_bz']
        summed_bz = emg_bz + neural_bz + audio_bz + paired_bz

        logging.debug(f"validation_step: {batch_idx=}, {loss=}, {emg_ctc_loss=}, {audio_ctc_loss=}, {summed_bz=}")

        # TODO: split text by emg, audio, neural
        target_text = batch['text']
        target_texts, pred_texts = self._beam_search_batch(batch)
        # print(f"text: {batch['text']}; target_text: {target_text}; pred_text: {pred_text}")
        for i, (target_text, pred_text) in enumerate(zip(target_texts, pred_texts)):
            if len(target_text) > 0:
                self.step_target.append(target_text)
                self.step_pred.append(pred_text)
                if i % 16 == 0 and type(self.logger) == NeptuneLogger:
                    # log approx 10 examples
                    self.logger.experiment[f"training/{task}/sentence_target"].append(target_text)
                    self.logger.experiment[f"training/{task}/sentence_pred"].append(pred_text)
            
        self.log(f"{task}/loss", loss, prog_bar=True, batch_size=summed_bz, sync_dist=True)
        self.log(f"{task}/emg_ctc_loss", emg_ctc_loss, prog_bar=False, batch_size=emg_bz, sync_dist=True)
        self.log(f"{task}/neural_ctc_loss", neural_ctc_loss, prog_bar=False, batch_size=neural_bz, sync_dist=True)
        self.log(f"{task}/audio_ctc_loss", audio_ctc_loss, prog_bar=False, batch_size=audio_bz, sync_dist=True)
        
        return loss
    
    def on_validation_epoch_end(self) -> None:
        # TODO: this may not be implemented correctly for DDP
        # raise NotImplementedError("on_validation_epoch_end not implemented neural, librispeech")
        logging.warning(f"start on_validation_epoch_end")
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
        logging.warning(f"on_validation_epoch_end: calc wer")
        wer = jiwer.wer(step_target, step_pred)
        self.step_target.clear()
        self.step_pred.clear()
        self.log("val/wer", wer, prog_bar=True, sync_dist=True)
        # self.profiler.stop(f"validation loop")
        # self.profiler.describe()
        logging.warning(f"on_validation_epoch_end: gc.collect()")
        gc.collect()
        torch.cuda.empty_cache() # TODO: see if fixes occasional freeze...?

    
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx, task="test")

    def log(self, *args, **kwargs):
        if 'batch_size' in kwargs and kwargs['batch_size'] == 0:
            pass
        else:
            super().log(*args, **kwargs)
            
    # def on_before_optimizer_step(self, optimizer):
    #     # Compute the 2-norm for each layer
    #     # If using mixed precision, the gradients are already unscaled here
    #     norms = pl.utilities.grad_norm(self.layer, norm_type=2)
    #     self.log_dict(norms)