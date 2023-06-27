import random
import torch
from torch import nn
import torch.nn.functional as F
from transformer import TransformerEncoderLayer
from data_utils import combine_fixed_length, decollate_tensor

import sys, os, jiwer
import pytorch_lightning as pl
from torchaudio.models.decoder import ctc_decoder
from s4 import S4
from data_utils import TextTransform
from magneto.models.hyena import HyenaOperator
from flash_attn.modules.block import Block
from magneto.models.s4d import S4D

from pytorch_lightning.profilers import PassThroughProfiler
from dataclasses import dataclass
from typing import Tuple

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
        self.act = nn.GELU()
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
                 learning_rate_warmup = 1000, profiler = None):
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
        # Only index once. 
        target_text  = self.text_transform.clean_2(batch['text'][0])

        return target_text, pred_text
    
    def on_train_epoch_start(self):
        # bad separation of concerns / composability,
        # but this seems forced by pytorch lightning
        # maybe should use Fabric in the future..
        if self.trainer.datamodule is not None:
            if hasattr(self.trainer.datamodule, 'TrainSampler'):
                self.trainer.datamodule.TrainSampler.set_epoch(self.current_epoch)
    
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
        self.log("val/wer", wer, prog_bar=True, sync_dist=True)
        # self.profiler.stop(f"validation loop")
        # self.profiler.describe()
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
        optimizer = torch.optim.AdamW(self.trainer.model.parameters(), lr=initial_lr)

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

        if self.global_step <= self.learning_rate_warmup:
            new_lr = self.global_step*self.target_lr/self.learning_rate_warmup
            self.set_lr(new_lr)
        else:
            if metric is None:
                scheduler.step()
            else:
                scheduler.step(metric)

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
        