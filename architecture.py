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

MODEL_SIZE = 768 # number of hidden dimensions
NUM_LAYERS = 6 # number of layers
DROPOUT = .2 # dropout

LM_DIR = "/oak/stanford/projects/babelfish/magneto/GaddyPaper/pretrained_models/librispeech_lm/"

class ResBlock(nn.Module):
    def __init__(self, num_ins, num_outs, stride=1):
        super().__init__()

        self.conv1 = nn.Conv1d(num_ins, num_outs, 3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm1d(num_outs)
        self.conv2 = nn.Conv1d(num_outs, num_outs, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(num_outs)

        if stride != 1 or num_ins != num_outs:
            self.residual_path = nn.Conv1d(num_ins, num_outs, 1, stride=stride)
            self.res_norm = nn.BatchNorm1d(num_outs)
        else:
            self.residual_path = None

    def forward(self, x):
        input_value = x

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        if self.residual_path is not None:
            res = self.res_norm(self.residual_path(input_value))
        else:
            res = input_value

        return F.relu(x + res)
    
    
class Model(pl.LightningModule):
    def __init__(self, num_features, model_size, dropout, num_layers, num_outs, text_transform: TextTransform,
                 steps_per_epoch, epochs, num_aux_outs=None, lr=3e-4,
                 lm_directory=LM_DIR):
        super().__init__()

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
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        
        # val/test procedure...
        self.text_transform = text_transform
        self.n_chars = len(text_transform.chars)
        lexicon_file = os.path.join(lm_directory, 'lexicon_graphemes_noApostrophe.txt')
        self.ctc_decoder = ctc_decoder(
            lexicon = lexicon_file,
            tokens  = text_transform.chars + ['_'],
            lm      = os.path.join(lm_directory, '4gram_lm.bin'),
            blank_token = '_',
            sil_token   = '|',
            nbest       = 1,
            lm_weight   = 2, # default is 2; Gaddy sets to 1.85
            #word_score  = -3,
            #sil_score   = -2,
            beam_size   = 150  # SET TO 150 during inference
        )

    def forward(self, x_feat, x_raw, session_ids):
        # x shape is (batch, time, electrode)

        if self.training:
            r = random.randrange(8)
            if r > 0:
                x_raw[:,:-r,:] = x_raw[:,r:,:] # shift left r
                x_raw[:,-r:,:] = 0
        
        x_raw = x_raw.transpose(1,2) # put channel before time for conv
        x_raw = self.conv_blocks(x_raw)
        x_raw = x_raw.transpose(1,2)
        x_raw = self.w_raw_in(x_raw)

        x = x_raw
        x = x.transpose(0,1) # put time first
        x = self.transformer(x)
        x = x.transpose(0,1)

        if self.has_aux_out:
            return self.w_out(x), self.w_aux(x)
        else:
            return self.w_out(x)
        
        
    def calc_loss(self, batch):
        X     = combine_fixed_length(batch['emg'], self.seqlen)
        X_raw = combine_fixed_length(batch['raw_emg'], self.seqlen*8)
        sess  = combine_fixed_length(batch['session_ids'], self.seqlen)        
        bz = X.shape[0]
    
        pred = self(X, X_raw, sess)
        pred = F.log_softmax(pred, 2)

        # seq first, as required by ctc
        pred = nn.utils.rnn.pad_sequence(decollate_tensor(pred, batch['lengths']), batch_first=False) 
        y    = nn.utils.rnn.pad_sequence(batch['text_int'], batch_first=True)
        loss = F.ctc_loss(pred, y, batch['lengths'], batch['text_int_lengths'], blank=self.n_chars)
        
        if torch.isnan(loss) or torch.isinf(loss):
            # print('batch:', batch_idx)
            print('Isnan output:',torch.any(torch.isnan(pred)))
            print('Isinf output:',torch.any(torch.isinf(pred)))
            raise ValueError("NaN/Inf detected in loss")
            
        return loss, bz
    
    def calc_wer(self, batch):
        X     = batch['emg'][0].unsqueeze(0)
        X_raw = batch['raw_emg'][0].unsqueeze(0)
        sess  = batch['session_ids'][0]

        pred  = F.log_softmax(self(X, X_raw, sess), -1).cpu()

        beam_results = self.ctc_decoder(pred)
        pred_int     = beam_results[0][0].tokens
        pred_text    = ' '.join(beam_results[0][0].words).strip().lower()
        target_text  = self.text_transform.clean_2(batch['text'][0][0])
        return jiwer.wer([target_text], [pred_text]), X.shape[0]
    
    def training_step(self, batch, batch_idx):
        loss, bz = self.calc_loss(batch)
        self.log("train/loss", loss, on_step=False, on_epoch=True, logger=True, prog_bar=True, batch_size=bz)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # loss = self.calc_loss(batch)
        wer, bz = self.calc_wer(batch)
        # self.log("val/loss", loss, prog_bar=True)
        self.log("val/wer", wer, prog_bar=True, batch_size=bz)
        return wer

    def test_step(self, batch, batch_idx):
        # loss = self.calc_loss(batch)
        wer, bz = self.calc_wer(batch)
        # self.log("test/loss", loss, prog_bar=True)
        self.log("test/wer", wer, prog_bar=True, batch_size=bz)
        return wer

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.lr,
            steps_per_epoch=self.steps_per_epoch, epochs=self.epochs)
        lr_scheduler = {'scheduler': scheduler, 'interval': 'step'}

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
        
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

            
    def forward(self, x_feat, x_raw, session_ids):
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
from h3 import H3
        

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

            
    def forward(self, x_feat, x_raw, session_ids):
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
        