import random
import torch
from torch import nn
import torch.nn.functional as F
from transformer import TransformerEncoderLayer

import sys
from s4 import S4

from absl import flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('model_size', 768, 'number of hidden dimensions')
flags.DEFINE_integer('num_layers', 6, 'number of layers')
flags.DEFINE_float('dropout', .2, 'dropout')

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
    
    
class Model(nn.Module):
    def __init__(self, num_features, num_outs, num_aux_outs=None):
        super().__init__()

        self.conv_blocks = nn.Sequential(
            ResBlock(8, FLAGS.model_size, 2),
            ResBlock(FLAGS.model_size, FLAGS.model_size, 2),
            ResBlock(FLAGS.model_size, FLAGS.model_size, 2),
        )
        self.w_raw_in = nn.Linear(FLAGS.model_size, FLAGS.model_size)
        encoder_layer = TransformerEncoderLayer(d_model=FLAGS.model_size, nhead=8, relative_positional=True, 
                                                relative_positional_distance=100, dim_feedforward=3072, dropout=FLAGS.dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, FLAGS.num_layers)
        self.w_out       = nn.Linear(FLAGS.model_size, num_outs)

        self.has_aux_out = num_aux_outs is not None
        if self.has_aux_out:
            self.w_aux = nn.Linear(FLAGS.model_size, num_aux_outs)

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
            nn.Linear(8, FLAGS.model_size),
            nn.Softsign(),
            nn.Linear(8, FLAGS.model_size)
        )
        
        
        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms     = nn.ModuleList()
        self.dropouts  = nn.ModuleList()
        self.linears   = nn.ModuleList()
        for i in range(FLAGS.num_layers):
            if i > 2:
                s4_dropout = FLAGS.dropout
            #   # channels = 2
            #else:
                s4_dropout = 0
            #  #  channels = 3
            
            s4_dropout = FLAGS.dropout
            
            dropout = FLAGS.dropout
            self.s4_layers.append(S4Layer(FLAGS.model_size, dropout, s4_dropout = s4_dropout))
        
        self.w_out = nn.Linear(FLAGS.model_size, num_outs)
        
        self.has_aux_out = num_aux_outs is not None
        if self.has_aux_out:
            self.w_aux = nn.Linear(FLAGS.model_size, num_aux_outs)

            
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
        self.encoder = nn.Linear(8, FLAGS.model_size)
        
        # Stack S4 layers as residual blocks
        self.h3_layers = nn.ModuleList()
        self.norms     = nn.ModuleList()
        self.dropouts  = nn.ModuleList()
        self.linears   = nn.ModuleList()
        for i in range(FLAGS.num_layers):
            self.h3_layers.append(
                H3(d_model = FLAGS.model_size, dropout=FLAGS.dropout, lr=None)
            )
            self.norms.append(nn.LayerNorm(FLAGS.model_size))
            self.dropouts.append(nn.Dropout1d(FLAGS.dropout))
        
        self.w_out = nn.Linear(FLAGS.model_size, num_outs)
        
        self.has_aux_out = num_aux_outs is not None
        if self.has_aux_out:
            self.w_aux = nn.Linear(FLAGS.model_size, num_aux_outs)

            
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
        