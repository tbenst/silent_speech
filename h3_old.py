import torch
import torch.nn as nn
from torch.fft import rfft, irfft
from einops import rearrange

from s4 import S4




class H3(nn.Module):
    """The Hungry Hungry Hippos (H3) layer.
    Args:
        d_model: The number of heads.
        d_state: The dimension of the state matrix.
        kernel_args: Arguments to pass to the S4D kernel.
    """

    def __init__(self, d_model, d_state=64, **kernel_args):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.q_proj  = nn.Linear(d_model, d_model)
        self.k_proj  = nn.Linear(d_model, d_model)
        self.v_proj  = nn.Linear(d_model, d_model)
        #self.s4d     = S4D(d_model, d_state, **kernel_args)
        #self.s4d     = S4(d_model, d_state, mode = 'diag', measure = "diag-lin", **kernel_args)
        self.s4d     = S4(d_model, d_state, **kernel_args)
        self.shift   = Shift(d_model, d_state)

    def forward(self, x):
        '''Input is (B, H, L)'''
     
        q = self.q_proj(x.transpose(-1, -2)).transpose(-1, -2)
        k = self.k_proj(x.transpose(-1, -2)).transpose(-1, -2)
        v = self.v_proj(x.transpose(-1, -2)).transpose(-1, -2)
        
        #q = rearrange(self.q_proj(x.transpose(-1, -2)).transpose(-1, -2), "b h l  ->  h l b")
        #k = rearrange(self.k_proj(x.transpose(-1, -2)).transpose(-1, -2), "b h l  ->  h l b")
        #v = rearrange(self.v_proj(x.transpose(-1, -2)).transpose(-1, -2), "b h l  ->  h l b")
        
        #print(k.shape)
        # ORIG:
        #q = rearrange(self.q_proj(x), "b h l  ->  h l b")
        #k = rearrange(self.k_proj(x), "b h l  ->  h l b")
        #v = rearrange(self.v_proj(x), "b h l  ->  h l b")
        shift_out = self.shift(k)
        #s4d_out   = self.s4d(v * shift_out)
        s4d_out,_ = self.s4d(v * shift_out)
        #out       = rearrange(q * s4d_out, "h l b -> b h l")
        out = q * s4d_out
        return out
    
    
    
class Shift(nn.Module):
    """The shift state space layer.
    Args:
        d_model: The number of heads.
        d_state: The dimension of the state matrix.

    The shift layer is a special case of an SSM layer with a fixed A matrix that
    allows tokens to mix with past tokens. For d_state = 4, the A matrix would be:

    A = [[0, 0, 0, 0],
         [1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 1, 0]]
    """

    def __init__(self, d_model, d_state):
        super().__init__()
        self.d_state = d_state
        self.d_model = d_model
        self.B       = torch.zeros(d_model, d_state).cuda()
        self.B[..., 0] = 1.0
        self.C = nn.Parameter(torch.randn(1, d_model, d_state)).cuda()
        self.D = nn.Parameter(torch.randn(d_model)).cuda()

    def forward(self, u):
        """Input and output shape (B, H, L)"""

        L = u.size(-1)

        # Construct kernel
        B_fc = rfft(self.B, n=2 * L).conj()
        C_f = rfft(self.C, n=2 * L)
        kernel = irfft(B_fc * C_f, n=2 * L)
        #kernel = rearrange(kernel[..., :L], "b h l -> b l h") # orig
        kernel = kernel[..., :L]

        # Perform convolution by kernel
        kernel_f = rfft(kernel, n=2 * L)
        u_f = rfft(u, n=2 * L)
        y = irfft(u_f * kernel_f, n=2 * L)

        # Truncate to original length and add skip connection
        y = y[..., :L] + u * self.D[:, None]
        return y
