##
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch, torchmetrics, sys
import torch.nn.functional as F
import numpy as np
# horrible hack to get around this repo not being a proper python package
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(SCRIPT_DIR)

from contrastive import cross_contrastive_loss, var_length_cross_contrastive_loss, \
    nobatch_cross_contrastive_loss, supervised_contrastive_loss, infoNCE_masks
    

def benchmark(func):
    def wrapper(*args, **kwargs):
        torch.cuda.reset_max_memory_allocated()
        t = timeit.timeit(lambda: func(*args, **kwargs), number=100)
        mem = torch.cuda.max_memory_allocated()
        print(f"{func.__name__}: {t:.6f} seconds, {mem/1024/1024:.6f} MB")
    return wrapper

##
import timeit

@benchmark
def cm1(x,y):
    representations = torch.einsum('ld,md->lm', x, y)

@benchmark
def cm2(x,y):
    representations = torch.cat([x,y], dim=0)

L = 4096
x = torch.rand(L,256).cuda()
y = torch.rand(L,256).cuda()

cm1(x,y)
cm2(x,y) # faster

##
import torch, torchmetrics
import torch.nn.functional as F

def pairwise_cos_sim(x,y):
    return torch.nn.functional.cosine_similarity(x[:,:,None], y.t()[None,:,:])  

benchmark(torchmetrics.functional.pairwise_cosine_similarity)(x, y) # faster
# benchmark(pairwise_cos_sim)(x, y) # really bad

##
benchmark(infoNCE_masks)(1024, 'cuda')

##
