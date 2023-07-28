##
import os, numpy as np, sys, torch
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(SCRIPT_DIR)
# import torch.utils.benchmark
import timeit

from align import align_from_distances
from fastdtw import dtw

def bench(func, number=100):
    def wrapper(*args, **kwargs):
        torch.cuda.reset_max_memory_allocated()
        t = timeit.timeit(lambda: func(*args, **kwargs), number=number)
        mem = torch.cuda.max_memory_allocated()
        print(f"{func.__name__}: {t:.6f} seconds, {mem/1024/1024:.6f} MB")
    return wrapper

##
L = 4096
xt = torch.rand(L,256).cuda()
yt = torch.rand(L,256).cuda()
x = np.random.rand(L,256)
y = np.random.rand(L,256)

# dynamic time warping of x onto y
# distance, path = dtw(x, y)
# bench(dtw, 10)(x, y) # mega slow

##
costs = torch.cdist(xt, yt).squeeze(0)
alignment = bench(align_from_distances)(costs.T.detach().cpu().numpy())

##
# TODO: write our own DTW in CUDA..?
# https://github.com/Maghoumi/pytorch-softdtw-cuda/blob/master/soft_dtw_cuda.py
