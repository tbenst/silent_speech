# import pytorch_lightning  # <--- Comment or uncomment to test
# import torch  # <--- Comment or uncomment to test

import numpy # adds 31 processes
import pytorch_lightning as pl # adds 9 processes
import scipy.signal # adds 31 processes

import os
import torch.multiprocessing as multiprocessing

# pool = multiprocessing.Pool(8)
# print(pool.map(os.sched_getaffinity, [0] * 8))
affinity = os.sched_getaffinity(0)
print(f"{affinity=}")