import os
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy # adds 31 processes
import pytorch_lightning as pl # adds 9 processes
import scipy.signal # adds 31 processes
from time import sleep

print("sleeping....")
sleep(10000)
exit(0)