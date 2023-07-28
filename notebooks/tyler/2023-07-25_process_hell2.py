import numpy # adds 31 processes
import pytorch_lightning as pl # adds 9 processes
import scipy.signal # adds 31 processes
from time import sleep

print("sleeping....")
sleep(10000)
exit(0)
# process num  | num children
# 1            | 0 
# 2            | 71
# 3            | 71
# 4            | 71
