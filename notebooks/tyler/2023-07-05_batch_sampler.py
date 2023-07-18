##
import os, sys, torch.distributed as dist
# horrible hack to get around this repo not being a proper python package
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(SCRIPT_DIR)

from dataloaders import SizeAwareStratifiedBatchSampler, DistributedSizeAwareStratifiedBatchSampler
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

##
class TestDataset(Dataset):
    def __init__(self, classes:np.ndarray, lengths:np.ndarray):
        self.classes = classes
        self.lengths = lengths
        
    def __len__(self):
        return len(self.classes)
    
    def __getitem__(self, idx):
        return self.classes[idx], self.lengths[idx]
    
def test_SizeAwareStratifiedBatchSampler():
    classes = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2] * 10)
    lengths = np.array([10, 20, 10, 40, 10, 20, 10, 50, 10] * 10)
    class_proportion = np.array([0.5, 0.25, 0.25])
    batch_size = 4
    max_len = 100
    
    dataset = TestDataset(classes, lengths)
    sampler = SizeAwareStratifiedBatchSampler(classes, lengths, class_proportion, batch_size, max_len, shuffle=False)
    dataloader = DataLoader(dataset, batch_sampler=sampler)
    for batch in dataloader:
        batch_classes, batch_lengths = batch
        tot_len = batch_lengths.sum()
        print(f'batch_classes: {batch_classes}, batch_lengths: {batch_lengths}, tot_len: {tot_len}')
        
test_SizeAwareStratifiedBatchSampler()
##
def test_DistributedSizeAwareStratifiedBatchSampler():
    classes = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2] * 10)
    lengths = np.array([10, 20, 10, 40, 10, 20, 10, 50, 10] * 10)
    class_proportion = np.array([0.5, 0.25, 0.25])
    batch_size = 8
    max_len = 100
    
    dataset = TestDataset(classes, lengths)
    sampler = DistributedSizeAwareStratifiedBatchSampler(classes, lengths,
        class_proportion, batch_size, max_len, shuffle=False, num_replicas=2)
    dataloader = DataLoader(dataset, batch_sampler=sampler)
    for batch in dataloader:
        batch_classes, batch_lengths = batch
        tot_len = batch_lengths.sum()
        print(f'rank: {sampler.rank}, batch_classes: {batch_classes}, batch_lengths: {batch_lengths}, tot_len: {tot_len}')
    
# do torchrun --standalone --nnodes=1 --nproc_per_node=2 2023-07-05_batch_sampler.py
dist.init_process_group()
test_DistributedSizeAwareStratifiedBatchSampler()
##
