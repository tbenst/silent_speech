##
import os

import torch, numpy as np
from torch.utils.data import DataLoader, Dataset

from pytorch_lightning import LightningModule, Trainer
from time import sleep


class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


class BoringModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(32, 2)

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        loss = self(batch).sum()
        self.log("train_loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss = self(batch).sum()
        self.log("valid_loss", loss)

    def test_step(self, batch, batch_idx):
        loss = self(batch).sum()
        self.log("test_loss", loss)

    def configure_optimizers(self):
        return torch.optim.SGD(self.layer.parameters(), lr=0.1)

class VarLengthBatchSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, data_source, max_batch_size=None, drop_last=True, seed=12340,
                 batch_size=None):
        """Randomly return batches of length [1,max_batch_size].
        
        Args:
            data_source: a dataset
            max_batch_size: the maximum batch size. This is required
            drop_last: if True, drop the last batch if too small
            seed: random seed
        """
        super().__init__(data_source)
        self.data_len = len(data_source)
        self.max_batch_size = max_batch_size
        self.seed = seed
        # in actual application, need to increment this in the training loop
        self.epoch = 0
        # self.approx_len = int(np.ceil(self.data_len / (self.max_batch_size // 2)))
        self.approx_len = 5 # to dramatically illustrate problem of not updating length
        print(f"approx_len: {self.approx_len}")
        self.len = self.approx_len # pytorch lightning
        self.drop_last = drop_last

    def rand_bz(self):
        return torch.randint(1,self.max_batch_size,(1,))[0]

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(self.data_len, generator=g).tolist()
        batch = [] # accumulate idxs, then reset
        batches = [] # accumulate batches
        bz = self.rand_bz()
        # print(f"\n{bz=}")
        for idx in indices:
            batch.append(idx)
            if len(batch) == bz:
                batches.append(batch)
                bz = self.rand_bz()
                # print(f"\nnew {bz=}")
                batch = []
        if not self.drop_last and len(batch) > 0:
            batches.append(batch)
        
        # update length to match random sampling results
        self.len = len(batches)
        print(f"\nnumber of batches should be: {self.len}")
        return iter(batches)
    
    def __len__(self):
        print(f"\n__len__: {self.len=}")
        return self.len

class PLVarLengthBatchSampler(VarLengthBatchSampler):
    def __init__(self, data_source, batch_size=None, max_batch_size=8, drop_last=True, seed=12340):
        """Wrapper for pytorch lightning compatibility
        """
        print(f"batch_size: {batch_size}")
        super().__init__(data_source, max_batch_size=max_batch_size, drop_last=drop_last, seed=seed)
        self.batch_size = max_batch_size

def run():
    train_dset = RandomDataset(32, 64)
    bz = 8
    train_data = DataLoader(train_dset,
        batch_sampler=PLVarLengthBatchSampler(train_dset, max_batch_size=bz))
    val_data = DataLoader(RandomDataset(32, 64), batch_size=32)
    test_data = DataLoader(RandomDataset(32, 64), batch_size=32)

    model = BoringModel()
    trainer = Trainer(
        # devices=1, # uses first __len__ call for all lengths
        devices=2, # uses approx_len for all lengths
        default_root_dir=os.getcwd(),
        num_sanity_val_steps=0,
        max_epochs=5,
        enable_model_summary=False
    )
    trainer.fit(model, train_dataloaders=train_data, val_dataloaders=val_data)
    trainer.test(model, dataloaders=test_data)

run()
##
