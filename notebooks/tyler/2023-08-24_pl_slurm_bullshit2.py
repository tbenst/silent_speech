import os

import torch
from torch.utils.data import DataLoader, Dataset

from pytorch_lightning import LightningModule, Trainer, LightningDataModule
from pytorch_lightning.strategies import DDPStrategy

class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len
    
class DataModule(LightningDataModule):
    def __init__(self, size, length, batch_size):
        super().__init__()
        
        self.train = RandomDataset(size, length)
        self.val = RandomDataset(size, length)
        self.test = RandomDataset(size, length)
        self.batch_size = batch_size
        
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
                self.train,
                pin_memory=True,
                num_workers=0,
                batch_size=self.batch_size,
            )
        
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val,
            pin_memory=True,
            num_workers=0,
            batch_size=self.batch_size,
        )
        
    def test_dataloader(self):
        return None


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


def run():
    dm = DataModule(32, 8780, batch_size=16)

    model = BoringModel()
    trainer = Trainer(
        devices=2,
        default_root_dir=os.getcwd(),
        num_sanity_val_steps=0,
        max_epochs=1,
        strategy=DDPStrategy(gradient_as_bucket_view=True, find_unused_parameters=True)
    )
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    run()