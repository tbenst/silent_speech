import os

import torch
from torch.utils.data import DataLoader, Dataset

from pytorch_lightning import LightningModule, Trainer

from pytorch_lightning.loggers import NeptuneLogger
import neptune.new as neptune, shutil


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
        print(f"{batch_idx=}")
        if batch_idx % 2 == 0:
            pass
            # self.logger.experiment["val/sentence_pred"].append("It was a dark and stormy night.")
        else:
            pass
            # self.logger.experiment["val/sentence_pred"].append("Once upon a time, in a far away land.")
        self.log("valid_loss", loss)

    def test_step(self, batch, batch_idx):
        loss = self(batch).sum()
        self.log("test_loss", loss)

    def configure_optimizers(self):
        return torch.optim.SGD(self.layer.parameters(), lr=0.1)


def run():
    train_data = DataLoader(RandomDataset(32, 64), batch_size=2)
    val_data = DataLoader(RandomDataset(32, 64), batch_size=2)
    test_data = DataLoader(RandomDataset(32, 64), batch_size=2)
    model = BoringModel()
    
    # neptune_logger = NeptuneLogger(
    #     # need to store credentials in your shell env
    #     api_key=os.environ["NEPTUNE_API_TOKEN"],
    #     project="neuro/Gaddy",
    #     # name=magneto.fullname(model), # from lib
    #     name=model.__class__.__name__,
    #     tags=[model.__class__.__name__],
    #     log_model_checkpoints=False,
    # )
    
    neptune_logger = None


    trainer = Trainer(
        accelerator="cpu",
        default_root_dir=os.getcwd(),
        limit_train_batches=2,
        limit_val_batches=2,
        limit_test_batches=2,
        num_sanity_val_steps=0,
        max_epochs=2,
        logger=neptune_logger,
        enable_model_summary=False,
    )
    trainer.fit(model, train_dataloaders=train_data, val_dataloaders=val_data)
    trainer.test(model, dataloaders=test_data)


if __name__ == "__main__":
    run()