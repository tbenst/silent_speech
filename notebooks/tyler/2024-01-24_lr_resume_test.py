##
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers.logger import DummyLogger
import os
from datetime import datetime
from pytorch_lightning.callbacks import ModelCheckpoint


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
        self.warmup_steps = 3
        self.lr = 0.1

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        loss = self(batch).sum()

        print(
            f"lr: {self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0]:.4f}",
            f"lr_opt: {self.trainer.optimizers[0].param_groups[0]['lr']:.4f}",
        )
        self.log("train/loss", loss, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.layer.parameters(), lr=self.lr)
        # Warmup steps and epochs
        milestone_epochs = [4, 6]

        # Total number of steps for training
        total_steps = self.trainer.estimated_stepping_batches
        print(f"DEBUG: {total_steps=}")

        # Define the lambda function for learning rate schedule
        lr_lambda = (
            lambda step: min(1.0, step / self.warmup_steps)
            if step < self.warmup_steps
            else 0.5
            ** len(
                [
                    m
                    for m in milestone_epochs
                    if m * total_steps // self.trainer.max_epochs <= step
                ]
            )
        )

        # Scheduler with linear warmup and decay at specified epochs
        scheduler = LambdaLR(optimizer, lr_lambda)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]


def run():
    train_data = DataLoader(RandomDataset(32, 64), batch_size=8)
    isotime = datetime.now().isoformat()
    output_directory = os.path.join(f"/tmp/{isotime}")

    logger = DummyLogger()
    model = BoringModel()

    callbacks = [
        ModelCheckpoint(
            monitor="train/loss",
            mode="min",
            dirpath=output_directory,
            save_top_k=1,  # TODO: try averaging weights afterwards to see if improve WER..?
            filename=model.__class__.__name__ + "-{epoch:02d}-{train/loss:.3f}",
        )
    ]

    trainer = Trainer(
        max_epochs=3,
        enable_model_summary=False,
        enable_progress_bar=False,
        default_root_dir=output_directory,
        logger=logger,
        callbacks=callbacks,
    )

    trainer.fit(model, train_dataloaders=train_data)

    scheduler = model.lr_schedulers()
    print(
        f"Scheduler state dict after saving:",
        scheduler.state_dict(),
    )

    print(f"Resume, {trainer.checkpoint_callback.best_model_path=}")

    model = BoringModel()
    trainer2 = Trainer(
        max_epochs=7,
        enable_model_summary=False,
        enable_progress_bar=False,
        default_root_dir=output_directory,
        logger=logger,
    )
    trainer2.fit(
        model,
        train_dataloaders=train_data,
        ckpt_path=trainer.checkpoint_callback.best_model_path,
    )
    
    scheduler = model.lr_schedulers()
    print(
        f"Scheduler state dict after load:",
        scheduler.state_dict(),
    )
    print(f"Finished, {trainer2.checkpoint_callback.best_model_path=}")


if __name__ == "__main__":
    run()

##
