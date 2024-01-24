##
import torch, os
import pytorch_lightning as pl
from transformers import get_constant_schedule_with_warmup
from torch.optim.lr_scheduler import MultiStepLR, ChainedScheduler, LambdaLR
from pytorch_lightning.loggers import NeptuneLogger


class DummyModel(pl.LightningModule):
    def __init__(self, learning_rate=0.001, weight_decay=0.0):
        super().__init__()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        # A simple linear layer model
        self.linear = torch.nn.Linear(10, 2)

    def forward(self, x):
        return self.linear(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        # Warmup steps and epochs
        warmup_steps = 1000
        milestone_epochs = [125, 150, 175]

        # Total number of steps for training
        total_steps = self.trainer.estimated_stepping_batches

        # Define the lambda function for learning rate schedule
        lr_lambda = (
            lambda step: min(1.0, step / warmup_steps)
            if step < warmup_steps
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

    def train_dataloader(self):
        # Creating a dummy dataloader
        train_data = torch.randn(1000, 10)
        train_labels = torch.randn(1000, 2)
        train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
        return torch.utils.data.DataLoader(train_dataset, batch_size=32)


# Training the model
model = DummyModel()
neptune_logger = NeptuneLogger(
    api_key=os.environ["NEPTUNE_API_TOKEN"],
    project="neuro/Gaddy",
    name=model.__class__.__name__,
    log_model_checkpoints=False,
)
trainer = pl.Trainer(
    max_epochs=200,
    logger=neptune_logger,
    callbacks=[
        # pl.callbacks.LearningRateMonitor(logging_interval="step")
        pl.callbacks.LearningRateMonitor(logging_interval="epoch")
    ],
)
trainer.fit(model)
neptune_logger.finalize("success")
neptune_logger.experiment.stop()
##