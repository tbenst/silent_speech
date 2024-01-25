import os
import signal
import subprocess
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from time import sleep


# Dummy dataset
class DummyDataset(Dataset):
    def __len__(self):
        return 1000

    def __getitem__(self, idx):
        # Generate random input and a single integer as target
        return torch.randn(10), torch.randint(0, 2, ())


# Dummy model
class DummyModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(10, 2)

    def forward(self, x):
        sleep(0.1)  # Simulating some processing
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        # y should be a 1D tensor of class indices
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


# Function to resubmit the job with the same run_id
def resubmit_job(sig, frame):
    print("Received SIGUSR1. Resubmitting job.")
    subprocess.run(["echo", "sbatch", "/path/to/your/script.sh"])
    exit(0)


# Setup signal handling for SIGUSR1
signal.signal(signal.SIGUSR1, resubmit_job)

# Training
dataset = DummyDataset()
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = DummyModel()
trainer = pl.Trainer(max_epochs=10)
trainer.fit(model, train_loader)

print("Training complete.")
