import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Tuple

class LitAE(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
        )

        self.decoder = nn.Sequential(
            nn.Linear(32, 256),
            nn.ReLU(),
            nn.Linear(256,512),
            nn.ReLU(),
            nn.Linear(512, 28*28),
        )
        self.criterion = nn.MSELoss()

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
    
    def training_step(self: pl.LightningModule, batch: Tuple[torch.Tensor, torch.Tensor]):
        x, _ = batch
        # print(x)
        x = x.view(-1, 28*28)
        target = x.view(-1, 28*28)
        _, decoded = self.forward(x)
        loss = self.criterion(decoded, target)

        return loss

    def evaluate(self, batch, stage=None):
        x, _ = batch
        x = x.view(-1, 28*28)
        target = x.view(-1, 28*28)
        _, decoded = self.forward(x)
        loss = self.criterion(decoded, target)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            # self.log(f"{stage}_acc", acc, prog_bar=True)

    def validation_step(self: pl.LightningModule, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        self.evaluate(batch, "val")

    def test_step(self: pl.LightningModule, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

