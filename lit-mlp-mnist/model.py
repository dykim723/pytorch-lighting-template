import torch
from torch import nn
import torch.nn.functional as F
from torchmetrics.functional import accuracy
import pytorch_lightning as pl
from typing import Tuple

class LitCNN(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        self.criterion = F.nll_loss
    
    def forward(self, x):
        batch_size, channel, width, height = x.size()
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x

    def training_step(self: pl.LightningModule, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, y = batch
        logit = self.forward(x)
        loss = self.criterion(logit, y)
        return loss
    
    def evaluate(self, batch, stage=None):
        
        x, y = batch
        print(torch.tensor(y))
        logits = self.forward(x)
        print(logits.shape)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

    def validation_step(self: pl.LightningModule, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        self.evaluate(batch, "val")

    def test_step(self: pl.LightningModule, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-3)
        return optimizer
    
    def predict_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        return pred