from model import LitCNN
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import pytorch_lightning as pl
from data_module import MNISTDataModule

mnistDM = MNISTDataModule()
mnistDM.prepare_data()
mnistDM.setup('fit')

m = LitCNN()
trainer = pl.Trainer(max_epochs=1)
trainer.fit(model=m, datamodule=mnistDM)

trainer.test(model=m, datamodule=mnistDM)
