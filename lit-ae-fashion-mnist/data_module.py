import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
from pytorch_lightning.utilities.types import (
    EVAL_DATALOADERS,
    TRAIN_DATALOADERS,
)
from torchvision import transforms, datasets

class FashionMNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str="./"):
        super().__init__()
        self.data_dir = "../data/FashionMNIST"
        
        
    def prepare_data(self) -> None:
        datasets.FashionMNIST(self.data_dir, train=True, download=True)
        datasets.FashionMNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        if stage == "fit":
            mnist_full = datasets.FashionMNIST(root="../data/FashionMNIST", train=True, download=True, transform=transforms.ToTensor())
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])
        
        if stage == "test":
            self.mnist_test = datasets.FashionMNIST(root="../data/FashionMNIST", train=False, transform=transforms.ToTensor())

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.mnist_train, batch_size=32)
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.mnist_val, batch_size=32)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.mnist_test, batch_size=32)