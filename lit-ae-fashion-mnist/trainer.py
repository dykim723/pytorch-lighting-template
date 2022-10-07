from model import LitAE
import pytorch_lightning as pl
from data_module import FashionMNISTDataModule

mnistDM = FashionMNISTDataModule()
mnistDM.prepare_data()
mnistDM.setup('fit')

m = LitAE()
trainer = pl.Trainer(max_epochs=1)
trainer.fit(model=m, datamodule=mnistDM)

trainer.test(model=m, datamodule=mnistDM)