import torch
from torch import nn
from tqdm.notebook import tqdm

class MyLightningModule(nn.Module):
  def __init__(self):
    super().__init__()

  def training_step(self, batch):
    return

  def validation_step(self, batch):
    return

  def configure_optimizers(self):
    return


class Trainer:
  def __init__(self, max_epochs):
    self.max_epochs = max_epochs

  def fit(self, model, trainloader, valloader=None):
    opt = model.configure_optimizers()
    epoch_tqdm = tqdm(range(self.max_epochs), total=self.max_epochs, leave=False)

    for epoch in epoch_tqdm:

      for train_batch in tqdm(trainloader, total=len(trainloader), leave=False):
        opt.zero_grad()
        train_out = model.training_step(train_batch)
        opt.step()
      if valloader:
        with torch.no_grad():
          for val_batch in valloader:
            val_out = model.validation_step(val_batch)

      print(train_out, val_out)