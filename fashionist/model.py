import pytorch_lightning as pl
import torch
import torch.nn.functional
from torch.utils.data import DataLoader

from fashionist.dataset import FashionDS


class Net(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 5, (3, 3))
        self.linear = torch.nn.Linear(26 * 26 * 5, 10)

    def forward(self, x):
        out = self.conv(x)
        out = torch.nn.functional.relu(out)
        out = out.view(-1, 26 * 26 * 5)
        out = self.linear(out)
        return out

    def training_step(self, batch, batch_id):
        x, y = batch
        x = x.float() / 255.
        yhat = self(x[:, None, ...])
        acc = (yhat.argmax(dim=1) == y).sum() * 1.0 / len(y)
        loss = torch.nn.functional.cross_entropy(yhat, y.long())
        self.log('train loss', loss.item())
        self.log('train acc', acc)
        return loss

    def configure_optimizers(self):
        self.lr = 3e-4
        opt = torch.optim.SGD(self.parameters(), self.lr, 0.9)
        sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[5, 10], gamma=0.1)
        return {
            'optimizer': opt,
            'lr_scheduler': sched,
        }

    def train_dataloader(self):
        return DataLoader(FashionDS(), batch_size=64)