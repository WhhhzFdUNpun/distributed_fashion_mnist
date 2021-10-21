import pytorch_lightning as pl

from fashionist.model import Net


def main(gpus=1, num_nodes=1):
    net = Net()
    trainer = pl.Trainer(
        max_epochs = 15,
        num_nodes=num_nodes,
        gpus=gpus,
        accelerator="ddp")
    trainer.fit(net)


if __name__ == '__main__':
    main()
