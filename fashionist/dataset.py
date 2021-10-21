import numpy as np
from torch.utils.data import Dataset

from fashionist.constants import STORAGE_DIR


class FashionDS(Dataset):
    index = None

    label_list = [
        'T-shirt/top',
        'Trouser',
        'Pullover',
        'Dress',
        'Coat',
        'Sandal',
        'Shirt',
        'Sneaker',
        'Bag',
        'Ankle boot'
    ]

    all_labels = np.frombuffer(
        STORAGE_DIR.joinpath('t10k-labels-idx1-ubyte').open('rb').read(),
        dtype=np.uint8,
        offset=8,
    )
    all_images = np.frombuffer(
        STORAGE_DIR.joinpath('t10k-images-idx3-ubyte').open('rb').read(),
        dtype=np.uint8,
        offset=16,
    ).reshape(len(all_labels), 784)

    def __init__(self, train=True):
        self.images = FashionDS.all_images
        self.labels = FashionDS.all_labels
        self.train = train
        self.train_fraction = len(self.labels) * 8 // 10
        if FashionDS.index is None:
            FashionDS.index = np.random.permutation(np.arange(len(self.labels)))

    def __len__(self):
        if self.train:
            return self.train_fraction
        else:
            return len(self.labels) - self.train_fraction

    def __getitem__(self, idx):
        if self.train:
            idx2 = FashionDS.index[idx]
        else:
            idx2 = FashionDS.index[self.train_fraction + idx]
        return self.images[idx2].reshape(28, 28), self.labels[idx2]
