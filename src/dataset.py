"""MNIST with a random rectangular patch masked out, for inpainting."""
import random

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class MaskedMNIST(Dataset):
    """Wraps torchvision's MNIST and masks a random square patch per image.

    Unlike masking a fixed, centered region, the patch location (and
    optionally its size) is randomized per sample, so the model has to infer
    *where* the gap is as well as *what* belongs there.
    """

    def __init__(self, train, mask_size=10, mask_size_jitter=2, data_dir="./data"):
        self.base = datasets.MNIST(
            root=data_dir,
            train=train,
            download=True,
            transform=transforms.ToTensor(),
        )
        self.mask_size = mask_size
        self.mask_size_jitter = mask_size_jitter

    def __len__(self):
        return len(self.base)

    def _random_mask(self, h, w):
        size = self.mask_size + random.randint(-self.mask_size_jitter, self.mask_size_jitter)
        size = max(4, min(size, min(h, w)))
        top = random.randint(0, h - size)
        left = random.randint(0, w - size)
        mask = torch.zeros(1, h, w)
        mask[:, top : top + size, left : left + size] = 1.0
        return mask

    def __getitem__(self, idx):
        image, _label = self.base[idx]
        _, h, w = image.shape
        mask = self._random_mask(h, w)
        masked_image = image * (1 - mask)
        return masked_image, image, mask


def get_dataloaders(batch_size=128, mask_size=10, data_dir="./data", num_workers=2):
    train_set = MaskedMNIST(train=True, mask_size=mask_size, data_dir=data_dir)
    test_set = MaskedMNIST(train=False, mask_size=mask_size, data_dir=data_dir)

    val_size = len(train_set) // 10
    train_size = len(train_set) - val_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_set, [train_size, val_size], generator=torch.Generator().manual_seed(0)
    )

    from torch.utils.data import DataLoader

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader
