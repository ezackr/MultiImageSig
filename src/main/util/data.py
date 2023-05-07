import os
from typing import Tuple

from torch.utils.data import Dataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms


data_path: str = os.path.dirname(__file__).rstrip("/src/main/util/data.py") + "/data"


def _get_transform():
    """
    :return: a transformation that converts images to normalized tensors.
    """
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )


def load_cifar10() -> Tuple[Dataset, Dataset]:
    """
    :return: the CIFAR10 dataset represented by tensors.
        - The first entry in the returned Tuple is the training data.
        - The second entry in the returned Tuple is the test data.
    """
    transform = _get_transform()
    # load training data.
    train_data = datasets.CIFAR10(
        root=data_path, train=True, download=True, transform=transform
    )
    # load test data.
    test_data = datasets.CIFAR10(
        root=data_path, train=False, download=True, transform=transform
    )
    return train_data, test_data
