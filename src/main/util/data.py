from PIL import Image
import os
from typing import Tuple

import torch
from torch.utils.data import Dataset, TensorDataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from src.main.util.signature import Signature

data_path: str = os.path.dirname(__file__).rstrip("/src/main/util/data.py") + "/data"


def load_cifar10(depth: int = 4) -> Tuple[Dataset, Dataset]:
    """
    :param depth: the depth of the signature transform.
    :return: the CIFAR10 dataset represented by tensors.
        - The first entry in the returned Tuple is the training data.
        - The second entry in the returned Tuple is the test data.
    """
    # get data transformation.
    transform = transforms.Compose([
        transforms.ToTensor(),
        Signature(depth)
    ])
    # load training data.
    train_data = datasets.CIFAR10(
        root=data_path, train=True, download=True, transform=transform
    )
    # load test data.
    test_data = datasets.CIFAR10(
        root=data_path, train=False, download=True, transform=transform
    )
    return train_data, test_data


def _load_sample_labels(root: str, label: int, depth: int = 4):
    transform = transforms.Compose([
        transforms.PILToTensor(),
        transforms.Lambda(lambda x: x / 255.0),
        Signature(depth)
    ])
    images = []
    for name in os.listdir(root):
        fp = os.path.join(root, name)
        if os.path.isfile(fp):
            img = transform(Image.open(fp))
            images.append(img)
    return torch.stack(images).permute(0, 1, 2), torch.ones(len(images)) * label


def load_concrete_cracks(depth: int = 4) -> Tuple[Dataset, Dataset]:
    """
    :param depth: the depth of the signature transform.
    :return: the Concrete Cracks dataset represented by tensors.
        - The first entry in the returned Tuple is the training data.
        - The second entry in the returned Tuple is the test data.
    """
    neg_samples, neg_labels = _load_sample_labels(
        root=data_path + "/concrete-crack/Negative",
        label=0,
        depth=depth
    )
    pos_samples, pos_labels = _load_sample_labels(
        root=data_path + "/concrete-crack/Positive",
        label=1,
        depth=depth
    )
    # split data into train and test sets (85/15 split).
    train_neg_samples, train_neg_labels = neg_samples[:17000], neg_labels[:17000]
    test_neg_samples, test_neg_labels = neg_samples[-3000:], neg_labels[-3000:]
    train_pos_samples, train_pos_labels = pos_samples[:17000], pos_labels[:17000]
    test_pos_samples, test_pos_labels = pos_samples[-3000:], pos_labels[-3000:]
    # combine tensors into datasets.
    train_samples = torch.vstack([train_neg_samples, train_pos_samples])
    train_labels = torch.cat([train_neg_labels, train_pos_labels])
    test_samples = torch.vstack([test_neg_samples, test_pos_samples])
    test_labels = torch.cat([test_neg_labels, test_pos_labels])
    return TensorDataset(train_samples, train_labels), TensorDataset(test_samples, test_labels)
