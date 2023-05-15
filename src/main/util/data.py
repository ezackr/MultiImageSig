from PIL import Image
import os
from typing import Tuple
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader, random_split
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from src.main.util.signature import Signature

data_path: str = os.path.join(os.path.dirname(__file__).rstrip(os.path.normpath("/src/main/util/data.py")), "data")


def _load_cifar10_samples_labels(cifar10_dataset: datasets.CIFAR10, transform: transforms.Compose):
    samples, labels = [], []
    for sample, label in tqdm(cifar10_dataset, desc="Processing dataset"):
        sample = transform(sample)
        samples.append(sample)
        labels.append(label)
    return torch.stack(samples), torch.tensor(labels)


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
        Signature(depth),
        transforms.Lambda(lambda x: x.to(dtype=torch.float32))
    ])

    # path for artifacts for train/test samples/labels
    artifacts_path = os.path.join(os.path.abspath(data_path), "cifar10/artifacts")

    # check if artifacts already exist.
    try:
        if len(os.listdir(os.path.join(artifacts_path))):
            print(f"Loading artifacts from {artifacts_path}")
            train_samples = torch.load(os.path.join(artifacts_path, "train_samples.pt"))
            train_labels = torch.load(os.path.join(artifacts_path, "train_labels.pt"))
            test_samples = torch.load(os.path.join(artifacts_path, "test_samples.pt"))
            test_labels = torch.load(os.path.join(artifacts_path, "test_labels.pt"))
            return TensorDataset(train_samples, train_labels), TensorDataset(test_samples, test_labels)
    except FileNotFoundError:
        os.makedirs(os.path.abspath(artifacts_path))

    print(f"No saved artifact found at {artifacts_path}\nLoading images for cifar10...")

    # load training data.
    train_data = datasets.CIFAR10(
        root=data_path, train=True, download=True
    )
    # load test data.
    test_data = datasets.CIFAR10(
        root=data_path, train=False, download=True
    )

    # load samples, labels and combine tensors into datasets.
    test_samples, test_labels = _load_cifar10_samples_labels(test_data, transform)
    train_samples, train_labels = _load_cifar10_samples_labels(train_data, transform)

    # add new artifacts.
    torch.save(train_samples, os.path.join(artifacts_path, "train_samples.pt"))
    torch.save(train_labels, os.path.join(artifacts_path, "train_labels.pt"))
    torch.save(test_samples, os.path.join(artifacts_path, "test_samples.pt"))
    torch.save(test_labels, os.path.join(artifacts_path, "test_labels.pt"))

    # return new dataset.
    return TensorDataset(train_samples, train_labels), TensorDataset(test_samples, test_labels)


def _load_sample_labels(root: str, label: int, depth: int = 4):
    transform = transforms.Compose([
        transforms.PILToTensor(),
        transforms.Lambda(lambda x: x / 255.0),
        Signature(depth),
        transforms.Lambda(lambda x: x.to(dtype=torch.float32))
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
    # check if artifacts already exist.
    if len(os.listdir(data_path + "/concrete-crack/artifacts")):
        print(f"Loading artifacts from {data_path}/concrete-crack/artifacts")
        train_samples = torch.load(data_path + "/concrete-crack/artifacts/train_samples.pt")
        train_labels = torch.load(data_path + "/concrete-crack/artifacts/train_labels.pt")
        test_samples = torch.load(data_path + "/concrete-crack/artifacts/test_samples.pt")
        test_labels = torch.load(data_path + "/concrete-crack/artifacts/test_labels.pt")
        return TensorDataset(train_samples, train_labels), TensorDataset(test_samples, test_labels)
    else:
        print(f"No saved artifact found at {data_path}/concrete-crack/artifacts")
    # load each sample.
    print(f"Loading images from {data_path}/concrete-crack")
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
    # add new artifacts.
    torch.save(train_samples, data_path + "/concrete-crack/artifacts/train_samples.pt")
    torch.save(train_labels, data_path + "/concrete-crack/artifacts/train_labels.pt")
    torch.save(test_samples, data_path + "/concrete-crack/artifacts/test_samples.pt")
    torch.save(test_labels, data_path + "/concrete-crack/artifacts/test_labels.pt")
    # return new dataset.
    return TensorDataset(train_samples, train_labels), TensorDataset(test_samples, test_labels)


def get_data_loaders(dataset: str, depth: int, batchsize: int) -> Tuple[torch.Size, DataLoader, DataLoader, DataLoader]:
    """
    Load a dataset, process a signature transform over all data with given depth, batching into batches of given
    size, and split train set into train and validation sets

    :param dataset: Dataset to load ["cifar" for cifar10, "concretecracks" for concrete cracks dataset]
    :param depth: Depth of signature transform
    :param batchsize: Batch size for all data loaders
    :return: Size of a sample in the dataset, and train, validation, and test DataLoaders
    """
    # Load train, test sets based on dataset requested
    train_val_data, test_data = None, None
    if dataset == "cifar":
        train_val_data, test_data = load_cifar10(depth)
    elif dataset == "concretecracks":
        train_val_data, test_data = load_concrete_cracks(depth)

    # Split train set into train/val sets with 90/10 split
    train_data, val_data = random_split(
        train_val_data,
        [int(0.9 * len(train_val_data)), int(0.1 * len(train_val_data))]
    )
    train_loader = DataLoader(train_data, batch_size=batchsize, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batchsize, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batchsize, shuffle=False)
    return train_data[0][0].shape, train_loader, val_loader, test_loader
