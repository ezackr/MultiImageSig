import time
from typing import Tuple
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from src.main.models import CNN, Encoder, FC
from src.main.util import load_cifar10, load_concrete_cracks

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def accuracy(model: nn.Module, dataset_loader: DataLoader):
    # Calculate accuracy of model on dataset loader
    correct = 0
    total = 0
    with torch.no_grad():
        for _, batch in enumerate(dataset_loader, 0):
            signatures, labels = batch[0].to(device), batch[1].to(device)
            outputs = model(signatures)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


def train(
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 10,
        lr: float = 0.01,
        momentum: float = 0.1,
        weight_decay: float = 0.05
):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    train_losses = []
    for i in range(epochs):
        start_time = time.time()
        model.train()
        total_loss = 0.0
        for x, y in tqdm(train_loader):
            inputs, labels = x.to(device), y.to(device)

            optimizer.zero_grad()

            pred = model(inputs)
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        train_losses.append(total_loss / len(train_loader))
        print(f"Epoch {i + 1}. "
              f"Train Loss={train_losses[-1]}. "
              f"Validation Accuracy={accuracy(model, val_loader)}. "
              f"Total Time={round((time.time() - start_time)/ 60, 2)}m")
    return train_losses


def _get_data(
        depth: int,
        batch_size: int,
        dataset_name: str
) -> Tuple[torch.Size, DataLoader, DataLoader, DataLoader]:
    start_time = time.time()
    print(f"Loading dataset...")
    if dataset_name == "cifar-10":
        train_val_data, test_data = load_cifar10(depth=depth)
    else:
        train_val_data, test_data = load_concrete_cracks(depth=depth)
    train_data, val_data = random_split(
        train_val_data,
        [int(0.9 * len(train_val_data)), int(0.1 * len(train_val_data))]
    )
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    print(f"Dataset loaded in {time.time() - start_time}s")
    return train_data[0][0].shape, train_loader, val_loader, test_loader


def main():
    depth: int = 4
    batch_size: int = 64
    dataset_name = "concrete-cracks"   # dataset_name should be either "cifar-10" or "concrete-cracks"

    # Load dataset, split into train/validation/test sets, and create DataLoaders.
    input_shape, train_loader, val_loader, test_loader = _get_data(depth, batch_size, dataset_name)

    # Initialize models
    fc_model = FC(input_shape)
    # cnn_model = CNN(input_shape)
    # attn_model = Encoder(input_shape[1])

    train(fc_model, train_loader, val_loader)


if __name__ == "__main__":
    main()
