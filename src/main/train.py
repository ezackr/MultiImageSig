import argparse
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from src.main.models import CNN, Encoder, FC
from src.main.util import load_cifar10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def accuracy(model: nn.Module, dataset_loader: DataLoader, checkpoint_path: str = None):
    # Calculate accuracy of model on dataset loader
    correct = 0
    total = 0
    model.eval()
    # Save model to checkpoint, if any
    # TODO
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
        weight_decay: float = 0.05,
        checkpoint_path: str = None
):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    train_losses = []
    for i in range(epochs):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            inputs, labels = x.to(device), y.to(device)

            optimizer.zero_grad()

            pred = model(inputs)
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        train_losses.append(total_loss / len(train_loader))
        print(f"Epoch {i}. Train Loss={train_losses[-1]}. Validation Accuracy={accuracy(model, val_loader)}")
    return train_losses


def _get_cifar_data(depth: int, batch_size: int) -> Tuple[torch.Size, DataLoader, DataLoader, DataLoader]:
    train_val_data, test_data = load_cifar10(depth=depth)
    train_data, val_data = random_split(
        train_val_data,
        [int(0.9 * len(train_val_data)), int(0.1 * len(train_val_data))]
    )
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_data[0][0].shape, train_loader, val_loader, test_loader


def main(model_type: str, depth: int, batch_size: int, checkpoint_path: str):
    # Load dataset, split into train/validation/test sets, and create DataLoaders.
    input_shape, train_loader, val_loader, test_loader = _get_cifar_data(depth, batch_size)

    # Initialize models
    model = None
    if model_type == "fc":
        model = FC(input_shape)
    elif model_type == "cnn":
        model = CNN(input_shape)
    elif model_type == "attn":
        model = Encoder(input_shape[1])

    # Run train method
    train(model, train_loader, val_loader, checkpoint_path=checkpoint_path)

    # Store trained model




def run_main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
            "-m",
            "--model",
            help="Model type (options: fc, cnn, attn)",
            required=True,
            choices=["fc","cnn", "attn"]
    )
    arg_parser.add_argument("-d", "--depth", help="Signature transform depth", default=4, type=int)
    arg_parser.add_argument("-b", "--batch", help="Training batch size", default=64, type=int)
    arg_parser.add_argument("-chk", "--checkpoint-path", help="Path to training checkpoint for model")

    args = arg_parser.parse_args()
    main(args)


if __name__ == "__main__":
    run_main()
