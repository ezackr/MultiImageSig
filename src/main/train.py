import argparse
from typing import Tuple
import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.main.models import CNN, Encoder, FC
from src.main.util import checkpoints, get_data_loaders

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)

base_path: str = os.path.dirname(__file__).rstrip(os.path.normpath("/src/main/train.py"))


def accuracy(model: nn.Module, dataset_loader: DataLoader, checkpoint_path: str = None):
    # Calculate accuracy of model on dataset loader
    correct = 0
    total = 0
    model.eval()
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
        epochs: int,
        learning_rate: float,
        weight_decay: float,
        initial_checkpoint_name: str = None,
        checkpoints_path: str = None
):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Load from best checkpoint
    if initial_checkpoint_name is not None:
        print(f"Loading initial checkpoint from {initial_checkpoint_name}")
        checkpoints.load_checkpoint(optimizer, model, os.path.join(checkpoints_path, initial_checkpoint_name))

    train_losses = []
    train_accuracies = []
    for epoch in range(epochs):
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
        train_accuracies.append(accuracy(model, val_loader))
        print(f"Epoch {epoch+1}. Train Loss={train_losses[-1]}. Validation Accuracy={train_accuracies[-1]}")
        if checkpoints_path is not None:
            checkpoint_name = checkpoints.generate_checkpoint_name(checkpoints_path, model, epoch+1)
            checkpoints.save_checkpoint(optimizer, model, checkpoint_name)
    return train_losses


def main(model_type: str, depth: int, batchsize: int, dataset: str, checkpoints_path: str, *args, **kwargs):
    # Load dataset, split into train/validation/test sets, and create DataLoaders.
    input_shape, train_loader, val_loader, test_loader = get_data_loaders(dataset, depth, batchsize)

    # Initialize model
    model = None
    if model_type == "fc":
        model = FC(input_shape)
    elif model_type == "cnn":
        model = CNN(input_shape)
    elif model_type == "attn":
        model = Encoder(input_shape[1])

    # Setup checkpoints path, if it doesn't exist
    checkpoints_full_path = os.path.join(base_path, checkpoints_path)
    if not os.path.exists(checkpoints_full_path):
        os.makedirs(checkpoints_full_path)

    print(f"Training model {model_type}")
    # Train model
    train(
        model,
        train_loader,
        val_loader,
        *args,
        checkpoints_path=checkpoints_full_path,
        **kwargs
    )


def run_main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "-m",
        "--model-type",
        help="Model type (options: fc, cnn, attn)",
        required=True,
        choices=["fc", "cnn", "attn"]
    )
    arg_parser.add_argument(
        "-ds",
        "--dataset",
        help="Name of dataset to train on",
        required=True,
        choices=["cifar", "concretecracks"]
    )
    arg_parser.add_argument("-d", "--depth", help="Signature transform depth", default=4, type=int)
    arg_parser.add_argument("-b", "--batchsize", help="Training batch size", default=64, type=int)
    arg_parser.add_argument("-n", "--epochs", help="Number of epochs", default=10, type=int)
    arg_parser.add_argument("-lr", "--learning-rate", help="Learning rate", default=0.01, type=float)
    arg_parser.add_argument("-w", "--weight-decay", help="Weight decay", default=0.05, type=float)
    arg_parser.add_argument(
        "-chkpts",
        "--checkpoints-path",
        help="Directory to store training checkpoints for model, relative to root directory of project",
        default=None,
        type=str
    )
    arg_parser.add_argument(
        "-ichkpt",
        "--initial-checkpoint-name",
        help="Name/path of checkpoint to start training at, relative to --checkpoints-path",
        default=None,
        type=str
    )

    args, _ = arg_parser.parse_known_args()  # Only parse known args
    main(**vars(args))


if __name__ == "__main__":
    run_main()
