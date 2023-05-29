import time
import argparse
from tqdm.auto import tqdm
import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.main.models import CNN, AttentionEncoder, FC
from src.main.util import checkpoints, get_data_loaders, accuracy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

base_path: str = os.path.dirname(__file__).rstrip(os.path.normpath("/src/main/train.py"))


def train(
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        learning_rate: float,
        weight_decay: float,
        depth: int,
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
    start_time = time.time()
    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()
        total_loss = 0.0
        for x, y in tqdm(train_loader, position=0, leave=True):
            inputs, labels = x.to(device), y.to(dtype=torch.long, device=device)

            optimizer.zero_grad()

            pred = model(inputs)
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        train_losses.append(total_loss / len(train_loader))
        train_accuracies.append(accuracy(model, val_loader, device))
        if checkpoints_path is not None:
            checkpoint_name = checkpoints.generate_checkpoint_name(checkpoints_path, model, epoch+1, depth)
            checkpoints.save_checkpoint(optimizer, model, checkpoint_name)
        print(f"Epoch {epoch + 1}. "
              f"Train Loss={round(train_losses[-1], 4)}. "
              f"Validation Accuracy={round(train_accuracies[-1], 4)}. "
              f"Total Time={round((time.time() - epoch_start_time) / 60, 2)}m")
    print(f"Total training time={round((time.time() - start_time) / 60, 2)}m")
    return train_losses


def main(model_type: str, depth: int, batchsize: int, dataset: str, checkpoints_path: str, *args, **kwargs):
    # Load dataset, split into train/validation/test sets, and create DataLoaders.
    num_classes, input_shape, train_loader, val_loader, test_loader = get_data_loaders(dataset, depth, batchsize, False)
    val_loader = test_loader  # Skip creating train/val split since we don't tune hyperparams

    # Initialize model
    model = None
    if model_type == "fc":
        model = FC(input_shape)
    elif model_type == "cnn":
        model = CNN(input_shape)
    elif model_type == "attn":
        model = AttentionEncoder(input_shape, num_classes=num_classes)

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
        depth=depth,
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
        choices=["cifar10", "concretecracks"]
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
