import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.main.models import CNN, AttentionEncoder, FC
from src.main.util import checkpoints, get_data_loaders, metrics, flops_and_params

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

base_path: str = os.path.dirname(__file__).rstrip(os.path.normpath("/src/main/train.py"))


def evaluate(
        model: nn.Module,
        eval_loader: DataLoader,
        checkpoint_name_path: str
):
    model.to(device)

    # Load from checkpoint
    print(f"Loading checkpoint to evaluate model on from: {checkpoint_name_path}")
    checkpoints.load_checkpoint(None, model, checkpoint_name_path)

    acc, f1 = metrics(model, eval_loader, device)
    print("Eval accuracy: ", acc)
    print("Eval F1: ", f1)

def main(model_type: str, dataset: str, depth: int, checkpoint_name: str, checkpoints_path: str):
    # Load dataset, split into train/validation/test sets, and create DataLoaders.
    num_classes, input_shape, train_loader, val_loader, test_loader = get_data_loaders(dataset, depth, 32, False)
    eval_loader = test_loader  # Skip creating train/val split since we don't tune hyperparams

    # Initialize model
    model = None
    if model_type == "fc":
        model = FC(input_shape, num_classes=num_classes)
    elif model_type == "cnn":
        model = CNN(input_shape, num_classes=num_classes)
    elif model_type == "attn":
        model = AttentionEncoder(input_shape, num_classes=num_classes)

    # Setup checkpoints path, if it doesn't exist
    checkpoints_full_path = os.path.join(os.path.join(base_path, checkpoints_path), checkpoint_name)

    print(f"Evaluating model {model_type}")
    flops, params = flops_and_params(model, next(iter(train_loader))[0][0].shape, num_classes)
    print(f"Number of parameters: {params}")
    print(f"Number of FLOPs: {flops}")

    # Evaluate the model
    evaluate(
        model,
        eval_loader,
        checkpoints_full_path
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
    arg_parser.add_argument(
        "-chkpts",
        "--checkpoints-path",
        help="Directory to store training checkpoints for model, relative to root directory of project",
        default=None,
        required=True,
        type=str
    )
    arg_parser.add_argument(
        "-ichkpt",
        "--checkpoint-name",
        help="Name/path of checkpoint to start training at, relative to --checkpoints-path",
        default=None,
        required=True,
        type=str
    )

    args, _ = arg_parser.parse_known_args()  # Only parse known args
    main(**vars(args))


if __name__ == "__main__":
    run_main()
