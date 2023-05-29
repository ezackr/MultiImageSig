import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from typing import Tuple
import flopth


def metrics(model: nn.Module, dataset_loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    """
    Calculate accuracy and F1 score of model on dataset loader

    :param model: Model to evaluate
    :param dataset_loader: Dataset to evaluate model over
    :param device: Device to evaluate model on
    :return: Accuracy (Number of true predictions / total number of examples)
    """
    total = 0
    y_true = torch.tensor([])
    y_pred = torch.tensor([])
    model.eval()
    with torch.no_grad():
        for x, y in dataset_loader:
            signatures, labels = x.to(device), y.to(dtype=torch.long, device=device)
            outputs = model(signatures)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            y_true = torch.cat((y_true, labels))
            y_pred = torch.cat((y_pred, predicted))
    correct = (y_pred == y_true).sum().item()
    f1 = f1_score(y_true, y_pred, average='micro')
    accuracy = correct / total
    return accuracy, f1


def flops_and_params(model: nn.Module, input_size: torch.Tensor):
    """
    Returns tuple of FLOPs and number of parameters in model, given inputs of size input_size
    :param model:
    :param input_size:
    :return:
    """
    return flopth.flopth(model, in_size=input_size)
