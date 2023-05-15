import torch
from torch import nn
from torch.utils.data import DataLoader


def accuracy(model: nn.Module, dataset_loader: DataLoader, device: torch.device) -> float:
    """
    Calculate accuracy of model on dataset loader

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
    return correct / total
