from src.main.util.data import load_cifar10
from .models.cnn import CNN
from .models.fc import FC
from .models.transformer import Encoder

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.utils.data import DataLoader, random_split

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


def train(model: nn.Module, train_loader: DataLoader, eval_loader: DataLoader, epochs: int = 10, lr: float = 0.01,
          momentum: float = 0.1, weight_decay: float = 0.05):
    # Train the model on train_loader with given hyperparameters, computing accuracy on eval_loader
    model.to(device)

    # set up loss and optimization
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    training_losses = []

    for epoch in range(epochs):
        running_loss = 0.0

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            training_losses.append(loss.item())

            if i % 10 == 9:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0
        print("Epoch %d. Eval accuracy=%f%%" % (epoch + 1, accuracy(model, eval_loader)))

    return training_losses


if __name__ == "__main__":
    depth = 3
    batch_size = 64

    # Load data, split into train/validation/test, and create DataLoaders
    trainval_data, test_data = load_cifar10(3)
    train_data, val_data = random_split(trainval_data, [int(0.9 * len(trainval_data)), int(0.1 * trainval_data)])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    # Initialize models
    inputDim = train_data[0][0].size()
    fc_model = FC(inputDim)
    cnn_model = CNN(inputDim)
    # transformer_model = Transformer()

    print("Training fully-connected model")
    train(fc_model, train_loader, val_loader)
