import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class CNN(nn.Module):
    """
    CNN to compute output in a number of classes (numClasses) from image signature input (of dimension inputDim)
    """
    def __init__(self, inputDim: Tuple[int, int], hiddenN: int = 50, numClasses: int = 10):
        super(CNN, self).__init__()
        # TODO: implement correct filters
        self.conv1 = nn.Conv1d(3, 32, kernel_size=(3,))
        self.conv2 = nn.Conv1d(32, 64, kernel_size=(3,))
        self.fc1 = nn.Linear(inputDim[0] * inputDim[1], hiddenN)
        self.fc2 = nn.Linear(hiddenN, numClasses)

    def forward(self, x: torch.tensor):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool1d(3)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool1d(3)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, 0.4)
        x = self.fc2(x)
        out = F.log_softmax(x, dim=1)
        return out