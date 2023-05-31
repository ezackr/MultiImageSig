import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class CNN(nn.Module):
    """
    CNN to compute output in a number of classes (num_classes) from image signature input (of dimension input_dim)
    """
    def __init__(self, input_dim: Tuple[int, int], hidden_n: int = 50, num_classes: int = 10):
        super(CNN, self).__init__()
        # Input has dimension (N, I1, I2)
        self.conv1 = nn.Conv1d(input_dim[0], 32, kernel_size=3, padding=0, dilation=1)  # (N, I1, I2) -> (N, 32, I3)
        i3 = math.floor((input_dim[1]+2*0-1*(3-1)-1)/1 + 1)  # after conv1d
        i4 = math.floor((i3-3)/3+1)  # max-pooling (N, 32, I3) -> (N, 32, I4)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=0, dilation=1)            # (N, 32, I4) -> (N, 64, I5)
        i5 = math.floor((i4+2*0-1*(3-1)-1)/1 + 1)  # after conv1d
        i6 = math.floor((i5-3)/3+1)                                         # max-pooling (N, 64, I5) -> (N, 64, I6)
        self.fc1 = nn.Linear(64 * i6, hidden_n)                             # (N, 64, I6) -> (N, 64*I6)
        self.fc2 = nn.Linear(hidden_n, num_classes)

    def forward(self, x: torch.tensor):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool1d(x, 3, stride=3, padding=0)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool1d(x, 3, stride=3, padding=0)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, 0.4)
        x = self.fc2(x)
        out = F.softmax(x, dim=1)
        return out
