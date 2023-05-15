import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class FC(nn.Module):
    """
    Dense neural network to compute output in a number of classes (num_classes)
    from image signature input (of dimension input_dim).
    """

    def __init__(self, input_dim: Tuple[int, int], hidden_n: int = 50, num_classes: int = 10):
        super(FC, self).__init__()
        self.fc1 = nn.Linear(input_dim[0] * input_dim[1], hidden_n)
        self.fc2 = nn.Linear(hidden_n, num_classes)

    def forward(self, x: torch.tensor):
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = F.dropout(x, 0.4)
        x = F.relu(x)
        x = self.fc2(x)
        out = F.log_softmax(x, dim=1)
        return out
