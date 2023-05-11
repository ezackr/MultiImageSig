import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class FC(nn.Module):
    """
    Dense neural network to compute output in a number of classes (numClasses) from image signature input (of dimension inputDim)
    """
    def __init__(self, inputDim : Tuple[int, int], hiddenN : int = 50, numClasses : int = 10):
      super(FC, self).__init__()
      self.fc1 = nn.Linear(inputDim[0] * inputDim[1], hiddenN)
      self.fc2 = nn.Linear(hiddenN, numClasses)

    def forward(self, x : torch.tensor):
       x = torch.flatten(x, 1)
       x = self.fc1(x)
       x = F.dropout(x, 0.4)
       x = F.relu(x)
       x = self.fc2(x)
       out = F.log_softmax(x, dim=1)
       return out