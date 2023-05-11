import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class Transformer(nn.Module):
    """
    Transformer to compute output in a number of classes (numClasses) from image signature input (of dimension inputDim)
    """
    def __init__(self, inputDim : Tuple[int, int], numClasses : int = 10):
      super(Transformer, self).__init__()
      # TODO

    def forward(self, x : torch.tensor):
       # TODO
       pass