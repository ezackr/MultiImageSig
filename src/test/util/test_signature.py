import torch

from src.main.util.signature import Signature


def test_signature_transform():
    image = torch.randn(size=(64, 64, 3))
    signature = Signature(depth=4)
    sig = signature(image)
    assert sig.shape == (64, 120)
