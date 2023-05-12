import torch

import iisignature


class Signature:
    """
    Signature transform on image tensor.

    Args:
        depth (int): Depth used for signature.
    """

    def __init__(self, depth: int = 4):
        self.depth = depth

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        signatures = []
        for stream in image:
            sig = torch.as_tensor(iisignature.sig(stream, self.depth))
            signatures.append(sig)
        return torch.stack(signatures)
