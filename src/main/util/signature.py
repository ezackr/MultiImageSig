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
        """
        :param image: tensor representation of an image with shape (C x H x W)
        :return: a signature representation of an image.
        """
        # change shape from (C x H x W) to (H x W x C).
        image = image.permute(1, 2, 0)
        signatures = []
        for stream in image:
            sig = torch.as_tensor(iisignature.sig(stream, self.depth))
            signatures.append(sig)
        return torch.stack(signatures)
