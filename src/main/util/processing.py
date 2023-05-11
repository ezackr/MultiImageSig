import signatory


class Signature:
    """
    Signature transform on image tensor

    Args:
        depth (int): Depth used for signature
    """

    def __init__(self, depth):
        assert isinstance(depth, int)
        self.depth = depth

    def __call__(self, sample):
        return signatory.signature(sample, self.depth)