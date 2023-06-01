import torch

from src.main.models.residual import ResidualBlock, ResNet


def test_residual_block():
    batch_size = 32
    num_sig = 227
    sig_dim = 120
    x = torch.rand(size=(batch_size, sig_dim, num_sig))

    out_channels = 64
    model = ResidualBlock(sig_dim, out_channels=out_channels)
    y_hat = model(x)
    assert y_hat.shape == (batch_size, out_channels, num_sig)


def test_resnet():
    batch_size = 32
    num_sig = 227
    sig_dim = 120
    x = torch.rand(size=(batch_size, sig_dim, num_sig))

    num_classes = 10
    model = ResNet(input_dim=(sig_dim, num_sig), num_classes=num_classes)
    y_hat = model(x)
    assert y_hat.shape == (batch_size, num_classes)
