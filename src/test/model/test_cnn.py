import torch

from src.main.models.cnn import CNN


def test_cnn():
    batch_size = 32
    dim_1_in = 512
    dim_2_in = 256
    hidden_n = 25
    num_classes = 5
    x = torch.rand(size=(batch_size, dim_1_in, dim_2_in))
    nn = CNN((dim_1_in, dim_2_in), hidden_n, num_classes)
    x_out = nn.forward(x)

    assert x_out.shape == (batch_size, num_classes)
