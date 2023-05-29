import torch

from src.main.models.transformer import AttentionEncoder, EncoderBlock


def test_encoder_block():
    batch_size = 32
    d_model = 512
    seq_len = 256
    x = torch.rand(size=(batch_size, seq_len, d_model))

    block = EncoderBlock(d_model=d_model, num_heads=8, dropout=0.0)
    x_enc = block(x)
    assert x.shape == x_enc.shape


def test_encoder():
    batch_size = 32
    input_dim = 120
    num_classes = 10
    seq_len = 256
    x = torch.rand(size=(batch_size, seq_len, input_dim))

    encoder = AttentionEncoder(
        input_shape=(seq_len, input_dim),
        num_layers=6,
        num_heads=8,
        num_classes=num_classes,
        dropout=0.0
    )
    x_enc = encoder(x)

    expected = [batch_size, num_classes]
    actual = list(x_enc.shape)
    assert expected == actual
