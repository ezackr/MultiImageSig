import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        # default convolutional layers.
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=(3,),
            stride=(stride,),
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=(3,),
            stride=(1,),
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        # skip connection.
        if stride != 1 or in_channels != out_channels:
            self.down_sample = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=(1,),
                    stride=(stride,),
                    bias=False
                ),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        res = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if hasattr(self, "down_sample"):
            res = self.down_sample(res)
        out += res
        out = self.relu(out)

        return out


def _make_layer(
        in_channels,
        out_channels,
        num_layers: int = 2,
        stride: int = 1
):
    layers = [ResidualBlock(in_channels, out_channels, stride)]
    for _ in range(num_layers - 1):
        layers.append(ResidualBlock(out_channels, out_channels))
    return nn.Sequential(*layers)


class ResNet(nn.Module):
    def __init__(
            self,
            in_channels: int,
            num_layers: int = 2,
            num_classes: int = 10
    ):
        super(ResNet, self).__init__()

        num_channels = 64

        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=num_channels,
            kernel_size=(7,),
            stride=(2,),
            padding=3,
            bias=False
        )
        self.bn1 = nn.BatchNorm1d(num_features=num_channels)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = _make_layer(
            in_channels=num_channels,
            out_channels=num_channels * (2 ** 0),
            num_layers=num_layers,
            stride=1
        )
        self.layer2 = _make_layer(
            in_channels=num_channels * (2 ** 0),
            out_channels=num_channels * (2 ** 1),
            num_layers=num_layers,
            stride=2
        )
        self.layer3 = _make_layer(
            in_channels=num_channels * (2 ** 1),
            out_channels=num_channels * (2 ** 2),
            num_layers=num_layers,
            stride=2
        )
        self.layer4 = _make_layer(
            in_channels=num_channels * (2 ** 2),
            out_channels=num_channels * (2 ** 3),
            num_layers=num_layers,
            stride=2
        )

        self.avg_pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.fc = nn.Linear(num_channels * (2 ** 3), num_classes)

    def forward(self, x):
        # reshapes input from (batch size, signal length, signal dim)
        # to (batch size, signal dim, signal length).
        x = torch.permute(x, (0, 2, 1))

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.max_pool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out
