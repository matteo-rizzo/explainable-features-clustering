import torch.nn as nn
from torch import Tensor


class CNN3D(nn.Module):

    def __init__(self, config: dict, **kwargs):
        super(CNN3D, self).__init__()
        # ----------------------------------------------------------
        num_channels, img_size, num_classes = (config["num_channels"],
                                               config["img_size"],
                                               config["num_classes"])
        # Pretend it has one channel and use channels as depth
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=16,
                               kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool3d(kernel_size=2)
        # ----------------------------------------------------------
        self.conv2 = nn.Conv3d(in_channels=16, out_channels=32,
                               kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool3d(kernel_size=2)
        # ----------------------------------------------------------
        self.flatten = nn.Flatten()
        # Adjust the input size of the fully connected layer
        self.fc1 = nn.Linear(in_features=25088*config["num_channels"], out_features=num_classes)
        # ----------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        # -----------------
        # `x` input tensor of shape [batch size, height, depth, width]
        x = x.unsqueeze(1)  # Insert a new dimension of size 1 at position 1
        # Now `x` has shape [batch size, 1, height, depth, width]
        x = x.permute(0, 1, 3, 2, 4)  # Rearrange dimensions
        # Now `x` has shape [batch size, channels, depth, height, width]
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        # -----------------
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        # -----------------
        x = self.flatten(x)
        x = self.fc1(x)
        # x = self.relu3(x)
        # x = self.fc2(x)
        # -----------------
        return x
