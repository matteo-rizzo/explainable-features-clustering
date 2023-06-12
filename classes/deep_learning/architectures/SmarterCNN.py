import torch.nn as nn
from torch import Tensor

from classes.deep_learning.architectures.modules.common import Conv


class SmarterCNN(nn.Module):

    def __init__(self, config: dict, **kwargs):
        super().__init__()
        num_channels, img_size, num_classes = (config["num_channels"],
                                               config["img_size"],
                                               config["num_classes"])
        self.conv1 = Conv(in_channels=num_channels, out_channels=32,
                          kernel_size=3, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.drop1 = nn.Dropout(0.2)
        # ----------------------------------------------------------
        self.conv2 = Conv(in_channels=32, out_channels=64,
                          kernel_size=3, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.drop2 = nn.Dropout(0.2)
        # ----------------------------------------------------------
        self.conv3 = Conv(in_channels=64, out_channels=128,
                          kernel_size=3, stride=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.drop3 = nn.Dropout(0.2)
        # ----------------------------------------------------------
        self.flatten = nn.Flatten()
        # Adjust the input size of the fully connected layer
        # Calculate the feature map size after the second pooling layer
        feature_map_size = img_size // (2 ** 3)  # divided by 2 for each max-pooling layers
        self.fc1 = nn.Linear(in_features=128 * feature_map_size * feature_map_size, out_features=num_classes)
        # ----------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        # -----------------
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.drop1(x)
        # -----------------
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.drop2(x)
        # -----------------
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.drop3(x)
        # -----------------
        x = self.flatten(x)
        x = self.fc1(x)
        # -----------------
        return x
