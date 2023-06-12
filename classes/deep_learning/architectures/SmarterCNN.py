import torch.nn as nn
from torch import Tensor

from classes.deep_learning.architectures.modules.common import Conv


class SmarterCNN(nn.Module):

    def __init__(self, config: dict, **kwargs):
        super().__init__()
        num_channels, img_size, num_classes = (config["num_channels"],
                                               config["img_size"],
                                               config["num_classes"])
        self.conv1 = Conv(num_channels, 32, kernel_size=5, stride=2, padding=2)
        self.conv2 = Conv(32, 32, kernel_size=5, stride=2, padding=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.2)
        # -----------------
        self.conv3 = Conv(32, 64, kernel_size=3, padding=1)
        self.conv4 = Conv(64, 64, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0.2)
        # -----------------
        self.conv5 = Conv(64, 128, kernel_size=2, padding=1)
        self.conv6 = Conv(128, 128, kernel_size=2, padding=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout(0.2)
        # -----------------
        self.conv7 = Conv(128, 256, kernel_size=2, padding=1)
        self.conv8 = Conv(256, 256, kernel_size=2, padding=1)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        # -----------------
        self.fc1 = nn.Linear(256, 512)
        self.act = nn.SiLU()
        self.dropout4 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(512, num_classes)
        # ----------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpool1(x)
        x = self.dropout1(x)
        # -----------------
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.maxpool2(x)
        x = self.dropout2(x)
        # -----------------
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.maxpool3(x)
        x = self.dropout3(x)
        # -----------------
        x = self.conv7(x)
        x = self.conv8(x)
        # -----------------
        x = self.global_avg_pool(x)
        x = self.flatten(x)
        # -----------------
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout4(x)
        x = self.fc2(x)
        # -----------------
        return x
