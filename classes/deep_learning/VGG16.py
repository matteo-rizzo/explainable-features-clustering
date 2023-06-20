import torch.nn as nn
from torch import Tensor

from classes.deep_learning.modules.common import Conv


class VGG16(nn.Module):

    def __init__(self, config: dict, **kwargs):
        super(VGG16, self).__init__()
        # ----------------------------------------------------------
        num_channels, num_classes = (config["num_channels"],
                                     config["num_classes"])
        self.layer_1 = Conv(in_channels=num_channels, out_channels=64,
                            kernel_size=3, stride=1)
        # ----------------------------------------------------------
        self.layer_2 = nn.Sequential(
            Conv(in_channels=64, out_channels=64,
                 kernel_size=3, stride=1),
            nn.MaxPool2d(kernel_size=2)
        )
        # ----------------------------------------------------------
        self.layer_3 = Conv(in_channels=64, out_channels=128,
                            kernel_size=3, stride=1)
        # ----------------------------------------------------------
        self.layer_4 = nn.Sequential(
            Conv(in_channels=128, out_channels=128,
                 kernel_size=3, stride=1),
            nn.MaxPool2d(kernel_size=2)
        )
        # ----------------------------------------------------------
        self.layer_5 = Conv(in_channels=128, out_channels=256,
                            kernel_size=3, stride=1)
        # ----------------------------------------------------------
        self.layer_6 = Conv(in_channels=256, out_channels=256,
                            kernel_size=3, stride=1)
        # ----------------------------------------------------------
        self.layer_7 = nn.Sequential(
            Conv(in_channels=256, out_channels=256,
                 kernel_size=3, stride=1),
            nn.MaxPool2d(kernel_size=2)
        )
        # ----------------------------------------------------------
        self.layer_8 = Conv(in_channels=256, out_channels=512,
                            kernel_size=3, stride=1)
        # ----------------------------------------------------------
        self.layer_9 = Conv(in_channels=512, out_channels=512,
                            kernel_size=3, stride=1)
        # ----------------------------------------------------------
        self.layer_10 = nn.Sequential(
            Conv(in_channels=512, out_channels=512,
                 kernel_size=3, stride=1),
            nn.MaxPool2d(kernel_size=2)
        )
        # ----------------------------------------------------------
        self.layer_11 = Conv(in_channels=512, out_channels=512,
                             kernel_size=3, stride=1)
        # ----------------------------------------------------------
        self.layer_12 = Conv(in_channels=512, out_channels=512,
                             kernel_size=3, stride=1)
        # ----------------------------------------------------------
        self.layer_13 = nn.Sequential(
            Conv(in_channels=512, out_channels=512,
                 kernel_size=3, stride=1),
            nn.MaxPool2d(kernel_size=2)
        )
        # ----------------------------------------------------------
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(7 * 7 * 512, 4096),
            nn.SiLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.SiLU())
        self.fc2 = nn.Sequential(
            nn.Linear(4096, num_classes))

    def forward(self, x: Tensor) -> Tensor:
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        x = self.layer_6(x)
        x = self.layer_7(x)
        x = self.layer_8(x)
        x = self.layer_9(x)
        x = self.layer_10(x)
        x = self.layer_11(x)
        x = self.layer_12(x)
        x = self.layer_13(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
