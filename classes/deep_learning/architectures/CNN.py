import torch.nn as nn
from torch import Tensor


class CNN(nn.Module):

    def __init__(self, **kwargs):
        super(CNN, self).__init__()
        # # ----------------------------------------------------------
        # self.conv1 = nn.Conv2d(in_channels=1, out_channels=16,
        #                        kernel_size=2, stride=1, padding=2)
        # self.relu1 = nn.ReLU()
        # self.pool1 = nn.MaxPool2d(kernel_size=2)
        # # ----------------------------------------------------------
        # self.conv2 = nn.Conv2d(in_channels=16, out_channels=32,
        #                        kernel_size=2, stride=1, padding=2)
        # self.relu2 = nn.ReLU()
        # self.pool2 = nn.MaxPool2d(kernel_size=2)
        # # ----------------------------------------------------------
        # self.flatten = nn.Flatten()
        # self.fc1 = nn.Linear(in_features=70656, out_features=10)
        # # self.relu3 = nn.ReLU()
        # # self.fc2 = nn.Linear(in_features=128, out_features=10)
        # # ----------------------------------------------------------
        # ----------------------------------------------------------
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16,
                               kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        # ----------------------------------------------------------
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32,
                               kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        # ----------------------------------------------------------
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=32 * 7 * 7, out_features=10)
        # self.relu3 = nn.ReLU()
        # self.fc2 = nn.Linear(in_features=128, out_features=10)
        # ----------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        # -----------------
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
