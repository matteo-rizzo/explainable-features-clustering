import logging

import colorlog
import torch
import torch.nn as nn


class CNN(nn.Module):

    def __init__(self, config_path: str = "config/architectures/cnn.yaml",
                 logger: logging.Logger = logging.getLogger(__name__)):
        super(CNN, self).__init__()
        # ----------------------------------------------------------
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16,
                               kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        # ----------------------------------------------------------
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32,
                               kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        # ----------------------------------------------------------
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=32 * 7 * 7, out_features=128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=128, out_features=10)
        # ----------------------------------------------------------

    def forward(self, x):
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
        x = self.relu3(x)
        x = self.fc2(x)
        # -----------------
        return x


def main():
    logger = logging.getLogger(__name__)
    logger.setLevel("INFO")
    ch = logging.StreamHandler()
    ch.setLevel("INFO")
    formatter = colorlog.ColoredFormatter(
        "%(log_color)s[%(asctime)s] - %(levelname)s - %(white)s%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    model = CNN(logger=logger)
    image = torch.zeros(1, 3, 32, 32)

    # print(image)
    # print(model.forward(image))
    # print(model)


if __name__ == "__main__":
    main()
