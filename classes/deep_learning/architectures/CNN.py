import logging

import colorlog
import torch.nn as nn
import yaml

from functional.arc_utils import parse_model


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        with open("config/architectures/cnn.yaml", "r") as f:
            cfg = yaml.safe_load(f)
        self.model, save = parse_model(cfg, logger=logger)
        # self.conv1 = nn.Conv2d(in_channels=3, out_channels=16,
        #                        kernel_size=3, stride=1, padding=1)
        # self.relu1 = nn.ReLU()
        # self.pool1 = nn.MaxPool2d(kernel_size=2)
        #
        # self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        # self.relu2 = nn.ReLU()
        # self.pool2 = nn.MaxPool2d(kernel_size=2)
        #
        # self.fc1 = nn.Linear(in_features=32 * 8 * 8, out_features=128)
        # self.relu3 = nn.ReLU()
        # self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.relu1(x)
        # x = self.pool1(x)
        #
        # x = self.conv2(x)
        # x = self.relu2(x)
        # x = self.pool2(x)
        #
        # x = x.view(-1, 32 * 8 * 8)
        #
        # x = self.fc1(x)
        # x = self.relu3(x)
        # x = self.fc2(x)

        # return self.model(x)
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
        return x


if __name__ == "__main__":
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
    model = CNN()
    # print(model)
