import logging

import colorlog
import torch

from classes.deep_learning.architectures.TorchModel import TorchModel


class CNN(TorchModel):

    def __init__(self, config_path: str = "config/architectures/cnn.yaml",
                 logger: logging.Logger = logging.getLogger(__name__)):
        super(CNN, self).__init__(config_path=config_path, logger=logger)

    def forward(self, x):
        super().forward(x)


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

    print(image)
    print(model.forward(image))
    # print(model)


if __name__ == "__main__":
    main()
