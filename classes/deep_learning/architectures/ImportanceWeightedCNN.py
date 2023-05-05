import logging

import colorlog
import numpy as np
import torch
from torch import Tensor, nn

from classes.deep_learning.architectures.CNN import CNN


class ImportanceWeightedCNN(CNN):

    def __init__(self, config_path: str = "config/architectures/importance_weighted_cnn.yaml",
                 logger: logging.Logger = logging.getLogger(__name__)):
        super(ImportanceWeightedCNN, self).__init__(config_path=config_path,
                                                    logger=logger)
        self.__importance_weight = nn.Parameter(torch.ones(1, 1, 28, 28))

    def forward(self, x: Tensor) -> Tensor:
        x = x * self.__importance_weight
        return super().forward(x)

    def get_importance_weights(self) -> np.ndarray:
        return self.__importance_weight.detach().numpy()


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
    model = ImportanceWeightedCNN(logger=logger)
    print(model)
    image = torch.zeros(1, 3, 32, 32)
    for name, param in model.named_parameters():
        print(f"{name}: {param.shape}")
    # print(image)
    # print(model.forward(image))
    # print(model)


if __name__ == "__main__":
    main()
