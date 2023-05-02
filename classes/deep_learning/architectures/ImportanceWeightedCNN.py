import logging

import colorlog
import numpy as np
import torch
import yaml
from torch import Tensor, nn

from classes.deep_learning.architectures.CNN import CNN


class ImportanceWeightedCNN(CNN):

    def __init__(self, config_path: str = "config/architectures/importance_weighted_cnn.yaml",
                 logger: logging.Logger = logging.getLogger(__name__)):
        # ----------------------------------------------------
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        weight = nn.Parameter(torch.ones(*cfg['importance_weight_layer']))
        # ----------------------------------------------------
        logger.info(f'{"-" * 95}')
        logger.info(f'{"   ":>1}{" ":>18}{" ":>3}{"params":>10}  {"module":<40}{"arguments":<30}')
        logger.info(f'{" ":>3}{" ":>18}{" ":>3}{np.prod(weight.shape):10.0f}  '
                    f'{str(weight.__class__)[8:-2]:<40}{cfg["importance_weight_layer"]}')
        # ----------------------------------------------------
        super(ImportanceWeightedCNN, self).__init__(config_path=config_path,
                                                    logger=logger)

        self.__importance_weight = weight

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
    image = torch.zeros(1, 3, 32, 32)

    # print(image)
    # print(model.forward(image))
    # print(model)


if __name__ == "__main__":
    main()
