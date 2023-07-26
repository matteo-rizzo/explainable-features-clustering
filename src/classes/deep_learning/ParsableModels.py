import logging

import colorlog
import numpy as np
import torch
import yaml
from torch import Tensor, nn

from functional.utilities.arc_utils import parse_model


class TorchParsableModel(nn.Module):
    def __init__(self, config_path: str, logger: logging.Logger):
        super(TorchParsableModel, self).__init__()
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        self.model = parse_model(cfg, logger=logger)
        logger.info(f'{"-" * 95}')

    def forward(self, x) -> Tensor:
        y, dt = [], []  # outputs
        for layer in self.model:
            # layer.f indicates "where from"
            # If not from previous layer (from earlier layers)
            if layer.from_layer != -1:
                x = (y[layer.from_layer] if isinstance(layer.from_layer, int)
                     else [x if j == -1 else y[j] for j in layer.from_layer])
            # Pass in current layer
            x = layer(x)
        return x


class ParsableCNN(TorchParsableModel):

    def __init__(self, config_path: str = "config/architectures/cnn.yaml",
                 logger: logging.Logger = logging.getLogger(__name__)):
        if config_path == "default":
            config_path = "config/architectures/cnn.yaml"
        logger.info(f'{"-" * 95}')
        super(ParsableCNN, self).__init__(config_path=config_path, logger=logger)

    def forward(self, x) -> Tensor:
        return super().forward(x)


class ParsableImportanceWeightedCNN(ParsableCNN):

    def __init__(self, config_path: str = "config/architectures/importance_weighted_cnn.yaml",
                 logger: logging.Logger = logging.getLogger(__name__)):
        if config_path == "default":
            config_path = "config/architectures/importance_weighted_cnn.yaml"
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
        super(ParsableImportanceWeightedCNN, self).__init__(config_path=config_path,
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
    model = ParsableCNN(logger=logger)
    image = torch.zeros(1, 3, 32, 32)

    # print(image)
    # print(model.forward(image))
    # print(model)


if __name__ == "__main__":
    main()
