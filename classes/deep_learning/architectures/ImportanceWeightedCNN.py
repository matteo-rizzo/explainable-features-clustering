import logging
import os

import numpy as np
import torch
from torch import Tensor, nn

from classes.deep_learning.architectures.CNN import CNN

CONFIG_PATH = os.path.join("config", "architectures", "importance_weighted_cnn.yaml")


class ImportanceWeightedCNN(CNN):

    def __init__(self, config_path: str = CONFIG_PATH, logger: logging.Logger = logging.getLogger(__name__)):
        super(ImportanceWeightedCNN, self).__init__(config_path=config_path, logger=logger)

        importance_weights = torch.rand((1, 1, 1, 4410))
        importance_weights = importance_weights / importance_weights.sum()
        self.__importance_weights = nn.Parameter(importance_weights)

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x * self.__importance_weights)

    def get_importance_weights(self) -> np.ndarray:
        return self.__importance_weights.detach().numpy()
