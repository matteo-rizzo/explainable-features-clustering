import os

import numpy as np
import torch
from torch import Tensor, nn

from classes.deep_learning.CNNs.CNN import CNN

CONFIG_PATH = os.path.join("config", "architectures", "importance_weighted_cnn.yaml")


class ImportanceWeightedCNN(CNN):

    def __init__(self, **kwargs):
        super(ImportanceWeightedCNN, self).__init__()

        # FIXME: should have a different shape. Also not rand; do a smart initialization
        self.__importance_weights = nn.Parameter(torch.ones(1, 1, 28, 28))
        # importance_weights = torch.rand((1, 1, 1, 4410))
        # importance_weights = importance_weights / importance_weights.sum()
        # self.__importance_weights = nn.Parameter(importance_weights)

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x * self.__importance_weights)

    def get_importance_weights(self) -> np.ndarray:
        return self.__importance_weights.detach().numpy()
