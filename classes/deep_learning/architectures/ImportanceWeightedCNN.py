import numpy as np
import torch
from torch import Tensor, nn

from classes.deep_learning.architectures.CNN import CNN


class ImportanceWeightedCNN(CNN):

    def __init__(self):
        super().__init__()
        self.__importance_weight = nn.Parameter(torch.ones(1, 3, 32, 32))

    def forward(self, x: Tensor) -> Tensor:
        x = x * self.__importance_weight
        return super().foward(x)

    def get_importance_weights(self) -> np.ndarray:
        return self.__importance_weight.detach().numpy()
