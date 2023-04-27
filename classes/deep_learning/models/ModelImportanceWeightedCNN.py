import numpy as np
import torch

from classes.deep_learning.architectures.ImportanceWeightedCNN import ImportanceWeightedCNN
from classes.deep_learning.models.Model import Model


class ModelImportanceWeightedCNN(Model):

    def __init__(self, device: torch.device):
        super().__init__(device, ImportanceWeightedCNN())

    def get_importance_weights(self) -> np.ndarray:
        return self._network.get_importance_weights()
