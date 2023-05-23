from typing import Union, Dict, Tuple, List

import numpy as np
import torch
from torch import Tensor

from classes.data.Vocabulary import Vocabulary
from classes.deep_learning.architectures.ImportanceWeightedCNN import ImportanceWeightedCNN
from classes.deep_learning.models.TrainableModel import TrainableModel


class ModelImportanceWeightedCNN(TrainableModel):

    def __init__(self, device: torch.device, words: List[float]):
        super().__init__(device, ImportanceWeightedCNN())
        self.__vocabulary = Vocabulary(words)

    def predict(self, x: Union[Tensor, Dict]) -> Union[Tuple, Tensor]:
        """
        Performs a prediction using the network and returns the output logits
        """
        return self._network(self.__vocabulary.embed(x).float())

    def get_importance_weights(self) -> np.ndarray:
        return self._network.get_importance_weights()
