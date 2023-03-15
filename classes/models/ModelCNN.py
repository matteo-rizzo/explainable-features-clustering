import torch

from classes.core.Model import Model
from classes.models.NetworkCNN import NetworkCNN


class ModelCNN(Model):

    def __init__(self, device: torch.device):
        super().__init__(device)
        self._network = NetworkCNN()
