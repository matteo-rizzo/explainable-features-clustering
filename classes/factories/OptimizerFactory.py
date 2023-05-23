import torch
import torch.nn as nn
from torch import optim


class OptimizerFactory:

    def __init__(self, network_parameters: nn.ParameterList, hyperparameters: dict):
        self.optimizers_map = {
            "SGD": optim.SGD(network_parameters, lr=hyperparameters["lr0"],
                             # momentum=hyperparameters['momentum'],
                             momentum=0.97,
                             nesterov=True),
            # "Adam": optim.Adam(network_parameters, lr=hyperparameters["lr0"],
            #                   betas=(hyperparameters['momentum'], 0.999)),
            # "AdamW": optim.AdamW(network_parameters, lr=hyperparameters["lr0"],
            #                     betas=(hyperparameters['momentum'], 0.999)),
        }

    def get(self, optimizer_type: str) -> torch.optim:
        if optimizer_type not in self.optimizers_map.keys():
            raise ValueError(
                f"Optimizer for {optimizer_type} is not implemented!"
                f"\nImplemented optimizers are: {list(self.optimizers_map.keys())}")
        return self.optimizers_map[optimizer_type]
