import logging

import torch.nn as nn
import yaml

from functional.arc_utils import parse_model


class TorchModel(nn.Module):
    def __init__(self, config_path: str, logger: logging.Logger):
        super(TorchModel, self).__init__()
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        self.model = parse_model(cfg, logger=logger)
        logger.info(f'{"-"*95}')

    def forward(self, x):
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
