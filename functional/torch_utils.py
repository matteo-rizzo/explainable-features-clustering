import os
import random
import re
from pathlib import Path

import numpy as np
import torch


def strip_optimizer(ckpt_path: Path = Path('best.pt'), desc: str = ''):
    # Strip optimizer from 'f' to finalize training, optionally save as 's'
    x = torch.load(ckpt_path, map_location=torch.device('cpu'))
    if x.get('ema'):
        x['model'] = x['ema']  # replace model with ema
    for k in 'optimizer', 'training_results', 'wandb_id', 'ema', 'updates':  # keys
        x[k] = None
    x['epoch'] = -1
    x['model'].half()  # to FP16
    for p in x['model'].parameters():
        p.requires_grad = False
    torch.save(x, desc or ckpt_path)
    mb = os.path.getsize(desc or ckpt_path) / 1E6  # filesize
    print(f"Optimizer stripped from {ckpt_path},{(' saved as %s,' % desc) if desc else ''} {mb:.1f}MB")


def set_random_seed(seed: int, device: str):
    """
    Set specific seed for reproducibility.

    :param seed: int, the seed to set
    :param device: cuda:number or cpu
    :return:
    """
    torch.manual_seed(seed)
    if device[:4] == 'cuda':
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_device(device_type: str = "cpu") -> torch.device:
    """
    Returns the device specified in the experiments parameters (if available, else fallback to a "cpu" device)
    :param device_type: the id of the selected device (if cuda device, must match the regex "cuda:\d"
    :return: the device specified in the experiments parameters (if available, else fallback to a "cpu" device)
    """
    if device_type == "cpu":
        return torch.device("cpu")

    if re.match(r"\bcuda:\b\d+", device_type):
        if not torch.cuda.is_available():
            print(f"WARNING: running on cpu since device {device_type} is not available")
            return torch.device("cpu")
        return torch.device(device_type)

    raise ValueError(f"ERROR: {device_type} is not a valid device! Supported device are 'cpu' and 'cuda:n'")
