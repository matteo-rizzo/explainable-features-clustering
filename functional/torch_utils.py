import os
from pathlib import Path

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
