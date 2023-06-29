from typing import Union

import torch.nn as nn
from torch.nn.common_types import _size_2_t


def autopad(kernel_size: _size_2_t,
            padding: Union[str, _size_2_t] = None):  # kernel, padding
    # Pad to 'same'
    if padding is None:
        padding = kernel_size // 2 if isinstance(kernel_size, int) else [x // 2 for x in kernel_size]  # auto-pad
    return padding


class Conv(nn.Module):
    # Standard convolution
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_2_t = 1,
                 stride: _size_2_t = 1,
                 padding: Union[str, _size_2_t] = None,
                 groups: int = 1,
                 activation: Union[nn.Module, bool] = True) \
            -> None:
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=autopad(kernel_size, padding),
                              groups=groups,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU() if activation is True else (
            activation if isinstance(activation, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))
