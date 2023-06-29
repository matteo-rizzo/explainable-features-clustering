import math
from typing import Callable


def one_cycle_lrs(y1: float = 0.0, y2: float = 1.0, steps: int = 100) -> Callable:
    # lambda function for sinusoidal ramp from y1 to y2
    # One Cycle Policy (Leslie Smith)
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1


def linear_lrs(steps: int, lrf) -> Callable:
    # Linear learning rate schedule
    return lambda x: (1 - x / (steps - 1)) * (1.0 - lrf) + lrf
