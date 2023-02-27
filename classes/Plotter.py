from typing import List

from torch import Tensor


class Plotter:

    def __init__(self):
        pass

    def plot_clusters(self, clusters: List[Tensor], save_fig: bool = True):
        pass

    def plot_importance(self, key_points: List[Tensor], importance: List[float], save_fig: bool = True):
        pass
