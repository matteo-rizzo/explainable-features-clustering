from typing import List, Tuple

from torch import Tensor


class Clusterizer:

    def __init__(self):
        pass

    def cluster(self, data: List[Tensor]) -> List[Tensor]:
        pass

    def rank_clusters(self, data: List[Tensor]) -> List[Tensor]:
        pass

    def rank_features(self, data: List[Tensor], ranking: List[Tensor]) -> Tuple[List[Tensor], List[float]]:
        pass
