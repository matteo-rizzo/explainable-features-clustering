from typing import List

from torch import Tensor


class KeyPointsExtractor:

    def __init__(self):
        self.__extractor = None

    def extract(self, x: Tensor) -> List[Tensor]:
        return self.__extractor(x)
