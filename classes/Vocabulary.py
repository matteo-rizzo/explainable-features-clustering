from typing import List

from torch import Tensor


class Vocabulary:

    def __int__(self):
        pass

    def build(self, data: List[Tensor]):
        pass

    def embed(self, data: Tensor) -> Tensor:
        pass
