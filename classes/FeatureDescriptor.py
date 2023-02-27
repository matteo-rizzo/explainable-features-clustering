from torch import Tensor


class FeatureDescriptor:

    def __init__(self):
        self.__descriptor = None

    def describe(self, x: Tensor) -> Tensor:
        return self.__descriptor(x)
