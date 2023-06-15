from torch import nn


class ActivationFactory:
    activation_map = {
        "Softmax": nn.Softmax(dim=1),
        "Sigmoid": nn.Sigmoid()
    }

    def get(self, activation_type: str) -> nn.Module:
        if activation_type not in self.activation_map.keys():
            raise ValueError(f"Activation for {activation_type} is not implemented!"
                             f"\nSupported activation are: {list(self.activation_map.keys())}")
        return self.activation_map[activation_type]
