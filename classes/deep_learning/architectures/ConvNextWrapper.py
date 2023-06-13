import torch.nn as nn
from torch import Tensor
from torchvision.models import convnext_small, convnext_base, convnext_tiny
from torchvision.models.convnext import LayerNorm2d


class ConvNextWrapper(nn.Module):

    def __init__(self, config: dict, **kwargs):
        super().__init__()
        pretrained_model = convnext_tiny(weights='DEFAULT')  # model
        n_inputs = pretrained_model.classifier._modules['2'].in_features
        # This is the same classifier they use in the paper, just replacing the head
        sequential_layers = nn.Sequential(
            LayerNorm2d([n_inputs,], eps=1e-06, elementwise_affine=True),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(n_inputs, config["num_classes"], bias=True),
        )
        pretrained_model.classifier = sequential_layers

        self.model = pretrained_model

    def forward(self, x: Tensor) -> Tensor:
        return self.model.forward(x)
