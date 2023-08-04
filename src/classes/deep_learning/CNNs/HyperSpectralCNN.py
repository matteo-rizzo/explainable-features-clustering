import torch.nn as nn


class HyperSpectralCNN(nn.Module):

    def __init__(self, config: dict, **kwargs):
        # Convolutional neural networks for hyperspectral image classification
        super(HyperSpectralCNN, self).__init__()
        # ----------------------------------------------------------
        num_channels, img_size, num_classes = (config["num_channels"],
                                               config["img_size"],
                                               config["num_classes"])
        # ----------------------------------------------------------
        # 3 Convolutional layers
        # Conv WxHxN (N channels) (1x1 kernels) (128 filters) (output: WxHx128)
        # Norm (2x2) (on each channel)
        # Dropout

        # Conv WxHx64
        # Norm
        # Dropout

        # Conv WxHxC (C classes)
        # Global Average Pooling
        # Output 1xC
        # --------------
        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=128, kernel_size=1)
        self.norm1 = nn.BatchNorm2d(128)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout2d(0.2)
        # --------------
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1)
        self.relu2 = nn.ReLU()
        self.norm2 = nn.BatchNorm2d(64)
        self.dropout2 = nn.Dropout2d(0.2)
        # --------------
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        # --------------

    def forward(self, x):
        # --------------
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.norm1(x)
        x = self.dropout1(x)
        # --------------
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.norm2(x)
        x = self.dropout2(x)
        # --------------
        x = self.conv3(x)
        x = self.gap(x)
        # --------------
        x = x.view(x.size(0), -1)
        return x
