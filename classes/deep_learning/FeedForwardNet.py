import torch.nn as nn


# Define the architecture of the feedforward network
class FeedForwardNet(nn.Module):
    def __init__(self, config: dict, **kwargs):
        super(FeedForwardNet, self).__init__()

        num_channels, num_classes = (config["num_channels"],
                                     config["num_classes"])

        hidden_size: int = 2000
        # TODO: perhaps I should use an embedding layer?!
        self.fc1 = nn.Linear(num_channels, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size * 2)
        self.fc3 = nn.Linear(hidden_size * 2, num_classes)
        # TODO: POOLING?

    def forward(self, x):
        # x.to("cuda:0")
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out
