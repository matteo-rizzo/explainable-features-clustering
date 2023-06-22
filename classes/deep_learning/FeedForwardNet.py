import torch.nn as nn
import torch.nn.functional

from classes.data.Vocabulary import Vocabulary


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
        self.fc2 = nn.Linear(hidden_size, hidden_size*2)
        self.fc3 = nn.Linear(hidden_size*2, num_classes)
        # POOLING?

        # self.vocab = vocab # FIXME: no, can't save cv2 keypoints

    def forward(self, x):
        # x = self.vocab.embed(x).to("cuda:0") # FIXME: ULTRA WRONG
        # x = torch.nn.functional.softmax(x) # TODO Softmax with max number features?
        x.to("cuda:0")
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out
