import torch.nn as nn

from classes.data.Vocabulary import Vocabulary


# Define the architecture of the feedforward network
class FeedForwardNet(nn.Module):
    def __init__(self, config: dict, vocab: Vocabulary, **kwarge):
        super(FeedForwardNet, self).__init__()

        num_channels, num_classes = (config["num_channels"],
                                     config["num_classes"])

        hidden_size = 2000

        self.fc1 = nn.Linear(num_channels, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)

        self.vocab = vocab # FIXME: no, can't save cv2 keypoints

    def forward(self, x):
        x = self.vocab.embed(x).to("cuda:0") # FIXME: ULTRA WRONG
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out
