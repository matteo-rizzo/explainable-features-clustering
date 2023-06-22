import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets import OxfordIIITPet

from classes.data.Vocabulary import Vocabulary


class KeypointPetDataset(Dataset):
    def __init__(self, keypoints: list, descriptors: list, vocab: Vocabulary,
                 root: str = "dataset", train: bool = True,
                 ):
        self.data = OxfordIIITPet(root=root,
                                  split="trainval" if train else "test",
                                  download=True)
        self.keypoints: list[tuple[cv2.KeyPoint]] = keypoints
        self.descriptors: list[np.ndarray] = descriptors
        self.vocab: Vocabulary = vocab



    def __getitem__(self, index: int):
        _, label = self.data[index]
        # img = self.vocab.embed(self.keypoints[index], self.descriptors[index])
        img = self.vocab.embed_unordered(self.descriptors[index])
        return img, label

    def __len__(self):
        return len(self.data)
