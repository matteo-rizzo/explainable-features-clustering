import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.datasets import OxfordIIITPet

from classes.clustering.Clusterer import Clusterer
from functional.utilities.image_handling import kps_to_mask_heatmaps


class MaskHeatmapPetDataset(Dataset):
    # TODO: img size is hard coded
    def __init__(self, keypoints: list,
                 descriptors: list,
                 clustering: Clusterer,
                 root: str = "dataset",
                 train: bool = True):
        transforms = T.Compose([
            T.Resize(224),
            T.CenterCrop((224, 224)),
            T.Grayscale(),
            T.ToTensor()
        ])
        self.data = OxfordIIITPet(root=root,
                                  transform=transforms,
                                  split="trainval" if train else "test",
                                  download=True)
        self.keypoints: list[tuple[cv2.KeyPoint]] = keypoints
        self.descriptors: list[np.ndarray] = descriptors
        self.clustering: Clusterer = clustering


    def __getitem__(self, index: int):
        image, label = self.data[index]
        heatmap = kps_to_mask_heatmaps(image,
                                       self.keypoints[index],
                                       self.clustering.predict(self.descriptors[index]),
                                       (self.clustering.n_clusters(), 224, 224))
        return heatmap, label

    def __len__(self):
        return len(self.data)
