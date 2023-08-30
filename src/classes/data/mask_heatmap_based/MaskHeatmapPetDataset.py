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
                 train: bool = True,
                 top_clusters: list[int] | None = None):
        transforms = T.Compose([
            T.ToTensor(),
            T.Resize(224, antialias=True),
            T.CenterCrop((224, 224)),
            # TODO: should we? It's the imagenet classic
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            T.Grayscale()
        ])
        self.data = OxfordIIITPet(root=root,
                                  transform=transforms,
                                  split="trainval" if train else "test",
                                  download=True)
        self.keypoints: list[tuple[cv2.KeyPoint]] = keypoints
        self.descriptors: list[np.ndarray] = descriptors
        self.clustering: Clusterer = clustering
        self.top_clusters: list[int] = top_clusters
        if top_clusters:
            # Map index of top clusters (which are in 0-n_clusters) to 0-n_top_clusters (e.g. 0-100)
            # k, v reverse because we want the mapping cluster_index to new index
            self.__cluster_index_map: dict = {k: v for v, k in enumerate(top_clusters)}

    def __getitem__(self, index: int):
        image, label = self.data[index]
        predictions = self.clustering.predict(self.descriptors[index])
        top_predictions = np.array([p if p in self.top_clusters else -1 for p in predictions])
        num_layers = self.clustering.n_clusters() if self.top_clusters is None else len(self.top_clusters)
        heatmap = kps_to_mask_heatmaps(image,
                                       self.keypoints[index],
                                       top_predictions,
                                       (num_layers, 224, 224),
                                       self.__cluster_index_map)
        return heatmap, label

    def __len__(self):
        return len(self.data)
