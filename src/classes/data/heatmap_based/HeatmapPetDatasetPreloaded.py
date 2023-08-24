import torch

from classes.data.heatmap_based.HeatmapPetDataset import HeatmapPetDataset
from src.classes.clustering.Clusterer import Clusterer


class HeatmapPetDatasetPreloaded(HeatmapPetDataset):
    def __init__(self, keypoints: list, descriptors: list,
                 clustering: Clusterer, root: str = "dataset",
                 train: bool = True):
        super().__init__(keypoints, descriptors, clustering, root, train)
        self.split: str = str(int(train))

    def __getitem__(self, index: int):
        _, label = self.data[index]
        heatmap = torch.load(f"dataset/heatmaps/heatmap_{self.split}_{index}.pt")
        return heatmap, label

    def __len__(self):
        return len(self.data)
