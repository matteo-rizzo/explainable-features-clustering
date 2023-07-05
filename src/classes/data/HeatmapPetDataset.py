import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset
from torchvision.datasets import OxfordIIITPet

from src.classes.clustering.Clusterer import Clusterer
from src.classes.data.OxfordIIITPetDataset import OxfordIIITPetDataset
from src.classes.feature_extraction.FeatureExtractingAlgorithm import FeatureExtractingAlgorithm
from src.functional.image_handling import kps_to_heatmaps
from src.functional.utils import default_logger
from src.wip.activation_heatmap.workbench import draw_activation
from src.wip.cluster_extraction import extract_and_cluster


class HeatmapPetDataset(Dataset):
    def __init__(self, keypoints: list,
                 descriptors: list,
                 clustering: Clusterer,
                 root: str = "dataset",
                 train: bool = True):
        self.data = OxfordIIITPet(root=root,
                                  split="trainval" if train else "test",
                                  download=True)
        self.keypoints: list[tuple[cv2.KeyPoint]] = keypoints
        self.descriptors: list[np.ndarray] = descriptors
        self.clustering: Clusterer = clustering

    def __getitem__(self, index: int):
        _, label = self.data[index]
        heatmap = kps_to_heatmaps(self.keypoints[index],
                                  self.clustering.predict(self.descriptors[index]),
                                  (self.clustering.n_clusters(), 224, 224))
        return heatmap, label

    def __len__(self):
        return len(self.data)


def main():
    # --- Config ---
    with open('config/training/training_configuration.yaml', 'r') as f:
        generic_config: dict = yaml.safe_load(f)
    with open('config/clustering/clustering_params.yaml', 'r') as f:
        clustering_config: dict = yaml.safe_load(f)
    # --- Logger ---
    logger = default_logger(generic_config["logger"])

    key_points_extractor = FeatureExtractingAlgorithm(algorithm="SIFT", logger=logger)

    train_loader = torch.utils.data.DataLoader(OxfordIIITPetDataset(train=True, augment=False),
                                               batch_size=1,
                                               shuffle=False,
                                               num_workers=generic_config["workers"],
                                               drop_last=False)

    clusterer, descriptors, keypoints = extract_and_cluster(clustering_config,
                                                            key_points_extractor,
                                                            logger,
                                                            train_loader)

    ds = HeatmapPetDataset(keypoints, descriptors, clusterer, train=True)

    loader_ds = torch.utils.data.DataLoader(ds,
                                            batch_size=50,
                                            shuffle=False,
                                            num_workers=generic_config["workers"],
                                            drop_last=False)

    for heat, label in loader_ds:
        for x in heat:
            draw_activation(x)
            break


if __name__ == "__main__":
    main()
