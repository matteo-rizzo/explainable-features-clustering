import time

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset
from torchvision.datasets import OxfordIIITPet
from tqdm import tqdm

from functional.utilities.image_handling import kps_to_heatmaps
from functional.utilities.utils import default_logger
from src.classes.clustering.Clusterer import Clusterer
from src.classes.data.OxfordIIITPetDataset import OxfordIIITPetDataset
from src.classes.feature_extraction.FeatureExtractingAlgorithm import FeatureExtractingAlgorithm
from functional.utilities.cluster_extraction import extract_and_cluster


# TODO: refactor / Just use base class
class SaveHeatmapDataset(Dataset):
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
        self.split: str = str(int(train))

    def __getitem__(self, index: int):
        _, label = self.data[index]  # else:
        # Calculate on the fly
        heatmap_pre = kps_to_heatmaps(self.keypoints[index],
                                      self.clustering.predict(self.descriptors[index]),
                                      (self.clustering.n_clusters(), 224, 224))
        torch.save(heatmap_pre, f"dataset/heatmaps/heatmap_{self.split}_{index}.pt")
        heatmap_post = torch.load(f"dataset/heatmaps/heatmap_{self.split}_{index}.pt")
        # assert heatmap_pre == heatmap_post
        return heatmap_post, label

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
    # --- Creation of dataset ---
    for train in [True, False]:
        key_points_extractor = FeatureExtractingAlgorithm(algorithm="SIFT", logger=logger)
        train_loader = torch.utils.data.DataLoader(OxfordIIITPetDataset(train=train, augment=False),
                                                   batch_size=1,
                                                   shuffle=False,
                                                   num_workers=generic_config["workers"],
                                                   drop_last=False)
        clusterer, descriptors, keypoints = extract_and_cluster(clustering_config,
                                                                key_points_extractor,
                                                                logger,
                                                                train_loader,
                                                                train)
        # --------------------------------------------------------------------------------
        ds = SaveHeatmapDataset(keypoints, descriptors, clusterer, train=train)

        loader_ds = torch.utils.data.DataLoader(ds,
                                                batch_size=1,
                                                shuffle=False,
                                                num_workers=generic_config["workers"],
                                                drop_last=False)

        logger.info("----------------------------------------------")
        t0 = time.perf_counter()
        for _ in tqdm(loader_ds, total=len(ds), desc=f"Saving {'train' if train else 'test'} dataset heatmaps"):
            a = _
            # Just get to save
        t1 = time.perf_counter()
        minutes = (t1 - t0) // 60
        seconds = (t1 - t0) % 60
        logger.info(f"Saved {'train' if train else 'test'} heatmaps in {minutes:.2f}m {seconds:.2f}s")
        logger.info("----------------------------------------------")


if __name__ == "__main__":
    main()
