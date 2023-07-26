import time

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset
from torchvision.datasets import OxfordIIITPet

from src.classes.clustering.Clusterer import Clusterer
from src.classes.data.OxfordIIITPetDataset import OxfordIIITPetDataset
from src.classes.feature_extraction.FeatureExtractingAlgorithm import FeatureExtractingAlgorithm
from functional.utilities.image_handling import kps_to_heatmaps
from functional.utilities.utils import default_logger
from src.wip.cluster_extraction import extract_and_cluster


class TestHeatmapDataset(Dataset):
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
        # --- Has not been preloaded ---
        _, label = self.data[index]
        # heatmap = kps_to_heatmaps(self.keypoints[index],
        #                           self.clustering.predict(self.descriptors[index]),
        #                           (self.clustering.n_clusters(), 224, 224))
        # torch.save(heatmap, f"dataset/heatmaps/heatmap_{index}.pt")
        heatmap = torch.load(f"dataset/heatmaps/heatmap_{index}.pt")
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
    # --- Creation of dataset ---
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
    # --------------------------------------------------------------------------------
    ds = TestHeatmapDataset(keypoints, descriptors, clusterer, train=True)

    loader_ds = torch.utils.data.DataLoader(ds,
                                            batch_size=1,
                                            shuffle=False,
                                            num_workers=generic_config["workers"],
                                            drop_last=False)

    logger.info("----------------------------------------------")
    min_all = 0.
    sec_all = 0.
    loops = 10
    for x in range(loops):
        t0 = time.perf_counter()
        i = 0
        for _ in loader_ds:
            a = _
            i += 1
            # if i == 50:
            #     break

        t1 = time.perf_counter()
        minutes = (t1 - t0) // 60
        min_all += minutes
        seconds = (t1 - t0) % 60
        sec_all += seconds
        logger.info(f"[{x+1}/{loops} (first 50 data points only)] {minutes:.2f}m {seconds:.2f}s")
    logger.info("----------------------------------------------")
    logger.info(f"[AVG (first 50 data points only)] {min_all/loops:.2f}m {sec_all/loops:.2f}s")

    # for i, ((heat, label), (img, label_)) in enumerate(zip(loader_ds, train_loader)):
    #     for x, y, z, u in zip(heat, label, img, label_):
    #         if i % 50 == 0:
    #             plt.imshow(z.squeeze())
    #             plt.title(train_loader.dataset.data.classes[u])
    #             plt.show()
    #             draw_activation(x, loader_ds.dataset.data.classes[y])


if __name__ == "__main__":
    main()
