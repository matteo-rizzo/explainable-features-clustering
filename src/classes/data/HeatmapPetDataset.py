import cv2
import numpy as np
import torch
import yaml
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torchvision.datasets import OxfordIIITPet
from tqdm import tqdm

from src.classes.clustering.Clusterer import Clusterer
from src.classes.data.OxfordIIITPetDataset import OxfordIIITPetDataset
from src.classes.feature_extraction.FeatureExtractingAlgorithm import FeatureExtractingAlgorithm
from src.functional.image_handling import kps_to_heatmaps
from src.visualization.image_visualization import draw_activation
from src.functional.utils import default_logger
from src.wip.cluster_extraction import extract_and_cluster


class HeatmapPetDataset(Dataset):
    # TODO: img size is hard coded
    def __init__(self, keypoints: list,
                 descriptors: list,
                 clustering: Clusterer,
                 root: str = "dataset",
                 train: bool = True,
                 preload: bool = False):
        self.data = OxfordIIITPet(root=root,
                                  split="trainval" if train else "test",
                                  download=True)
        self.keypoints: list[tuple[cv2.KeyPoint]] = keypoints
        self.descriptors: list[np.ndarray] = descriptors
        self.clustering: Clusterer = clustering
        self.preloaded_heatmaps: list | None = None

        # FIXME: takes way too much RAM, doesn't work.
        if preload:
            self.preloaded_heatmaps = []
            for index, (img, label) in tqdm(enumerate(self.data), desc="Preloading heatmaps...", total=len(self.data)):
                heatmap = kps_to_heatmaps(self.keypoints[index],
                                          self.clustering.predict(self.descriptors[index]),
                                          (self.clustering.n_clusters(), 224, 224))
                self.preloaded_heatmaps.append((heatmap, label))

    def __getitem__(self, index: int):
        if not self.preloaded_heatmaps:
            # --- Has not been preloaded ---
            _, label = self.data[index]
            heatmap = kps_to_heatmaps(self.keypoints[index],
                                      self.clustering.predict(self.descriptors[index]),
                                      (self.clustering.n_clusters(), 224, 224))
            return heatmap, label
        else:
            # --- Has been preloaded ---
            return self.preloaded_heatmaps[index]

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
                                            batch_size=1,
                                            shuffle=False,
                                            num_workers=generic_config["workers"],
                                            drop_last=False)

    for i, ((heat, label), (img, label_)) in enumerate(zip(loader_ds, train_loader)):
        for x, y, z, u in zip(heat, label, img, label_):
            if i % 50 == 0:
                plt.imshow(z.squeeze())
                plt.title(train_loader.dataset.data.classes[u])
                plt.show()
                draw_activation(x, loader_ds.dataset.data.classes[y])


if __name__ == "__main__":
    main()
