import logging
import yaml
import torch

from classes.data.OxfordIIITPetDataset import OxfordIIITPetDataset
from classes.feature_extraction.FeatureExtractingAlgorithm import FeatureExtractingAlgorithm
from functional.utilities.cluster_extraction import extract_and_cluster


def prepare_clusters_and_features(config: dict, clustering_config: dict, logger: logging.Logger,
                                  train: bool, clean: bool = False, clustering_algorithm: str = "kmeans"):
    with open('config/feature_extraction/SIFT_config.yaml', 'r') as f:
        sift_config: dict = yaml.safe_load(f)
    key_points_extractor = FeatureExtractingAlgorithm(algorithm="SIFT", logger=logger,
                                                      **sift_config)

    # FeatureExtractingAlgorithm(algorithm="SIFT")

    train_loader = torch.utils.data.DataLoader(OxfordIIITPetDataset(train=train, augment=False),
                                               batch_size=1,
                                               shuffle=False,
                                               num_workers=config["workers"],
                                               drop_last=False)

    clusterer, descriptors, keypoints = extract_and_cluster(clustering_config,
                                                            key_points_extractor,
                                                            logger,
                                                            train_loader,
                                                            train,
                                                            clean=clean,
                                                            clustering_algorithm=clustering_algorithm)
    return clusterer, descriptors, keypoints
