import logging

import torch

from classes.data.OxfordIIITPetDataset import OxfordIIITPetDataset
from classes.feature_extraction.FeatureExtractingAlgorithm import FeatureExtractingAlgorithm
from wip.cluster_extraction import extract_and_cluster


def prepare_clusters_and_features(config: dict, clustering_config: dict, logger: logging.Logger,
                                  train: bool, clean: bool = False):
    key_points_extractor = FeatureExtractingAlgorithm(algorithm="SIFT", logger=logger)

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
                                                            clean=clean)
    return clusterer, descriptors, keypoints
