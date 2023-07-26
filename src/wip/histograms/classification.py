import logging

import numpy as np
import torch
import torchmetrics
import yaml
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection

from src.classes.clustering.Clusterer import Clusterer
from src.classes.core.Trainer import Trainer
from src.classes.data.KeypointPetDataset import KeypointPetDataset
from src.classes.data.OxfordIIITPetDataset import OxfordIIITPetDataset
from src.classes.data.Vocabulary import Vocabulary
from src.classes.deep_learning.FeedForwardNet import FeedForwardNet
from src.classes.feature_extraction.FeatureExtractingAlgorithm import FeatureExtractingAlgorithm
from functional.utilities.torch_utils import set_random_seed


def main():
    NUM_WORDS: int = 100
    # --- Console output ---
    logger = logging.getLogger(__name__)
    # --- Load configurations ---
    with open('../../../config/training/training_configuration.yaml', 'r') as f:
        config: dict = yaml.safe_load(f)
    with open('../../../config/training/hypeparameter_configuration.yaml', 'r') as f:
        hyperparameters: dict = yaml.safe_load(f)
    # with open('config/shi_thomasi_feature_extraction.yaml', 'r') as f:
    #     feature_extraction_config: dict = yaml.safe_load(f)
    with open('../../../config/clustering/clustering_params.yaml', 'r') as f:
        clustering_config: dict = yaml.safe_load(f)
    # --- Data loaders ---
    # TODO: necessary for now for the keypoint dataset!
    set_random_seed(42, config["device"])
    train_loader = torch.utils.data.DataLoader(OxfordIIITPetDataset(train=True, augment=False),
                                               batch_size=config["batch_size"],
                                               shuffle=True,
                                               num_workers=config["workers"],
                                               drop_last=False)
    test_loader = torch.utils.data.DataLoader(OxfordIIITPetDataset(train=False, augment=False),
                                              batch_size=config["batch_size"],
                                              shuffle=True,
                                              num_workers=config["workers"],
                                              drop_last=False)
    # --- Feature extraction ---
    key_points_extractor = FeatureExtractingAlgorithm(algorithm="SIFT", logger=logger)

    train_keypoints, train_descriptors = key_points_extractor.get_keypoints_and_descriptors(train_loader)
    test_keypoints, test_descriptors = key_points_extractor.get_keypoints_and_descriptors(test_loader)
    # Gather all descriptors in a single big array
    flat_train_descriptors = np.concatenate(train_descriptors)
    # --- Clustering ---
    clusterer = Clusterer(algorithm="KMEANS", logger=logger, **clustering_config["kmeans_args"])
    labels = clusterer.fit_predict(flat_train_descriptors)
    centroids = clusterer.get_centroids()
    # --- Cluster plotting ---
    ranking = clusterer.rank_clusters(flat_train_descriptors, centroids, labels, print_values=False)
    words = [centroids[cluster_label] for (cluster_label, _) in ranking[:NUM_WORDS]]
    # --- AAAAAAAAA ---
    vocab = Vocabulary(centroids, clusterer)
    # TODO: just to be sure
    set_random_seed(42, config["device"])
    train_kp_loader = torch.utils.data.DataLoader(
        KeypointPetDataset(keypoints=train_keypoints,
                           descriptors=train_descriptors,
                           vocab=vocab, train=True),
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["workers"],
        drop_last=True)

    test_kp_loader = torch.utils.data.DataLoader(
        KeypointPetDataset(keypoints=test_keypoints,
                           descriptors=test_descriptors,
                           vocab=vocab, train=False),
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["workers"],
        drop_last=True)

    # --- Metrics for training ---
    metric_collection = MetricCollection({
        'accuracy': torchmetrics.Accuracy(task="multiclass",
                                          num_classes=config["num_classes"]),
    })
    # # # --- Training ---
    # # TODO: Transformer?
    trainer = Trainer(FeedForwardNet,
                      config=config,
                      hyperparameters=hyperparameters,
                      metric_collection=metric_collection,
                      logger=logger)
    trainer.train(train_kp_loader, test_kp_loader, vocab=vocab)


if __name__ == "__main__":
    main()
