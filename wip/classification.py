import logging

import numpy as np
import torch
import torchmetrics
import yaml
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from sklearn.svm import LinearSVC

from classes.clustering.Clusterer import Clusterer
from classes.core.Trainer import Trainer
from classes.data.KeypointPetDataset import KeypointPetDataset
from classes.data.OxfordIIITPetDataset import OxfordIIITPetDataset
from classes.data.Vocabulary import Vocabulary
from classes.deep_learning.FeedForwardNet import FeedForwardNet
from classes.feature_extraction.FeatureExtractingAlgorithm import FeatureExtractingAlgorithm
from functional.torch_utils import set_random_seed


def main():
    # --- Console output ---
    logger = logging.getLogger(__name__)
    # --- Load configurations ---
    with open('../config/training/training_configuration.yaml', 'r') as f:
        config: dict = yaml.safe_load(f)
    with open('../config/training/hypeparameter_configuration.yaml', 'r') as f:
        hyperparameters: dict = yaml.safe_load(f)
    # with open('config/shi_thomasi_feature_extraction.yaml', 'r') as f:
    #     feature_extraction_config: dict = yaml.safe_load(f)
    with open('../config/clustering/clustering_params.yaml', 'r') as f:
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
    # ranking = clusterer.rank_clusters(flat_train_descriptors, centroids, labels, print_values=False)
    # words = [centroids[cluster_label] for (cluster_label, _) in ranking[:NUM_WORDS]]
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
    # ----------------------------------------------------------------------
    # Convert the data from torch tensors to numpy arrays
    data = []
    labels = []
    for batch in train_kp_loader:
        batch_data, batch_labels = batch
        data.append(batch_data.numpy())
        labels.append(batch_labels.numpy())

    data = np.concatenate(data)
    labels = np.concatenate(labels)

    # Create and train the SVM model
    svm_model = LinearSVC()
    svm_model.fit(data, labels)

    data = []
    labels = []
    for batch in test_kp_loader:
        batch_data, batch_labels = batch
        data.append(batch_data.numpy())
        labels.append(batch_labels.numpy())

    print(svm_model.score(data, labels))

    # --- Metrics for training ---
    # metric_collection = MetricCollection({
    #     'accuracy': torchmetrics.Accuracy(task="multiclass",
    #                                       num_classes=config["num_classes"]),
    # })
    # # # --- Training ---
    # # TODO: Transformer?
    # trainer = Trainer(FeedForwardNet,
    #                   config=config,
    #                   hyperparameters=hyperparameters,
    #                   metric_collection=metric_collection,
    #                   logger=logger)
    # trainer.train(train_kp_loader, test_kp_loader, vocab=vocab)


if __name__ == "__main__":
    main()
