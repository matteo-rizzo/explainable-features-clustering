import logging

import numpy as np
import torch
import torchmetrics
import yaml
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from tqdm import tqdm

from classes.FeatureExtractingAlgorithm import FeatureExtractingAlgorithm
from classes.clustering.Clusterer import Clusterer
from classes.clustering.DimensionalityReducer import DimensionalityReducer
from classes.clustering.KMeansClustering import KMeansClustering
from classes.core.Trainer import Trainer
from classes.data.Food101Dataset import Food101Dataset
from classes.data.MNISTDataset import MNISTDataset
from classes.data.Vocabulary import Vocabulary
from classes.deep_learning.models.ModelImportanceWeightedCNN import ModelImportanceWeightedCNN
from functional.data_utils import create_stratified_splits
from functional.torch_utils import get_device

PLOT = False
NUM_WORDS = 100


def main():
    # --- Console output ---
    logger = logging.getLogger(__name__)
    # --- Load configurations ---
    with open('config/training/training_configuration.yaml', 'r') as f:
        config: dict = yaml.safe_load(f)
    with open('config/training/hypeparameter_configuration.yaml', 'r') as f:
        hyperparameters: dict = yaml.safe_load(f)
    # with open('config/shi_thomasi_feature_extraction.yaml', 'r') as f:
    #     feature_extraction_config: dict = yaml.safe_load(f)
    with open('config/clustering/clustering_params.yaml', 'r') as f:
        clustering_config: dict = yaml.safe_load(f)
    # --- Data loaders ---
    # train_loader = torch.utils.data.DataLoader(MNISTDataset(train=True),
    #                                     batch_size=config["batch_size"],
    #                                     shuffle=True,
    #                                     num_workers=config["workers"],
    #                                     drop_last=True)
    # test_loader = torch.utils.data.DataLoader(MNISTDataset(train=False),
    #                                    batch_size=config["batch_size"],
    #                                    shuffle=True,
    #                                    num_workers=config["workers"],
    #                                    drop_last=True)

    # TODO: Working on a subsample
    train_subset, test_subset = create_stratified_splits(Food101Dataset(train=True),
                                                         n_splits=1, train_size=5050, test_size=1010)
    train_loader = torch.utils.data.DataLoader(train_subset,
                                        batch_size=config["batch_size"],
                                        shuffle=True,
                                        num_workers=config["workers"],
                                        drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_subset,
                                       batch_size=config["batch_size"],
                                       shuffle=True,
                                       num_workers=config["workers"],
                                       drop_last=True)

    # --- Feature extraction ---
    key_points_extractor = FeatureExtractingAlgorithm(algorithm="SIFT", logger=logger)

    train_keypoints, train_descriptors = key_points_extractor.get_keypoints_and_descriptors(train_loader, rgb=True)
    test_keypoints, test_descriptors = key_points_extractor.get_keypoints_and_descriptors(test_loader, rgb=True)
    # Gather all descriptors in a single big array
    flat_train_descriptors = np.concatenate(train_descriptors)
    flat_test_descriptors = np.concatenate(test_descriptors)
    # --- Clustering ---
    clusterer = Clusterer(algorithm="KMEANS", logger=logger, **clustering_config["kmeans_args"])
    labels = clusterer.fit_predict(flat_train_descriptors)
    centroids = clusterer.get_centroids()
    # --- Cluster plotting ---
    if PLOT:
        dimensionality_reducer = DimensionalityReducer(algorithm="UMAP", logger=logger, **clustering_config["umap_args_2d"])
        vectors_2d = dimensionality_reducer.fit_transform(flat_train_descriptors)
        clusterer.plot(vectors_2d, labels, "KMEANS")
    # Plot the data points and centroids
    # if PLOT:
    #     plt.scatter(flat_descriptors[:, 0], flat_descriptors[:, 1], c=labels)
    #     plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=100, linewidths=3, c='k')
    #     plt.show()

    ranking = clusterer.rank_clusters(flat_train_descriptors, centroids, labels, print_values=True)
    words = [centroids[cluster_label] for (cluster_label, _) in ranking[:NUM_WORDS]]

    vocab = Vocabulary(words)
    # device = get_device(DEVICE_TYPE)
    #
    # model = ModelImportanceWeightedCNN(device, words)
    # model.set_optimizer(OPTIMIZER, LEARNING_RATE)
    # model.set_criterion(CRITERION)
    # model.train_mode()
    #
    # for epoch in range(EPOCHS):
    #
    #     running_loss, correct, total = 0.0, 0, 0
    #     for i, (x, y) in tqdm(enumerate(train), desc="Training epoch: {}".format(epoch)):
    #         x, y = x.to(device), y.to(device)
    #         o = model.predict(x).to(device)
    #         print(o.shape)
    #         loss = model.update_weights(o, y)
    #         running_loss += loss
    #         total, correct = model.get_accuracy(o, y, total, correct)
    #
    #     train_loss, train_accuracy = running_loss / len(train), 100 * correct / total
    #
    #     running_loss, correct, total = 0.0, 0, 0
    #     for i, (x, y) in tqdm(enumerate(test), desc="Testing epoch: {}".format(epoch)):
    #         x, y = x.to(device), y.to(device)
    #         o = model.predict(x).to(device)
    #         loss = model.get_loss(o, y)
    #         running_loss += loss
    #         total, correct = model.get_accuracy(o, y, total, correct)
    #
    #     test_loss, test_accuracy = running_loss / len(test), 100 * correct / total
    #
    #     print(f'Epoch [{epoch + 1:d}], '
    #           f'train loss: {train_loss:.3f}, '
    #           f'train accuracy: {train_accuracy:.3f}, '
    #           f'test loss: {test_loss:.3f}, '
    #           f'test accuracy: {test_accuracy:.3f}')
    #
    # # ----------------------------------------------------------------------
    # # --- Metrics for training ---
    # metric_collection = MetricCollection({
    #     'accuracy': torchmetrics.Accuracy(task="multiclass", num_classes=10),
    #     'precision': torchmetrics.Precision(task="multiclass", num_classes=10, average="macro"),
    #     'recall': torchmetrics.Recall(task="multiclass", num_classes=10, average="macro"),
    #     "F1": torchmetrics.F1Score(task="multiclass", num_classes=10, average="macro")
    # })
    # # --- Training ---
    # trainer = Trainer(ModelImportanceWeightedCNN, config=config,
    #                   hyperparameters=hyperparameters,
    #                   metric_collection=metric_collection, logger=logger)
    # trainer.train(train_loader, test_loader)


if __name__ == "__main__":
    main()
