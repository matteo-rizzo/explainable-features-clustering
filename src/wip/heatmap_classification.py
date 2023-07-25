import argparse
import logging

import torch
import torchmetrics
import yaml
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection

from classes.deep_learning.CNN3D import CNN3D
from src.classes.core.Trainer import Trainer
from src.classes.data.HeatmapPetDataset import HeatmapPetDataset
from src.classes.data.OxfordIIITPetDataset import OxfordIIITPetDataset
from src.classes.deep_learning.CNN import CNN
from src.classes.deep_learning.SmarterCNN import SmarterCNN
from src.classes.deep_learning.VGG16 import VGG16
from src.classes.feature_extraction.FeatureExtractingAlgorithm import FeatureExtractingAlgorithm
from src.functional.data_utils import create_stratified_splits
from src.functional.utils import default_logger
from src.wip.cluster_extraction import extract_and_cluster


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


def main():
    # --- Argument parsing ---
    parser = argparse.ArgumentParser(description="Heatmap classification script")
    parser.add_argument("--clean", action="store_true", default=False,
                        help="Perform a clean feature extraction and clustering (default: False)")
    args = parser.parse_args()
    clean: bool = args.clean
    # --- Config ---
    with open('config/training/training_configuration.yaml', 'r') as f:
        config: dict = yaml.safe_load(f)
    with open('config/clustering/clustering_params.yaml', 'r') as f:
        clustering_config: dict = yaml.safe_load(f)
    with open('config/training/hypeparameter_configuration.yaml', 'r') as f:
        hyperparameters: dict = yaml.safe_load(f)
    # --- Logger ---
    logger = default_logger(config["logger"])
    # TODO we should cluster on ALL data maybe? maybe not, seems weird
    # --- TRAIN DS ---
    clusterer, descriptors, keypoints = prepare_clusters_and_features(config, clustering_config,
                                                                      logger, train=True, clean=clean)
    train_ds = HeatmapPetDataset(keypoints, descriptors, clusterer, train=True)

    train_size: int = int(len(train_ds) * 0.8)
    test_size: int = len(train_ds) - train_size
    train_split, val_split = create_stratified_splits(train_ds, train_size=train_size, test_size=test_size)
    train_loader_ds = torch.utils.data.DataLoader(train_split,
                                                  batch_size=config["batch_size"],
                                                  shuffle=True,
                                                  num_workers=config["workers"],
                                                  drop_last=False)
    # --- VAL DS ---
    val_loader_ds = torch.utils.data.DataLoader(val_split,
                                                batch_size=config["batch_size"],
                                                shuffle=True,
                                                num_workers=config["workers"],
                                                drop_last=False)
    # --- TEST DS ---
    clusterer, descriptors, keypoints = prepare_clusters_and_features(config,
                                                                      clustering_config,
                                                                      logger,
                                                                      train=False)
    test_ds = HeatmapPetDataset(keypoints, descriptors, clusterer, train=False)
    test_loader_ds = torch.utils.data.DataLoader(test_ds,
                                                 batch_size=config["batch_size"],
                                                 shuffle=False,
                                                 num_workers=config["workers"],
                                                 drop_last=False)
    # --- Metrics for training ---
    metric_collection = MetricCollection({
        'accuracy': torchmetrics.Accuracy(task="multiclass",
                                          num_classes=config["num_classes"]),
        'macro_F1': torchmetrics.F1Score(task="multiclass",
                                         average="micro",
                                         num_classes=config["num_classes"]),
        'micro_F1': torchmetrics.F1Score(task="multiclass",
                                         average="micro",
                                         num_classes=config["num_classes"]),
    })
    # --- Training ---
    trainer = Trainer(CNN3D,
                      config=config,
                      hyperparameters=hyperparameters,
                      metric_collection=metric_collection,
                      logger=logger)
    trainer.train(train_dataloader=train_loader_ds,
                  val_dataloader=val_loader_ds,
                  test_dataloader=test_loader_ds)


if __name__ == "__main__":
    main()
