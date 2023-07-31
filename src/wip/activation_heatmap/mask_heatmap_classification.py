import argparse

import torch
import torchmetrics
import yaml
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection

from classes.data.MaskHeatmapPetDataset import MaskHeatmapPetDataset
from functional.utilities.cluster_utilities import prepare_clusters_and_features
from functional.utilities.data_utils import create_stratified_splits
from functional.utilities.utils import default_logger
from src.classes.core.Trainer import Trainer
from src.classes.deep_learning.CNN import CNN


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
    clusterer_train, descriptors_train, keypoints_train = prepare_clusters_and_features(config, clustering_config,
                                                                                        logger, train=True, clean=clean)
    # train_ds = HeatmapPetDataset(keypoints, descriptors, clusterer, train=True)
    train_ds = MaskHeatmapPetDataset(keypoints_train, descriptors_train, clusterer_train, train=True)
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
    clusterer_test, descriptors_test, keypoints_test = prepare_clusters_and_features(config,
                                                                                     clustering_config,
                                                                                     logger,
                                                                                     train=False)
    # test_ds = HeatmapPetDataset(keypoints, descriptors, clusterer, train=False)
    test_ds = MaskHeatmapPetDataset(keypoints_test, descriptors_test, clusterer_test, train=False)
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
    trainer = Trainer(CNN,
                      config=config,
                      hyperparameters=hyperparameters,
                      metric_collection=metric_collection,
                      logger=logger)
    trainer.train(train_dataloader=train_loader_ds,
                  val_dataloader=val_loader_ds,
                  test_dataloader=test_loader_ds)


if __name__ == "__main__":
    main()
