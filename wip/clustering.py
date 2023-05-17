import numpy as np
import yaml
from torch.utils.data import DataLoader

from classes.FeatureExtractingAlgorithm import FeatureExtractingAlgorithm
from classes.clustering.Clusterer import Clusterer
from classes.clustering.DimensionalityReducer import DimensionalityReducer
from classes.data.MNISTDataset import MNISTDataset
from functional.utils import default_logger


def main():
    # --- Config ---
    with open('config/training/training_configuration.yaml', 'r') as f:
        generic_config: dict = yaml.safe_load(f)
    with open('config/clustering/clustering_params.yaml', 'r') as f:
        clustering_config: dict = yaml.safe_load(f)
    # --- Logger ---
    logger = default_logger(generic_config["logger"])
    # --- Dataset ---
    train_loader = DataLoader(MNISTDataset(train=True),
                              batch_size=generic_config["batch_size"],
                              shuffle=False,
                              num_workers=generic_config["workers"])
    # -----------------------------------------------------------------------------------
    # TODO: genetic algorithm to maximise these features?
    # -----------------------------------------------------------------------------------
    # --- Keypoint extraction and feature description ---
    key_points_extractor = FeatureExtractingAlgorithm(algorithm="SIFT", logger=logger)
    keypoints, descriptors = key_points_extractor.get_keypoints_and_descriptors(train_loader)
    flat_descriptors = np.concatenate(descriptors)
    # -- Reduction ---
    dimensionality_reducer = DimensionalityReducer(algorithm="UMAP", logger=logger, **clustering_config["umap_args"])
    reduced_vectors = dimensionality_reducer.fit_transform(flat_descriptors)
    # -- HDBSCAN Clustering ---
    clusterer = Clusterer(algorithm="HDBSCAN", logger=logger, **clustering_config["hdbscan_args"])
    labels = clusterer.fit_predict(flat_descriptors)
    clusterer.plot(reduced_vectors, labels)
    # -- HAC Clustering --
    clusterer = Clusterer(algorithm="HAC", logger=logger, **clustering_config["hac_args"])
    labels = clusterer.fit_predict(flat_descriptors)
    clusterer.plot(reduced_vectors, labels)
    # -- KMEANS Clustering --
    clusterer = Clusterer(algorithm="KMEANS", logger=logger, **clustering_config["kmeans_args"])
    labels = clusterer.fit_predict(flat_descriptors)
    clusterer.plot(reduced_vectors, labels)


if __name__ == "__main__":
    main()
