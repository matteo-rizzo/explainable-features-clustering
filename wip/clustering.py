import yaml
from torch.utils.data import DataLoader

from classes.CornerExtractingAlgorithm import CornerExtractingAlgorithm
from classes.clustering.Clusterer import Clusterer
from classes.clustering.DimensionalityReducer import DimensionalityReducer
from classes.data.MNISTDataset import MNISTDataset
from functional.utils import default_logger
import numpy as np

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
    # key_points_extractor = FeatureExtractingAlgorithm(algorithm="SIFT", logger=logger)
    # keypoints, descriptors = key_points_extractor.get_keypoints_and_descriptors(train_loader)
    # flat_descriptors = np.concatenate(descriptors)

    args = {
        "maxCorners": None,  # Maximum number of corners to detect
        "qualityLevel": 0.01,  # Quality level threshold
        "minDistance": 2,  # Minimum distance between detected corners
        "blockSize": 3,  # Size of the neighborhood considered for corner detection
        "useHarrisDetector": False,  # Whether to use the Harris corner detector or not
        "k": 0.04  # Free parameter for the Harris detector
    }

    fea = CornerExtractingAlgorithm(algorithm="SHI-TOMASI", multi_scale=False, logger=logger)
    # FIXME: fishy, look again
    flat_descriptors = np.concatenate(fea.run(train_loader, shape=(3, 3), **args))
    # vectors = fea.corner_to_vector(image, corners, shape=(3, 3))
    # -- Reduction ---
    dimensionality_reducer = DimensionalityReducer(algorithm="UMAP", logger=logger, **clustering_config["umap_args"])
    reduced_vectors = dimensionality_reducer.fit_transform(flat_descriptors)
    # -- HDBSCAN Clustering ---
    clusterer = Clusterer(algorithm="HDBSCAN", logger=logger, **clustering_config["hdbscan_args"])
    labels = clusterer.fit_predict(flat_descriptors)
    clusterer.plot_3d(reduced_vectors, labels, "HDBSCAN")
    # -- HAC Clustering --
    clusterer = Clusterer(algorithm="HAC", logger=logger, **clustering_config["hac_args"])
    labels = clusterer.fit_predict(flat_descriptors)
    clusterer.plot_3d(reduced_vectors, labels, "HAC")
    # -- KMEANS Clustering --
    clusterer = Clusterer(algorithm="KMEANS", logger=logger, **clustering_config["kmeans_args"])
    labels = clusterer.fit_predict(flat_descriptors)
    # clusterer.plot(reduced_vectors, labels, "KMEANS")
    clusterer.plot_3d(reduced_vectors, labels, "KMEANS")


if __name__ == "__main__":
    main()
