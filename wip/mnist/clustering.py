import yaml
from torch.utils.data import DataLoader

from classes.CornerExtractingAlgorithm import CornerExtractingAlgorithm
from classes.FeatureExtractingAlgorithm import FeatureExtractingAlgorithm
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
    with open('config/feature_extraction.yaml', 'r') as f:
        feature_extraction_config: dict = yaml.safe_load(f)
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
    # fea = CornerExtractingAlgorithm(algorithm="SHI-TOMASI", multi_scale=False, logger=logger)
    with open('config/feature_extraction.yaml', 'r') as f:
        feature_extraction_config: dict = yaml.safe_load(f)
    fea = FeatureExtractingAlgorithm(algorithm="SHI-TOMASI_BOX", multi_scale=False,
                                     logger=logger, **feature_extraction_config)
    keypoints, descriptors = fea.get_keypoints_and_descriptors(train_loader)
    # gather all descriptors in a single big array
    flat_descriptors = np.concatenate(descriptors)
    # vectors = fea.corner_to_vector(image, corners, shape=(3, 3))
    # -- Reduction ---
    dimensionality_reducer = DimensionalityReducer(algorithm="UMAP", logger=logger, **clustering_config["umap_args_2d"])
    vectors_2d = dimensionality_reducer.fit_transform(flat_descriptors)
    dimensionality_reducer = DimensionalityReducer(algorithm="UMAP", logger=logger, **clustering_config["umap_args_3d"])
    vectors_3d = dimensionality_reducer.fit_transform(flat_descriptors)
    # -- HDBSCAN Clustering ---
    clusterer = Clusterer(algorithm="HDBSCAN", logger=logger, **clustering_config["hdbscan_args"])
    labels = clusterer.fit_predict(flat_descriptors)
    clusterer.plot(vectors_2d, labels, "HDBSCAN")
    clusterer.plot_3d(vectors_3d, labels, "HDBSCAN")
    # -- HAC Clustering --
    clusterer = Clusterer(algorithm="HAC", logger=logger, **clustering_config["hac_args"])
    labels = clusterer.fit_predict(flat_descriptors)
    clusterer.plot(vectors_2d, labels, "HAC")
    clusterer.plot_3d(vectors_3d, labels, "HAC")
    # -- KMEANS Clustering --
    clusterer = Clusterer(algorithm="KMEANS", logger=logger, **clustering_config["kmeans_args"])
    labels = clusterer.fit_predict(flat_descriptors)
    clusterer.plot(vectors_2d, labels, "KMEANS")
    clusterer.plot_3d(vectors_3d, labels, "KMEANS")


if __name__ == "__main__":
    main()
