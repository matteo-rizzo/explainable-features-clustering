import numpy as np
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from classes.FeatureExtractingAlgorithm import FeatureExtractingAlgorithm
from classes.clustering.Clusterer import Clusterer
from classes.clustering.DimensionalityReducer import DimensionalityReducer
from classes.data.Food101Dataset import Food101Dataset
from functional.utils import default_logger


def main():
    # --- Config ---
    with open('config/training/training_configuration.yaml', 'r') as f:
        generic_config: dict = yaml.safe_load(f)
    with open('config/clustering/clustering_params.yaml', 'r') as f:
        clustering_config: dict = yaml.safe_load(f)
    # with open('config/feature_extraction.yaml', 'r') as f:
    #     feature_extraction_config: dict = yaml.safe_load(f)
    # --- Logger ---
    logger = default_logger(generic_config["logger"])
    # --- Dataset ---
    train_loader = DataLoader(Food101Dataset(train=False),
                              batch_size=generic_config["batch_size"],
                              shuffle=False,
                              num_workers=generic_config["workers"])

    # dataset = Food101Dataset(train=False)
    # indices = np.arange(len(dataset))
    # train_indices, test_indices = train_test_split(indices, train_size=100 * 10)
    #
    # # Warp into Subsets and DataLoaders
    # train_dataset = Subset(dataset, train_indices)
    #
    # train_loader = DataLoader(train_dataset,
    #                           batch_size=generic_config["batch_size"],
    #                           shuffle=False,
    #                           num_workers=generic_config["workers"]
    #                           )
    # -----------------------------------------------------------------------------------
    # TODO: genetic algorithm to maximise these features?
    # -----------------------------------------------------------------------------------
    # --- Keypoint extraction and feature description ---
    key_points_extractor = FeatureExtractingAlgorithm(algorithm="SIFT", nfeatures=400, logger=logger)
    keypoints, descriptors = key_points_extractor.get_keypoints_and_descriptors(train_loader, rgb=True)
    flat_descriptors = np.concatenate(descriptors)
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
