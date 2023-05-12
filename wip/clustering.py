import numpy as np
import yaml
from torch.utils.data import DataLoader

from classes.FeatureExtractingAlgorithm import FeatureExtractingAlgorithm
from classes.clustering.Clusterer import Clusterer
from classes.clustering.DimensionalityReducer import DimensionalityReducer
from classes.data.MNISTDataset import MNISTDataset
from functional.utils import default_logger, join_dicts


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
    args = join_dicts(clustering_config["umap_args"])
    dimensionality_reducer = DimensionalityReducer(algorithm="UMAP", logger=logger, **args)
    reduced_vectors = dimensionality_reducer.fit_transform(flat_descriptors)
    # -- Clustering ---
    args = join_dicts(clustering_config["hdbscan_args"])
    clusterer = Clusterer(algorithm="HDBSCAN", logger=logger, **args)
    labels = clusterer.fit_predict(reduced_vectors)
    clusterer.plot(reduced_vectors, labels)
    # -- Clustering --
    args = join_dicts(clustering_config["hac_args"])
    clusterer = Clusterer(algorithm="HAC", logger=logger, **args)
    labels = clusterer.fit_predict(reduced_vectors)
    clusterer.plot(reduced_vectors, labels)

    # clustering.fit(flat_descriptors)
    # labels, centroids = clustering.get_labels(), clustering.get_centroids()

    # Plot the data points and centroids
    # plt.scatter(flat_descriptors[:, 0], flat_descriptors[:, 1], c=labels)
    # plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=100, linewidths=3, c='k')
    # plt.show()


if __name__ == "__main__":
    main()
