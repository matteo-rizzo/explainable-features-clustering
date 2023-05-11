import logging
import time

import colorlog
import numpy as np
import seaborn as sns
import yaml
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from classes.FeautureExtractingAlgorithm import FeautureExtractingAlgorithm
from classes.data.MNISTDataset import MNISTDataset
from functional.utils import print_minutes

try:
    # Nvidia rapids / cuml gpu support
    from cuml import UMAP, PCA  # Also: Incremental PCA, Truncated SVD, Random Projections, TSNE
    from cuml.cluster import HDBSCAN, AgglomerativeClustering, KMeans  # Also: DBScan

    print("Importing decomposition and clustering algorithms with GPU support")
except ImportError:
    # Standard cpu support
    from umap import UMAP
    from hdbscan import HDBSCAN
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.decomposition import PCA

    print("Importing decomposition and clustering algorithms with CPU support")

args = {"n_neighbors": 15,
        "min_dist": 0.0,
        "n_components": 2,
        "random_state": 42069,
        "metric": "euclidean"}


class Clusterer:
    def __init__(self):
        pass


class DimensionalityReducer:
    def __init__(self, algorithm: str = "UMAP", logger=logging.getLogger(__name__), **kwargs):
        self.name: str = algorithm
        self.logger: logging.Logger = logger
        if algorithm.upper() == "UMAP":
            self.__reducer = UMAP(**kwargs)
        elif algorithm.upper() == "PCA":
            self.__reducer = PCA(**kwargs)
        self.logger.info(f"Initializing dimensionality reduction with {algorithm.upper()} algorithm.")

    def fit_transform(self, vectors: np.ndarray) -> np.ndarray:
        self.logger.info(f"Starting {self.name} fit + transform...")
        t0 = time.perf_counter()
        reduced_vectors: np.ndarray = self.__reducer.fit_transform(vectors)
        print_minutes(time.perf_counter() - t0, self.name, self.logger)
        return reduced_vectors

    def fit(self, vectors: np.ndarray) -> None:
        t0 = time.perf_counter()
        self.logger.info(f"Starting {self.name} fit...")
        print_minutes(time.perf_counter() - t0, self.name, self.logger)
        self.__reducer.fit(vectors)

    def transform(self, vectors: np.ndarray) -> np.ndarray:
        return self.__reducer.transform(vectors)


def plot(cluster_labels, reduced_vectors):
    # Set the seaborn style to "darkgrid"
    sns.set(style="darkgrid")

    clustered = (cluster_labels >= 0)
    plt.figure(figsize=(10, 10), dpi=200)
    plt.scatter(reduced_vectors[~clustered, 0],
                reduced_vectors[~clustered, 1],
                color=(0.5, 0.5, 0.5),
                s=0.2,
                alpha=0.5)

    plt.scatter(reduced_vectors[clustered, 0],
                reduced_vectors[clustered, 1],
                c=cluster_labels[clustered],
                s=0.2,
                cmap="Spectral")

    # labels, centroids = clustering.get_labels(), clustering.get_centroids()
    #
    # Plot the data points and centroids
    # plt.scatter(flat_descriptors[:, 0], flat_descriptors[:, 1], c=labels)
    # plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=100, linewidths=3, c='k')

    plt.show()


def run_hdbscan(descriptors):
    print("Running HDBSCAN...")
    clusterer = HDBSCAN(min_cluster_size=10,
                        min_samples=3,
                        cluster_selection_method="eom",
                        metric="euclidean")

    return clusterer.fit_predict(descriptors)


def run_hac(descriptors):
    print("Running HAC...")
    clusterer = AgglomerativeClustering(n_clusters=10, affinity='euclidean')
    return clusterer.fit_predict(descriptors)


def main():
    # --- Config ---
    with open('config/training/training_configuration.yaml', 'r') as f:
        config = yaml.safe_load(f)
    # --- Logger ---
    logger = logging.getLogger(__name__)
    logger.setLevel(config["logger"])
    ch = logging.StreamHandler()
    ch.setLevel(config["logger"])
    formatter = colorlog.ColoredFormatter(
        "%(log_color)s[%(asctime)s] - %(levelname)s - %(white)s%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    # --- Dataset ---
    train_loader = DataLoader(MNISTDataset(train=True),
                              batch_size=config["batch_size"],
                              shuffle=False,
                              num_workers=config["workers"])
    # -----------------------------------------------------------------------------------
    # TODO: genetic algorithm to maximise these features?
    # -----------------------------------------------------------------------------------
    # --- Keypoint extraction and feature description ---
    key_points_extractor = FeautureExtractingAlgorithm(algorithm="SIFT")
    keypoints, descriptors = key_points_extractor.get_keypoints_and_descriptors(train_loader)
    # -- Reduction ---
    flat_descriptors = np.concatenate(descriptors)
    dimensionality_reducer = DimensionalityReducer(algorithm="UMAP", **args)
    reduced_vectors = dimensionality_reducer.fit_transform(flat_descriptors)
    # -- Clustering ---
    t0 = time.perf_counter()
    cluster_labels = run_hdbscan(descriptors)
    plot(cluster_labels, reduced_vectors)
    print_minutes(time.perf_counter() - t0, "HDBSCAN")
    # -- Clustering ---
    t0 = time.perf_counter()
    cluster_labels = run_hac(descriptors)
    plot(cluster_labels, reduced_vectors)
    print_minutes(time.perf_counter() - t0, "HAC")

    # clustering.fit(flat_descriptors)
    # labels, centroids = clustering.get_labels(), clustering.get_centroids()

    # Plot the data points and centroids
    # plt.scatter(flat_descriptors[:, 0], flat_descriptors[:, 1], c=labels)
    # plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=100, linewidths=3, c='k')
    # plt.show()


if __name__ == "__main__":
    main()
