import time

import hdbscan
import numpy as np
import umap
import yaml
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from torch.utils.data import DataLoader
from tqdm import tqdm
import seaborn as sns

from classes.FeautureExtractingAlgorithm import FeautureExtractingAlgorithm
from classes.data.MNISTDataset import MNISTDataset


def reducer(descriptors):
    _reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.0,
        n_components=2,
        random_state=42069,
        metric="cosine"
    )
    # print("Running umap...")
    # reduced_vectors = _reducer.fit_transform(latent_vectors)
    t0 = time.perf_counter()
    with tqdm(total=1, desc="Running umap...", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') as pbar:
        # Fit UMAP and update progress bar
        _reducer.fit(descriptors)
        pbar.update(1)
    print(f"{time.perf_counter()-t0:.2f}s")
    reduced_vectors = _reducer.transform(descriptors)
    # -- Clustering ---
    cluster_labels = run_hdbscan(descriptors)
    plot(cluster_labels, reduced_vectors)

    cluster_labels = run_hac(descriptors)
    plot(cluster_labels, reduced_vectors)


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
    # -- Clustering ---
    clusterer = hdbscan.HDBSCAN(min_cluster_size=10,
                                min_samples=3,
                                cluster_selection_method="eom",
                                metric="euclidean")

    return clusterer.fit_predict(descriptors)


def run_hac(descriptors):
    print("Running HAC...")

    # Create and fit Agglomerative Clustering
    clusterer = AgglomerativeClustering(n_clusters=10,
                                        metric='euclidean',
                                        # linkage="average"
                                        # linkage='ward',
                                        # distance_threshold=2.0
                                        )
    return clusterer.fit_predict(descriptors)


def main():
    with open('../config/training/training_configuration.yaml', 'r') as f:
        config = yaml.safe_load(f)

    train_loader = DataLoader(MNISTDataset(train=True),
                              batch_size=config["batch_size"],
                              shuffle=False,
                              num_workers=config["workers"])
    test_loader = DataLoader(MNISTDataset(train=False),
                             batch_size=config["batch_size"],
                             shuffle=False,
                             num_workers=config["workers"])
    # -----------------------------------------------------------------------------------
    # TODO: genetic algorithm to maximise these features?
    # -----------------------------------------------------------------------------------
    key_points_extractor = FeautureExtractingAlgorithm(algorithm="SIFT")
    keypoints, descriptors = key_points_extractor.get_keypoints_and_descriptors(train_loader)

    # TODO HDBScan
    # clustering = KMeansClustering(n_clusters=10)
    # print(descriptors)
    flat_descriptors = np.concatenate(descriptors)
    print(flat_descriptors.shape)
    reducer(flat_descriptors[234243:, :])

    # clustering.fit(flat_descriptors)
    # labels, centroids = clustering.get_labels(), clustering.get_centroids()

    # Plot the data points and centroids
    # plt.scatter(flat_descriptors[:, 0], flat_descriptors[:, 1], c=labels)
    # plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=100, linewidths=3, c='k')
    # plt.show()


if __name__ == "__main__":
    main()
    # show_4()
