import logging
import time
from pathlib import Path

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm

from functional.utilities.utils import print_minutes, log_on_default

try:
    # Nvidia rapids / cuml gpu support
    from cuml.cluster import AgglomerativeClustering, KMeans, HDBSCAN

    DEVICE: str = "GPU"
except ImportError:
    # Standard cpu support
    from sklearn.cluster import AgglomerativeClustering, KMeans
    from hdbscan import HDBSCAN

    DEVICE: str = "CPU"

log_on_default("INFO", f"Importing clustering algorithms with {DEVICE} support.")


class Clusterer:
    def __init__(self, algorithm: str, logger=logging.getLogger(__name__), **kwargs):
        self.__algorithm_name: str = algorithm
        self.__logger = logger
        if algorithm.upper() == "HDBSCAN":
            self.clusterer = HDBSCAN(**kwargs)
        elif algorithm.upper() == "HAC":
            # In cuml, metric is called affinity (in sklearn it has been changed to metric)
            if DEVICE == "GPU":
                kwargs["affinity"] = kwargs.pop("metric")
            self.clusterer = AgglomerativeClustering(**kwargs)
            self.knn = KNeighborsClassifier()
        elif algorithm.upper() == "KMEANS":
            self.clusterer = KMeans(**kwargs)
        # elif algorithm.upper() == "GMM":
        #     self.clusterer = GaussianMixture(**kwargs)
        else:
            raise ValueError("Invalid algorithm, must be in ['HDBSCAN', 'HAC', 'KMEANS']")
        pass

    def fit_predict(self, vectors: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()

        n_clusters: int | str = self.clusterer.n_clusters if self.__algorithm_name.upper() != "HDBSCAN" else "NA"
        self.__logger.info(f"Running {self.__algorithm_name} fit_predict [k = {n_clusters}]...")
        cluster_labels = self.clusterer.fit_predict(vectors)
        print_minutes(seconds=(time.perf_counter() - t0), input_str=self.__algorithm_name, logger=self.__logger)
        if self.__algorithm_name.upper() == "HAC":
            # Fit a knn classifier based on the labels of the clustering
            self.knn.fit(vectors, self.clusterer.labels_)
        return cluster_labels

    def fit(self, vectors: np.ndarray) -> None:
        t0 = time.perf_counter()
        self.__logger.info(f"Running {self.__algorithm_name} fit [k = {self.clusterer.n_clusters}]...")
        self.clusterer.fit(vectors)
        if self.__algorithm_name.upper() == "HAC":
            # Fit a knn classifier based on the labels of the clustering
            self.knn.fit(vectors, self.clusterer.labels_)
        print_minutes(seconds=(time.perf_counter() - t0), input_str=self.__algorithm_name, logger=self.__logger)

    def predict(self, vector: np.ndarray):
        match self.__algorithm_name.upper():
            case "HDBSCAN":
                # TODO: fixme
                return self.clusterer.approximate_predict(self.clusterer, vector)
            case "HAC":
                return self.knn.predict(vector)
            # Case default
            case _:
                return self.clusterer.predict(vector)

    def score(self, vectors: np.ndarray) -> float:
        return self.clusterer.score(vectors)

    def get_estimator(self):
        return self.clusterer

    def get_centroids(self):
        # FIXME: add "if it exists" (density based don't have it)
        return self.clusterer.cluster_centers_

    @staticmethod
    def rank_clusters(data: np.ndarray | list,
                      centroids: np.ndarray | list,
                      labels: list | np.ndarray,
                      print_values: bool = False) -> \
            list[tuple]:
        # TODO: check
        clusters_ranking = []
        # Labels are assumed to be in range [0-num_labels]
        for i in tqdm(range(len(np.unique(labels))), desc="Ranking clusters"):
            cluster_variance = np.var(data[labels == i])
            cluster_distance = np.linalg.norm(data[labels == i] - centroids[i], axis=1)
            cluster_size = np.sum(labels == i)
            # TODO: EVALUATE DIMENSION
            # Seems good!
            importance_score = (cluster_variance / np.mean(cluster_distance)) / (cluster_size / len(data))
            clusters_ranking.append((i, importance_score))

        # print(silhouette_score(data, labels))
        # Calinski-Harabasz index and Davies-Bouldin index evaluate the overall quality of clustering based on
        # different aspects, such as variance ratios and cluster similarities.
        # print(calinski_harabasz_score(data, labels))
        # print(davies_bouldin_score(data, labels))
        # print(silhouette_score(data, labels))

        clusters_ranking = sorted(clusters_ranking, key=lambda x: x[1], reverse=False)
        if print_values:
            for i, ranking in enumerate(clusters_ranking):
                print(f"[{i}] Cluster {ranking[0]}: Importance Score = {ranking[1]}")

        return clusters_ranking

    def n_clusters(self):
        return self.clusterer.n_clusters

    @staticmethod
    def plot(vectors, labels, name: str = "", save: bool = False):
        sns.set(style="darkgrid")
        clustered = (labels >= 0)
        fig = plt.figure(figsize=(10, 10), dpi=300)
        plt.scatter(vectors[~clustered, 0],
                    vectors[~clustered, 1],
                    color=(0.5, 0.5, 0.5),
                    s=0.4,
                    alpha=0.5)

        plt.scatter(vectors[clustered, 0],
                    vectors[clustered, 1],
                    c=labels[clustered],
                    s=0.4,
                    cmap="Spectral")
        plt.title(name)
        # labels, centroids = clustering.get_labels(), clustering.get_centroids()
        #
        # Plot the data points and centroids
        # plt.scatter(flat_descriptors[:, 0], flat_descriptors[:, 1], c=labels)
        # plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=100, linewidths=3, c='k')

        if save:
            Path("plots").mkdir(exist_ok=True)
            fig.savefig(f'plots/{name}_2d.png', dpi=fig.dpi)
        else:
            plt.show()

    @staticmethod
    def plot_3d(vectors, labels, name: str = "", save: bool = False):
        sns.set(style="darkgrid")
        clustered = (labels >= 0)

        fig = plt.figure(figsize=(20, 20), dpi=300)
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(vectors[~clustered, 0],
                   vectors[~clustered, 1],
                   vectors[~clustered, 2],
                   color=(0.5, 0.5, 0.5),
                   s=0.4,
                   alpha=0.5)

        ax.scatter(vectors[clustered, 0],
                   vectors[clustered, 1],
                   vectors[clustered, 2],
                   c=labels[clustered],
                   s=0.4,
                   cmap="Spectral")

        ax.set_title(name)

        if save:
            Path("plots").mkdir(exist_ok=True)
            fig.savefig(f'plots/{name}_3d.png', dpi=fig.dpi)
        else:
            plt.show()
