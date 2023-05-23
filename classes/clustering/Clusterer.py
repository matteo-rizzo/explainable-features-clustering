import logging
import time

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from functional.utils import print_minutes, log_on_default

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
            self.__clusterer = HDBSCAN(**kwargs)
        elif algorithm.upper() == "HAC":
            # In cuml, metric is called affinity (in sklearn it has been changed to metric)
            if DEVICE == "GPU":
                kwargs["affinity"] = kwargs.pop("metric")
            self.__clusterer = AgglomerativeClustering(**kwargs)
        elif algorithm.upper() == "KMEANS":
            self.__clusterer = KMeans(**kwargs)
        else:
            raise ValueError("Invalid algorithm, must be in ['HDBSCAN', 'HAC', 'KMEANS']")
        pass

    def fit_predict(self, vectors: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        self.__logger.info(f"Running {self.__algorithm_name} fit_predict...")
        cluster_labels = self.__clusterer.fit_predict(vectors)
        print_minutes(seconds=(time.perf_counter() - t0), input_str=self.__algorithm_name, logger=self.__logger)
        return cluster_labels

    def fit(self, vectors: np.ndarray) -> None:
        t0 = time.perf_counter()
        self.__logger.info(f"Running {self.__algorithm_name} fit...")
        self.__clusterer.fit(vectors)
        print_minutes(seconds=(time.perf_counter() - t0), input_str=self.__algorithm_name, logger=self.__logger)

    def score(self, vectors: np.ndarray) -> float:
        return self.__clusterer.score(vectors)

    def get_estimator(self):
        return self.__clusterer

    @staticmethod
    def plot(vectors, labels, name: str = ""):
        sns.set(style="darkgrid")
        clustered = (labels >= 0)
        plt.figure(figsize=(10, 10), dpi=300)
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

        plt.show()


    @staticmethod
    def plot_3d(vectors, labels, name: str = ""):
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

        plt.show()
        fig.savefig(f'{name}.png', dpi=fig.dpi)