import logging
import time

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from functional.utils import print_minutes, log_on_default

try:
    # Nvidia rapids / cuml gpu support
    from cuml.cluster import HDBSCAN, AgglomerativeClustering, KMeans  # Also: DBScan

    DEVICE: str = "GPU"
except ImportError:
    # Standard cpu support
    from hdbscan import HDBSCAN
    from sklearn.cluster import AgglomerativeClustering, KMeans

    DEVICE: str = "CPU"

log_on_default("INFO", f"Importing clustering algorithms with {DEVICE} support.")


class Clusterer:
    def __init__(self, algorithm: str, logger=logging.getLogger(__name__), **kwargs):
        self.name: str = algorithm
        self.logger = logger
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
        self.logger.info(f"Running {self.name}...")
        cluster_labels = self.__clusterer.fit_predict(vectors)
        print_minutes(seconds=(time.perf_counter() - t0), input_str=self.name, logger=self.logger)
        return cluster_labels

    def score(self, vectors: np.ndarray) -> float:
        return self.__clusterer.score(vectors)

    @staticmethod
    def plot(vectors, labels):
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

        # labels, centroids = clustering.get_labels(), clustering.get_centroids()
        #
        # Plot the data points and centroids
        # plt.scatter(flat_descriptors[:, 0], flat_descriptors[:, 1], c=labels)
        # plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=100, linewidths=3, c='k')

        plt.show()
