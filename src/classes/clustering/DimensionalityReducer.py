import logging
import time

import numpy as np

from src.functional.utils import print_minutes, log_on_default

try:
    # Nvidia rapids / cuml gpu support
    from cuml import UMAP, PCA  # Also: Incremental PCA, Truncated SVD, Random Projections, TSNE

    DEVICE: str = "GPU"
except ImportError:
    # Standard cpu support
    from umap import UMAP
    from sklearn.decomposition import PCA

    DEVICE: str = "CPU"

log_on_default("INFO", f"Importing decomposition algorithms with {DEVICE} support.")


class DimensionalityReducer:
    def __init__(self, algorithm: str = "UMAP", logger=logging.getLogger(__name__), **kwargs):
        self.__algorithm_name: str = algorithm
        self.__logger: logging.Logger = logger
        if algorithm.upper() == "UMAP":
            self.__reducer = UMAP(**kwargs)
        elif algorithm.upper() == "PCA":
            self.__reducer = PCA(**kwargs)
        else:
            raise ValueError("Invalid algorithm, must be in ['UMAP', 'PCA']")
        self.__logger.info(f"Initializing dimensionality reduction with {algorithm.upper()} algorithm.")

    def fit_transform(self, vectors: np.ndarray) -> np.ndarray:
        self.__logger.info(f"Starting {self.__algorithm_name} fit + transform...")
        t0 = time.perf_counter()
        reduced_vectors: np.ndarray = self.__reducer.fit_transform(vectors)
        print_minutes(time.perf_counter() - t0, self.__algorithm_name, self.__logger)
        return reduced_vectors

    def fit(self, vectors: np.ndarray) -> None:
        t0 = time.perf_counter()
        self.__logger.info(f"Starting {self.__algorithm_name} fit...")
        print_minutes(time.perf_counter() - t0, self.__algorithm_name, self.__logger)
        self.__reducer.fit(vectors)

    def transform(self, vectors: np.ndarray) -> np.ndarray:
        return self.__reducer.transform(vectors)
