import itertools
import logging
from pathlib import Path
from typing import TypeVar, Iterable, Callable, Generic

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from torch.utils.data import DataLoader

from classes.FeatureExtractingAlgorithm import FeatureExtractingAlgorithm
from classes.clustering.Clusterer import Clusterer
from classes.clustering.DimensionalityReducer import DimensionalityReducer
from classes.data.MNISTDataset import MNISTDataset
from functional.utils import default_logger

try:
    # Nvidia rapids / cuml gpu support
    from cuml.cluster import silhouette_score
except ImportError:
    # Standard cpu support
    from sklearn.metrics import silhouette_score



T = TypeVar("T")


def grid_search(estimator_class: T,
                grid_params: dict,
                metric_fun: Callable[[Generic[T]], float],
                large_is_better: bool,
                estimator_fit_args: Iterable = None,
                estimator_fit_kwargs: dict = None,
                estimator_kwargs: dict = None) -> list[dict]:
    print("Started grid search...")
    # Stable clusters
    estimator_fit_args = estimator_fit_args if estimator_fit_args is not None else list()
    estimator_fit_kwargs = estimator_fit_kwargs if estimator_fit_kwargs is not None else dict()
    estimator_kwargs = estimator_kwargs if estimator_kwargs is not None else dict()

    best_score = -1.0
    best_parameters = None
    all_results = list()

    compare_fn = float.__gt__ if large_is_better else float.__lt__

    k_args, args_list = zip(*grid_params.items())
    args_len = [list(range(len(a))) for a in args_list]
    args_index_combinations: list[tuple] = itertools.product(*args_len)

    for comb_idx in args_index_combinations:
        # Prepare argument combination
        vals = [arg[i] for arg, i in zip(args_list, comb_idx)]
        kv_args = dict(zip(k_args, vals))

        # Clustering
        est = estimator_class(**kv_args, **estimator_kwargs).fit(*estimator_fit_args, **estimator_fit_kwargs)
        # DBCV score
        score = metric_fun(est)
        n_clusters = int(est.labels_.max() + 1)
        ext_args = {**kv_args, "score": score, "n_clusters": n_clusters}
        # if we got a better score, store it and the parameters
        if compare_fn(score, best_score):
            best_score = score
            best_parameters = ext_args
        all_results.append(ext_args)

    print("Best score: {:.4f}".format(best_score))
    print("Best parameters: {}".format(best_parameters))
    return all_results


# Evaluate Multiple Metrics:

#     Instead of relying solely on the distortion and silhouette score, consider evaluating multiple clustering evaluation metrics. Different metrics capture different aspects of cluster quality, such as compactness, separation, or stability.
#     Some commonly used metrics include the Calinski-Harabasz index, Davies-Bouldin index, silhouette coefficient, or even domain-specific metrics if applicable.
#     By considering multiple metrics, you can gain a more comprehensive understanding of the clustering performance and make a more informed decision about the optimal number of clusters.

# Incorporate Stability Analysis:
#     Cluster stability analysis helps assess the robustness and reliability of the clustering results.
#     One approach is to perform multiple runs of the clustering algorithm with random initializations and calculate the stability of the resulting clusters.
#     Stability measures, such as the Jaccard similarity or Variation of Information, can quantify the consistency of the cluster assignments across different runs.
#     By considering stability, you can identify more stable and reliable clusters, reducing the potential impact of initialization randomness.

# Utilize Internal Validation Indices:
#     Internal validation indices provide quantitative measures to assess the quality of clustering without relying on external reference data.
#     Indices like the Dunn index or the Xie-Beni index evaluate cluster compactness and separation.
#     By incorporating internal validation indices, you can gain additional insights into the cluster structure and better estimate the optimal number of clusters.

# Explore Alternative Algorithms:
#     Instead of relying solely on K-means and HDBSCAN, consider exploring other clustering algorithms that are suitable for your data and problem domain.
#     Algorithms like hierarchical clustering, DBSCAN, spectral clustering, or Gaussian mixture models may provide different perspectives on the optimal number of clusters.
#     Evaluating multiple algorithms can help validate the results and provide a more robust estimation of the optimal number of clusters.

# Cross-Validation or Resampling:
#     Cross-validation or resampling techniques can be employed to assess the stability and generalization performance of the clustering results.
#     Splitting the data into multiple subsets or performing bootstrap resampling can help evaluate the consistency of the clustering across different data subsets.
#     This can provide insights into the stability of the estimated optimal number of clusters and help assess the generalization performance on unseen data.

# Consider Domain Knowledge:
#     Incorporate domain knowledge or expert insights into the evaluation process.
#     If you have prior knowledge about the data and the expected number of clusters based on the problem domain, it can guide the selection of the optimal number of clusters.
#     Domain-specific constraints or requirements may also influence the clustering evaluation and the determination of the optimal number of clusters.

def find_optimal_n_clusters(clustering_algorithm: str,
                            descriptors: np.ndarray,
                            reduced_descriptors: np.ndarray,
                            logger: logging.Logger,
                            **kwargs) -> None:
    # --- Config ---
    plot_path: Path = Path("dumps/plots")
    with open("config/clustering/clustering_grid_search.yaml", encoding="UTF-8") as f:
        conf_search = yaml.load(f, Loader=yaml.FullLoader)["clustering"]
    config = conf_search[clustering_algorithm.lower()]
    # --- Optional normalization ---
    if kwargs.get("normalize", False):
        descriptors /= np.linalg.norm(descriptors, axis=1).reshape(-1, 1)

    # --- KMEANS ---
    if clustering_algorithm.upper() == "KMEANS":
        distortions, silhouette_scores = [], []
        k_range = range(config["k_start"], config["k_end"])
        for k in k_range:
            clustering = Clusterer(algorithm="KMEANS", logger=logger, n_clusters=k, random_state=0)
            # clustering = cuml.cluster.KMeans(n_clusters=k, n_init="auto", random_state=0)
            labels = clustering.fit_predict(reduced_descriptors)
            # Should be same as inertia_
            distortion = clustering.score(reduced_descriptors)
            distortions.append(distortion)  # lower the better
            # Higher the better (ideal > .5)
            sh_score = silhouette_score(reduced_descriptors, labels)
            silhouette_scores.append(silhouette_score(reduced_descriptors, labels))
            logger.info(f"[k = {k}] Distortion: {distortion}, Silouhette score: {sh_score}")

        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(k_range, distortions, "bx-")
        ax[0].set_xlabel("Number of Clusters")
        ax[0].set_ylabel("Distortion (the lower the better)")
        ax[0].set_title("Elbow Method")
        ax[0].grid(True)

        ax[1].plot(k_range, silhouette_scores, "bx-")
        ax[1].set_xlabel("Number of Clusters")
        ax[1].set_ylabel("Silhouette Score (the higher the better)")
        ax[1].set_title("Silhouette Method")
        ax[1].grid(True)

        plt.show()

        optimal_n_clusters = np.argmax(silhouette_scores) + config["k_start"]
    # --- HDBSCAN ---
    elif clustering_algorithm.upper() == "HDBSCAN":
        # Define the score function
        raise NotImplementedError
        def fun_dbcv(est: hdbscan.HDBSCAN) -> float:
            return float(est.relative_validity_)

        results = grid_search(hdbscan.HDBSCAN, grid_params=config,
                              metric_fun=fun_dbcv,
                              estimator_fit_args=(reduced_descriptors,),
                              large_is_better=True,
                              estimator_kwargs=dict(gen_min_span_tree=True))

        dfr = pd.DataFrame.from_records(results).sort_values(by="score", ascending=False)

        plot_path.mkdir(exist_ok=True, parents=True)
        dfr.to_csv(plot_path / "hdbscan_grid_results.csv")

        optimal_n_clusters = dfr["n_clusters"][0]
    # --- HAC ---
    elif clustering_algorithm.upper() == "HAC":
        optimal_n_clusters = 0

    print(f"The optimal number of clusters is {optimal_n_clusters}")


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
    # --- Keypoint extraction and feature description ---
    key_points_extractor = FeatureExtractingAlgorithm(algorithm="SIFT", logger=logger)
    keypoints, descriptors = key_points_extractor.get_keypoints_and_descriptors(train_loader)
    flat_descriptors = np.concatenate(descriptors)
    # -- Reduction ---
    dimensionality_reducer = DimensionalityReducer(algorithm="UMAP", logger=logger, **clustering_config["umap_args"])
    reduced_vectors = dimensionality_reducer.fit_transform(flat_descriptors)

    find_optimal_n_clusters("KMEANS", flat_descriptors, reduced_vectors, logger)


if __name__ == "__main__":
    main()
