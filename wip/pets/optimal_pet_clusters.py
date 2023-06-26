import itertools
import logging
from pathlib import Path
from typing import TypeVar, Iterable, Callable, Generic

import matplotlib.pyplot as plt
import numpy as np
import yaml
from torch.utils.data import DataLoader

from classes.clustering.Clusterer import Clusterer
from classes.data.OxfordIIITPetDataset import OxfordIIITPetDataset
from classes.feature_extraction.FeatureExtractingAlgorithm import FeatureExtractingAlgorithm
from functional.utils import default_logger, colorstr

try:
    # Nvidia rapids / cuml gpu support
    from cuml.cluster import silhouette_score
    N_INIT = 1
except ImportError:
    # Standard cpu support
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    N_INIT = 'auto'

T = TypeVar("T")


def grid_search(estimator_class: T,
                grid_params: dict,
                metric_fun: Callable[[Generic[T]], float],
                large_is_better: bool,
                estimator_fit_args: Iterable = None,
                estimator_fit_kwargs: dict = None,
                estimator_kwargs: dict = None,
                logger=logging.getLogger(__name__)) -> list[dict]:
    logger.info(f"Started {colorstr('bold', 'magenta', 'grid search')}{colorstr('white', '...')}")
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
    # TODO: REMOVE
    length = len(list(args_index_combinations))
    args_index_combinations: list[tuple] = itertools.product(*args_len)
    for idx, comb_idx in enumerate(args_index_combinations):
        # Prepare argument combination
        vals = [arg[i] for arg, i in zip(args_list, comb_idx)]
        kv_args = dict(zip(k_args, vals))

        # Clustering
        est = estimator_class(algorithm="HDBSCAN", logger=logger, **kv_args, **estimator_kwargs)
        est.fit(*estimator_fit_args, **estimator_fit_kwargs)
        est = est.get_estimator()
        # DBCV score
        score = metric_fun(est)
        n_clusters = int(est.labels_.max() + 1)
        ext_args = {**kv_args, "score": score, "n_clusters": n_clusters}
        logger.info(f"[{idx}/{length}] - score: {score:.3f} (best: {best_score:.3f})")
        # if we got a better score, store it and the parameters
        if compare_fn(score, best_score):
            best_score = score
            best_parameters = ext_args
        all_results.append(ext_args)

    print("Best score: {:.4f}".format(best_score))
    print("Best parameters: {}".format(best_parameters))
    return all_results


def find_optimal_n_clusters(clustering_algorithm: str,
                            descriptors: np.ndarray,
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

    optimal_n_clusters = -1
    # --- KMEANS ---
    if clustering_algorithm.upper() == "KMEANS":
        distortions, ch_scores, da_scores = [], [], []
        k_range = range(config["k_start"], config["k_end"], config["k_step"])
        for k in k_range:
            clustering = Clusterer(algorithm="KMEANS",
                                   logger=logger, n_clusters=k,
                                   random_state=0,
                                   n_init=N_INIT)
            # clustering = cuml.cluster.KMeans(n_clusters=k, n_init="auto", random_state=0)
            labels = clustering.fit_predict(descriptors)
            # Distortion is the average of the euclidean squared distance
            # from the centroid of the respective clusters.
            # Inertia is the sum of squared distances of samples to their closest cluster centre.
            distortion = clustering.score(descriptors)
            distortions.append(distortion)  # lower the better
            # Higher the better (ideal > .5)
            ch_score = calinski_harabasz_score(descriptors, labels)
            da_score = davies_bouldin_score(descriptors, labels)
            ch_scores.append(ch_score)
            da_scores.append(da_score)
            logger.info(f"[k = {k}] Distortion: {distortion:.2f}, "
                        f"Calinski harabasz: {ch_score:.2f}, "
                        f"Davies bouldin: {da_score:.2f}")  # , Silouhette score: {sh_score}")

        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        ax[0].plot(k_range, distortions, "bx-")
        ax[0].set_xlabel("Number of Clusters")
        ax[0].set_ylabel("Distortion (the lower the better)")
        ax[0].set_title("Elbow Method")
        ax[0].grid(True)

        ax[1].plot(k_range, ch_scores, "bx-")
        ax[1].set_xlabel("Number of Clusters")
        ax[1].set_ylabel("calinski_harabasz_score (higher better)")
        ax[1].set_title("calinski harabasz Method")
        ax[1].grid(True)

        ax[2].plot(k_range, da_scores, "bx-")
        ax[2].set_xlabel("Number of Clusters")
        ax[2].set_ylabel("davies_bouldin_score (lower better)")
        ax[2].set_title("davies bouldin Method")
        ax[2].grid(True)

        plt.show()

        print(f"The optimal number of clusters (distortion) is {np.argmin(distortions)+ config['k_start']}")
        print(f"The optimal number of clusters (calinski_harabasz_score) is {np.argmax(ch_scores)+ config['k_start']}")
        print(f"The optimal number of clusters (davies_bouldin_score) is {np.argmin(da_scores)+ config['k_start']}")
    # --- HDBSCAN ---
    # elif clustering_algorithm.upper() == "HDBSCAN":
    #     # Define the score function
    #     # raise NotImplementedError
    #     def fun_dbcv(est: hdbscan.HDBSCAN) -> float:
    #         return float(est.relative_validity_)
    #
    #     results = grid_search(Clusterer, grid_params=config,
    #                           metric_fun=fun_dbcv,
    #                           estimator_fit_args=(descriptors,),
    #                           large_is_better=True,
    #                           estimator_kwargs=dict(gen_min_span_tree=True),
    #                           logger=logger)
    #
    #     dfr = pd.DataFrame.from_records(results).sort_values(by="score", ascending=False)
    #
    #     plot_path.mkdir(exist_ok=True, parents=True)
    #     dfr.to_csv(plot_path / "hdbscan_grid_results.csv")
    #
    #     optimal_n_clusters = dfr["n_clusters"][0]
    # # --- HAC ---
    # elif clustering_algorithm.upper() == "HAC":
    #     optimal_n_clusters = 0

    # print(f"The optimal number of clusters is {optimal_n_clusters}")


def main():
    # --- Config ---
    with open('config/training/training_configuration.yaml', 'r') as f:
        generic_config: dict = yaml.safe_load(f)
    with open('config/shi_thomasi_feature_extraction.yaml', 'r') as f:
        feature_extraction_config: dict = yaml.safe_load(f)
    # --- Logger ---
    logger = default_logger(generic_config["logger"])
    # --- Dataset ---
    train_loader = DataLoader(OxfordIIITPetDataset(train=True),
                              batch_size=generic_config["batch_size"],
                              shuffle=False,
                              num_workers=generic_config["workers"],
                              drop_last=False)
    # -----------------------------------------------------------------------------------
    # --- Keypoint extraction and feature description ---
    key_points_extractor = FeatureExtractingAlgorithm(algorithm="SIFT", logger=logger)
    # key_points_extractor = FeatureExtractingAlgorithm(algorithm="SHI-TOMASI_BOX", multi_scale=False,
    #                                  logger=logger, **feature_extraction_config)
    keypoints, descriptors = key_points_extractor.get_keypoints_and_descriptors(train_loader)
    flat_descriptors = np.concatenate(descriptors)

    find_optimal_n_clusters("KMEANS", flat_descriptors, logger)


if __name__ == "__main__":
    main()
