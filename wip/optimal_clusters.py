import itertools
from pathlib import Path
from typing import TypeVar, Iterable, Callable, Generic

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from hdbscan import HDBSCAN
from sklearn.cluster import KMeans
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


def find_optimal_n_clusters(clustering_algorithm,
                            descriptors,
                            reduced_descriptors,
                            conf_search: str | Path, **kwargs) -> None:
    _plot_path: Path = Path("dumps/plots")

    with open(conf_search, encoding="UTF-8") as f:
        conf_search = yaml.load(f, Loader=yaml.FullLoader)["clustering"]

    # documents = [d.body for d in documents]

    # descriptors = _embedding_model.encode(documents, show_progress_bar=False)

    if kwargs.get("normalize", False):
        descriptors /= np.linalg.norm(descriptors, axis=1).reshape(-1, 1)

    # umap_embeddings = _topic_model._reduce_dimensionality(descriptors)

    # 2. Select best hyperparameters

    if isinstance(clustering_algorithm, KMeans):
        c = conf_search["kmeans"]
        distortions, silhouette_scores = [], []
        k_range = range(c["k_start"], c["k_end"])
        for k in k_range:
            clustering = KMeans(n_clusters=k, n_init="auto", random_state=0)
            clustering.fit(reduced_descriptors)
            distortions.append(clustering.inertia_)  # lower the better
            silhouette_scores.append(
                silhouette_score(reduced_descriptors, clustering.labels_))  # higher the better (ideal > .5)

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

        optimal_n_clusters = np.argmax(silhouette_scores) + c["k_start"]
    else:
        # logging.captureWarnings(True)
        # parameters and distributions to sample from
        c = conf_search["hdbscan"]

        # Define the score function
        def fun_dbcv(est: HDBSCAN) -> float:
            return float(est.relative_validity_)

        results = grid_search(HDBSCAN, grid_params=c,
                              metric_fun=fun_dbcv,
                              estimator_fit_args=(reduced_descriptors,),
                              large_is_better=True,
                              estimator_kwargs=dict(gen_min_span_tree=True))

        dfr = pd.DataFrame.from_records(results).sort_values(by="score", ascending=False)

        _plot_path.mkdir(exist_ok=True, parents=True)
        dfr.to_csv(_plot_path / "hdbscan_grid_results.csv")

        optimal_n_clusters = dfr["n_clusters"][0]

    print(f"The optimal number of clusters is {optimal_n_clusters}")
