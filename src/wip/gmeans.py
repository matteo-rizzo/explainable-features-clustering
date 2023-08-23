import numpy as np
import torch
import yaml
from pyclustering.cluster import cluster_visualizer
from pyclustering.cluster.gmeans import gmeans
from pyclustering.samples.definitions import FCPS_SAMPLES
from pyclustering.utils import read_sample
from torch.utils.data import DataLoader

from functional.utilities.utils import default_logger
from src.classes.data.OxfordIIITPetDataset import OxfordIIITPetDataset
from src.classes.feature_extraction.FeatureExtractingAlgorithm import FeatureExtractingAlgorithm
from functional.utilities.cluster_extraction import extract_and_cluster


def example():
    # Read sample 'Lsun' from file.
    sample = read_sample(FCPS_SAMPLES.SAMPLE_LSUN)
    # Create instance of G-Means algorithm. By default algorithm start search from single cluster.
    gmeans_instance = gmeans(sample, repeat=10).process()
    # Extract clustering results: clusters and their centers
    clusters = gmeans_instance.get_clusters()
    centers = gmeans_instance.get_centers()
    print(len(clusters))
    # Print total sum of metric errors
    print("Total WCE:", gmeans_instance.get_total_wce())
    # Visualize clustering results
    visualizer = cluster_visualizer()
    visualizer.append_clusters(clusters, sample)
    visualizer.show()


def main():
    # --- Config ---
    with open('config/training/training_configuration.yaml', 'r') as f:
        generic_config: dict = yaml.safe_load(f)
    with open('config/clustering/clustering_params.yaml', 'r') as f:
        clustering_config: dict = yaml.safe_load(f)
    # --- Logger ---
    logger = default_logger(generic_config["logger"])
    # --- Dataset ---
    train_loader = torch.utils.data.DataLoader(OxfordIIITPetDataset(train=True, augment=False),
                                               batch_size=1,
                                               shuffle=False,
                                               num_workers=generic_config["workers"],
                                               drop_last=False)
    # -----------------------------------------------------------------------------------
    # --- Keypoint extraction and feature description ---
    key_points_extractor = FeatureExtractingAlgorithm(algorithm="SIFT", logger=logger)

    # TODO: skip superfluous clustering
    _, descriptors, keypoints = extract_and_cluster(clustering_config, key_points_extractor, logger,
                                                    train_loader)
    flat_descriptors = np.concatenate(descriptors)
    logger.info("Running gmeans...")
    gmeans_instance = gmeans(flat_descriptors, repeat=1).process()
    clusters = gmeans_instance.get_clusters()
    logger.info(f"{len(clusters)} clusters.")
    centers = gmeans_instance.get_centers()
    # Print total sum of metric errors
    logger.info(f"Total WCE:{gmeans_instance.get_total_wce()}")
    # Visualize clustering results
    # visualizer = cluster_visualizer()
    # visualizer.append_clusters(clusters, sample)
    # visualizer.show()


if __name__ == "__main__":
    main()
    # example()