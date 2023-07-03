import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import torch
import yaml

from src.classes.data.OxfordIIITPetDataset import OxfordIIITPetDataset
from src.classes.feature_extraction.FeatureExtractingAlgorithm import FeatureExtractingAlgorithm
from src.functional.utils import default_logger
from src.wip.cluster_extraction import extract_and_cluster


def draw_activation(activation_maps):
    # Create a grid of subplots based on the number of tensors
    num_rows: int = 3
    num_cols: int = 3
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(9, 9))
    im = None
    for i, tensor in enumerate(activation_maps):
        row_idx = i // num_cols
        col_idx = i % num_cols
        ax = axes[row_idx, col_idx]
        im = ax.imshow(tensor, cmap='inferno')
        ax.set_title(f'Heatmap {i + 1}')
        ax.axis('off')
    # Create a big colorbar on the right side
    cax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
    cb = fig.colorbar(im, cax=cax)
    # plt.tight_layout()
    plt.show()


def main():
    # First, we need to have a way to visualize the heatmaps
    # Method 1 - Have few vocab words
    # Method 2 - Have a ton and select top k
    # Method 3 - Have a ton and only visualize a few to be sure

    # --- PSEUDOCODE ---
    # Extract descriptors
    # Run clustering
    # Now you have the clusters (visual vocab)
    # Now you have to create the new dataset, which is made out of heatmaps
    # In order to test this, do this on a few images and a few words
    # Maybe even make a small vocabulary just for testing
    # E.g. vocabulary of 10 and then visualize them all together
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

    clusterer, descriptors, keypoints = extract_and_cluster(clustering_config, key_points_extractor, logger,
                                                            train_loader)
    # Take the train loader
    # Load an image
    for img, label in train_loader:
        plt.imshow(img.squeeze(), cmap='gray')
        plt.show()
        # Find keypoints
        kps, descs = key_points_extractor.get_keypoints_and_descriptors(img)
        # Predict clustering
        cluster_indexes = clusterer.predict(descs)
        # For now, create a tensor of 0 with as many layers as words
        heatmap = torch.zeros((clusterer.n_clusters(), generic_config["img_size"], generic_config["img_size"]))
        # Then, pair coordinates and cluster prediction to assign to layer
        for kp, cluster in zip(kps, cluster_indexes):
            x_coord, y_coord = kp.pt
            # We can start with the dot, then move to the gaussian bit
            heatmap[cluster, int(x_coord), int(y_coord)] += 1.0

        draw_activation(heatmap)

        # Gaussian parameters
        sigma = 10.0  # Controls the spread of the Gaussian
        heatmap = torch.zeros((clusterer.n_clusters(), generic_config["img_size"], generic_config["img_size"]))
        # Then, pair coordinates and cluster prediction to assign to layer
        for kp, cluster in zip(kps, cluster_indexes):
            x_coord, y_coord = kp.pt

            # Generate a grid of coordinates corresponding to the heatmap indices
            x_indices, y_indices = torch.meshgrid(torch.arange(224), torch.arange(224), indexing='ij')

            # Calculate the squared distance from each grid point to the center (x_coord, y_coord)
            squared_dist = (x_indices - x_coord) ** 2 + (y_indices - y_coord) ** 2

            # Calculate the Gaussian distribution
            gaussian = torch.exp(-squared_dist / (2 * sigma ** 2))

            # Add the Gaussian values to the heatmap for the corresponding cluster
            heatmap[cluster] += gaussian

        draw_activation(heatmap)
        # FIXME
        break
    logger.info("Program execution completed.")


if __name__ == "__main__":
    main()
