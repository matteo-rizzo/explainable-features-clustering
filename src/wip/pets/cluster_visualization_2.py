import random
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as F
import yaml
from matplotlib import patches
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.classes.data.OxfordIIITPetDataset import OxfordIIITPetDataset
from src.classes.feature_extraction.FeatureExtractingAlgorithm import FeatureExtractingAlgorithm
from src.functional.utils import default_logger
from src.wip.cluster_extraction import extract_and_cluster


def extract_patch(img, keypoint, patch_size: int = 24, larger_patch: int = 32):  # , min_patch_size: int = 16):
    x, y = map(int, keypoint.pt)  # Convert coordinates to integers
    # keypoint_size = keypoint.size

    # Adjust the patch size based on the keypoint size
    # patch_size = int(max(keypoint_size * 2, min_patch_size))

    top = max(y - larger_patch // 2, 0)
    bottom = min(y + larger_patch // 2, img.shape[0])
    left = max(x - larger_patch // 2, 0)
    right = min(x + larger_patch // 2, img.shape[1])

    patch = img[top:bottom, left:right]
    # TODO: adjust patche based on rotation (make optional?)
    # positive rotation is correct
    rotated_patch = F.rotate(patch.unsqueeze(0), keypoint.angle, interpolation=F.InterpolationMode.BILINEAR).squeeze()
    smaller_patch = F.center_crop(rotated_patch, [patch_size, patch_size])
    return smaller_patch


def fill_cluster_patches(img,
                         img_keypoints,
                         img_descriptors_clusters,
                         cluster_dict):
    for keypoint, cluster_idx in zip(img_keypoints, img_descriptors_clusters):
        patch = extract_patch(img, keypoint)
        cluster_dict[cluster_idx].append(patch)


def plot_patches(patches_list, cluster_idx):
    num_patches = 64
    grid_size = 8

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(16, 16))
    fig.suptitle(f"Cluster: {str(cluster_idx)}", fontsize=16)  # Add title for the entire plot
    fig.tight_layout()
    # shuffles in place
    random.shuffle(patches_list)
    for i in range(min(num_patches, len(patches_list))):
        patch = patches_list[i]
        ax = axes[i // grid_size, i % grid_size]
        ax.imshow(patch, cmap='gray')
        ax.axis('off')

        # Add red square around the center of the patch
        rect = patches.Rectangle((4, 4), 16, 16, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)

    # Save the plot as an image file
    Path("plots").mkdir(exist_ok=True)
    plt.savefig(f'plots/patch_plot_{cluster_idx}.png')
    plt.close()


def plot_cluster_sift_patches():
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

    # --- PLOTTING ---
    cluster_patches = defaultdict(list)
    for idx, (img, label) in tqdm(enumerate(train_loader), desc="Extracting patches...", total=len(train_loader)):
        img = img.squeeze()
        img_descriptors = descriptors[idx]
        img_keypoints = keypoints[idx]
        img_descriptors_clusters = clusterer.predict(img_descriptors)
        fill_cluster_patches(img,
                             img_keypoints,
                             img_descriptors_clusters,
                             cluster_patches)

    for i in tqdm(range(clustering_config["kmeans_args"]["n_clusters"]), desc="Saving plots"):
        plot_patches(cluster_patches[i], i)


if __name__ == "__main__":
    plot_cluster_sift_patches()
