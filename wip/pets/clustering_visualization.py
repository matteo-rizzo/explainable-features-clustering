from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib import gridspec
from torch.utils.data import DataLoader
from tqdm import tqdm

from classes.clustering.Clusterer import Clusterer
from classes.data.OxfordIIITPetDataset import OxfordIIITPetDataset
from classes.feature_extraction.FeatureExtractingAlgorithm import FeatureExtractingAlgorithm
from functional.utils import default_logger
import matplotlib.patches as patches

def plot_keypoint_patches(image, keypoints):
    num_patches = 36
    patch_size = 16
    grid_size = 6

    image = image.squeeze()

    fig = plt.figure(figsize=(10, 8), facecolor='gray')
    gs = gridspec.GridSpec(nrows=grid_size, ncols=grid_size + 1)

    # Plot original image on the leftmost column
    ax0 = fig.add_subplot(gs[:, 0])
    ax0.imshow(image, cmap='gray')
    ax0.axis('off')

    for i in range(min(num_patches, len(keypoints))):
        keypoint = keypoints[i]
        x, y = map(int, keypoint.pt)  # Convert coordinates to integers
        top: int = y - patch_size // 2
        if top < 0:
            top = 0
        bottom: int = y + patch_size // 2
        if bottom > 224:
            bottom = 224
        left: int = x - patch_size // 2
        if left < 0:
            left = 0
        right: int = x + patch_size // 2
        if right > 224:
            right = 224
        patch = image[top:bottom, left:right]
        ax = fig.add_subplot(gs[i // grid_size, i % grid_size + 1])
        ax.imshow(patch, cmap='gray')
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def plot_img_sift_patches():
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
                                               shuffle=True,
                                               num_workers=generic_config["workers"],
                                               drop_last=False)
    # -----------------------------------------------------------------------------------
    # --- Keypoint extraction and feature description ---
    key_points_extractor = FeatureExtractingAlgorithm(algorithm="SIFT", nfeatures=400, logger=logger)
    keypoints, descriptors = key_points_extractor.get_keypoints_and_descriptors(train_loader)

    for idx, (img, label) in enumerate(train_loader):
        plot_keypoint_patches(img, keypoints[idx])


def extract_patch(img, keypoint, patch_size: int = 32):
    x, y = map(int, keypoint.pt)  # Convert coordinates to integers
    top: int = y - patch_size // 2
    if top < 0:
        top = 0
    bottom: int = y + patch_size // 2
    if bottom > 224:
        bottom = 224
    left: int = x - patch_size // 2
    if left < 0:
        left = 0
    right: int = x + patch_size // 2
    if right > 224:
        right = 224
    patch = img[top:bottom, left:right]
    return patch


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

    for i in range(min(num_patches, len(patches_list))):
        patch = patches_list[i]
        ax = axes[i // grid_size, i % grid_size]
        ax.imshow(patch, cmap='gray')
        ax.axis('off')

        # Add red square around the center of the patch
        rect = patches.Rectangle((8, 8), 16, 16, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)

    plt.show()
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
                                               shuffle=True,
                                               num_workers=generic_config["workers"],
                                               drop_last=False)
    # -----------------------------------------------------------------------------------
    # --- Keypoint extraction and feature description ---
    key_points_extractor = FeatureExtractingAlgorithm(algorithm="SIFT", logger=logger)
    keypoints, descriptors = key_points_extractor.get_keypoints_and_descriptors(train_loader)

    # -- KMEANS Clustering --
    flat_descriptors = np.concatenate(descriptors)
    clusterer = Clusterer(algorithm="KMEANS", logger=logger, **clustering_config["kmeans_args"])
    clusterer.fit_predict(flat_descriptors)

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
        # print(img)

    for i in range(10):
        plot_patches(cluster_patches[i], i)
        # plot_patches(cluster_patches[5], 5)
        # plot_patches(cluster_patches[12], 12)
        # plot_patches(cluster_patches[222], 222)
        # plot_patches(cluster_patches[3], 3)
        # plot_patches(cluster_patches[30], 30)
    # Go through images
    # Find descriptors
    # Predict the cluster they belong to (?)
    # Extract patches
    # Add them to a list
    # Plot clusters made this way

    # preprocessed_image = []
    # for image, label in train_loader:
    #     # image = gray(image)
    #     keypoint, descriptor = key_points_extractor.get_keypoints_and_descriptors(image)
    #     if descriptor is not None:
    #         histogram = build_histogram(descriptor, clusterer.clusterer)
    #         preprocessed_image.append(histogram)
    # print(preprocessed_image)


# def build_histogram(descriptor_list, cluster_alg):
#     histogram = np.zeros(len(cluster_alg.cluster_centers_))
#     cluster_result = cluster_alg.predict(descriptor_list)
#     for i in cluster_result:
#         histogram[i] += 1.0
#     return histogram


if __name__ == "__main__":
    plot_cluster_sift_patches()
