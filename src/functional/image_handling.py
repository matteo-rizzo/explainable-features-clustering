import math

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt


def kps_to_heatmaps(kps: tuple[cv2.KeyPoint], cluster_indexes: np.ndarray[int], heatmap_args: tuple[int, int, int]):
    """
    Transforms the keypoints of an image into an n dimensional heatmap where n is the number of clusters
    in which the keypoints have been separated.

    :param kps: list/tuple of kps for an image
    :param cluster_indexes: Cluster indexes for each keypoint
    :param heatmap_args: arguments to create the empty heatmap (num_clusters, width, height)
    :return: heatmap of appropriately scaled and rotated keypoints
    """
    # TODO: bottleneck
    layers, img_w, img_h = heatmap_args
    heatmap = torch.zeros(heatmap_args)

    for kp, cluster_idx in zip(kps, cluster_indexes):
        y_coord, x_coord = kp.pt
        scale = kp.size  # Keypoint scale parameter
        angle = kp.angle  # Keypoint angle parameter
        # Generate a grid of coordinates corresponding to the heatmap indices
        x_indices, y_indices = torch.meshgrid(torch.arange(img_w), torch.arange(img_h), indexing='ij')
        # Shift the coordinates so that the keypoint is at the origin
        x_indices = x_indices - x_coord
        y_indices = y_indices - y_coord
        # Rotate the coordinates by the keypoint angle
        cos_angle = math.cos(math.radians(angle))
        sin_angle = math.sin(math.radians(angle))
        x_rotated = cos_angle * x_indices - sin_angle * y_indices
        y_rotated = sin_angle * x_indices + cos_angle * y_indices
        # Squeeze the coordinates along the y-axis
        y_squeezed = y_rotated / 2  # Change this value to control the amount of squeezing
        # Calculate the squared distance from each grid point to the center (0, 0)
        squared_dist = x_rotated ** 2 + y_squeezed ** 2
        sigma = scale  # stddev / sqrt of variance
        # Calculate the Gaussian distribution
        gaussian = torch.exp(-squared_dist / (2 * sigma ** 2))
        # Add the Gaussian values to the heatmap for the corresponding cluster
        heatmap[cluster_idx] += gaussian
    return heatmap


def draw_activation(activation_maps, label: str = "Activation"):
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

    # Set the overall title using the label parameter
    fig.suptitle(label, fontsize=16)
    # Create a big colorbar on the right side
    cax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
    cb = fig.colorbar(im, cax=cax)
    # plt.tight_layout()
    plt.show()

    plt.close()
    summed_tensor = torch.sum(activation_maps, dim=0)
    plt.imshow(summed_tensor , cmap='inferno')
    plt.title("All summed")
    plt.show()
