import logging

import cv2
import numpy as np
import torch
from torch import Tensor

from functional.utilities.arc_utils import make_divisible
from visualization.image_visualization import draw_activation


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
    # Generate a grid of coordinates corresponding to the heatmap indices
    x_indices, y_indices = torch.meshgrid(torch.arange(img_w), torch.arange(img_h), indexing='ij')
    angles: list = []
    coords: list = []
    scales: list = []
    for kp in kps:
        angles.append(kp.angle)
        coords.append(kp.pt)
        scales.append(kp.size)
    # --- Angles ---
    # TODO: small loss of precision in angles
    angles: torch.Tensor = torch.FloatTensor(angles)
    cos_angles = torch.cos(torch.deg2rad(angles))
    sin_angles = torch.sin(torch.deg2rad(angles))
    for idx, cluster_idx in enumerate(cluster_indexes):
        # Shift the coordinates so that the keypoint is at the origin
        y_coord, x_coord = coords[idx]
        x_indices_shifted = x_indices - x_coord
        y_indices_shifted = y_indices - y_coord
        # Rotate the coordinates by the keypoint angle
        x_rotated = cos_angles[idx] * x_indices_shifted - sin_angles[idx] * y_indices_shifted
        y_rotated = sin_angles[idx] * x_indices_shifted + cos_angles[idx] * y_indices_shifted
        # Squeeze the coordinates along the y-axis
        y_squeezed = y_rotated / 2  # Change this value to control the amount of squeezing
        # Calculate the squared distance from each grid point to the center (0, 0)
        squared_dist = x_rotated ** 2 + y_squeezed ** 2
        sigma = scales[idx]  # stddev / sqrt of variance
        # Calculate the Gaussian distribution
        gaussian = torch.exp(-squared_dist / (2 * sigma ** 2))
        # Add the Gaussian values to the heatmap for the corresponding cluster
        heatmap[cluster_idx] += gaussian
    return heatmap

def kps_to_mask_heatmaps(image,
                         kps: tuple[cv2.KeyPoint],
                         cluster_indexes: np.ndarray[int],
                         heatmap_args: tuple[int, int, int]):
    layers, img_w, img_h = heatmap_args
    heatmap = torch.zeros(heatmap_args)
    # Generate a grid of coordinates corresponding to the heatmap indices
    x_indices, y_indices = torch.meshgrid(torch.arange(img_w), torch.arange(img_h), indexing='ij')
    coords: list = []
    scales: list = []
    for kp in kps:
        coords.append(kp.pt)
        scales.append(kp.size)

    # Just scale because rotation information is embedded in pixels
    for idx, cluster_idx in enumerate(cluster_indexes):
        # Shift the coordinates so that the keypoint is at the origin
        y_coord, x_coord = coords[idx]

        # Calculate the squared distance from each grid point to the center (0, 0)
        squared_dist = (x_indices - x_coord) ** 2 + (y_indices - y_coord) ** 2
        sigma = scales[idx]  # stddev / sqrt of variance
        # Calculate the Gaussian distribution
        gaussian = torch.exp(-squared_dist / (2 * sigma ** 2))
        # Add the Gaussian values to the heatmap for the corresponding cluster
        heatmap[cluster_idx] += gaussian
    # Normalize to max 1
    heatmap /= heatmap.max()
    heatmap = image * heatmap
    # draw_activation(heatmap)
    return heatmap




def check_img_size(img_size: int, stride: int = 32, logger: logging.Logger = logging.getLogger(__name__)) -> int:
    # Verify img_size is a multiple of stride s
    new_size: int = make_divisible(img_size, int(stride))  # ceil gs-multiple
    if new_size != img_size:
        logger.warning(f'WARNING: --img-size {img_size:g} must be multiple '
                       f'of max stride {stride:g}, updating to {new_size:g}')
    return new_size


def rescale_img(img: Tensor) -> np.ndarray:
    return (img.cpu().numpy() * 255).astype(np.uint8)
