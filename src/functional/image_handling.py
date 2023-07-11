import logging
import math

import cv2
import numpy as np
import torch
from torch import Tensor

from src.functional.arc_utils import make_divisible


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


def check_img_size(img_size: int, stride: int = 32, logger: logging.Logger = logging.getLogger(__name__)) -> int:
    # Verify img_size is a multiple of stride s
    new_size: int = make_divisible(img_size, int(stride))  # ceil gs-multiple
    if new_size != img_size:
        logger.warning(f'WARNING: --img-size {img_size:g} must be multiple '
                       f'of max stride {stride:g}, updating to {new_size:g}')
    return new_size


def rescale_img(img: Tensor) -> np.ndarray:
    return (img.cpu().numpy() * 255).astype(np.uint8)
