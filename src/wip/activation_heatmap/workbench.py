import math

import cv2
import matplotlib.pyplot as plt
import torch
import yaml

from src.classes.data.OxfordIIITPetDataset import OxfordIIITPetDataset
from src.classes.feature_extraction.FeatureExtractingAlgorithm import FeatureExtractingAlgorithm
from functional.utilities.image_handling import kps_to_heatmaps, rescale_img
from src.visualization.image_visualization import draw_activation
from functional.utilities.utils import default_logger
from src.wip.cluster_extraction import extract_and_cluster


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
        # Find keypoints
        kps, descs = key_points_extractor.get_keypoints_and_descriptors(img)

        # --- Show img ---
        plt.imshow(img.squeeze(), cmap='gray')
        plt.show()
        # --- Show kps ---
        output_image = cv2.drawKeypoints(rescale_img(img.squeeze()), kps, None,
                                         flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        plt.imshow(output_image)
        plt.show()
        # -----------------
        # ---------- WITH ONLY SCALE ----------
        # Predict clustering
        cluster_indexes = clusterer.predict(descs)
        # For now, create a tensor of 0 with as many layers as words
        giga_heatmap = torch.zeros((1, generic_config["img_size"], generic_config["img_size"]))
        heatmap = torch.zeros((clusterer.n_clusters(), generic_config["img_size"], generic_config["img_size"]))
        # Then, pair coordinates and cluster prediction to assign to layer
        for kp, cluster in zip(kps, cluster_indexes):
            y_coord, x_coord = kp.pt
            scale = kp.size  # Keypoint scale parameter

            # Generate a grid of coordinates corresponding to the heatmap indices
            x_indices, y_indices = torch.meshgrid(torch.arange(224), torch.arange(224), indexing='ij')
            # Calculate the squared distance from each grid point to the center (x_coord, y_coord)
            squared_dist = (x_indices - x_coord) ** 2 + (y_indices - y_coord) ** 2
            # TODO: TEST OTHER THINGS
            # Note: This works fairly well
            sigma = scale  # stddev / sqrt of variance
            # Calculate the Gaussian distribution
            # Approach 1 - Use scale as stdev
            gaussian = torch.exp(-squared_dist / (2 * sigma ** 2))
            # Approach 2 - Use scale as variance (no power)
            # gaussian = torch.exp(-squared_dist / (2 * sigma))
            # Add the Gaussian values to the heatmap for the corresponding cluster
            heatmap[cluster] += gaussian
            # TODO: remove
            giga_heatmap[0] += gaussian

        draw_activation(heatmap)
        draw_activation(giga_heatmap)
        # ---------- WITH ROTATION ----------
        # # Predict clustering
        # cluster_indexes = clusterer.predict(descs)
        # # For now, create a tensor of 0 with as many layers as words
        # giga_heatmap = torch.zeros((1, generic_config["img_size"], generic_config["img_size"]))
        # heatmap = torch.zeros((clusterer.n_clusters(), generic_config["img_size"], generic_config["img_size"]))
        # # Then, pair coordinates and cluster prediction to assign to layer
        # for kp, cluster in zip(kps, cluster_indexes):
        #     y_coord, x_coord = kp.pt
        #     scale = kp.size  # Keypoint scale parameter
        #     # angle = math.radians(kp.angle)  # Keypoint angle parameter (in radians)
        #     angle = 0
        #     # stddev / sqrt of variance
        #     sigma_major = scale * 5
        #     sigma_minor = scale / 5
        #
        #     # Generate a grid of coordinates corresponding to the heatmap indices
        #     x_indices, y_indices = torch.meshgrid(torch.arange(224), torch.arange(224), indexing='ij')
        #
        #     # Apply scale and angle transformations to the coordinates
        #     cos_theta = torch.cos(torch.tensor(angle))
        #     sin_theta = torch.sin(torch.tensor(angle))
        #     #          diff b/w coord n center | rotation by angle                       | add back to original
        #     grid_x_rot = (x_indices - x_coord) * cos_theta - (y_indices - y_coord) * sin_theta + x_coord
        #     grid_y_rot = (x_indices - x_coord) * sin_theta + (y_indices - y_coord) * cos_theta + y_coord
        #
        #     grid_x_stretch = (grid_x_rot - x_coord) / sigma_major
        #     grid_y_stretch = (grid_y_rot - y_coord) / sigma_minor
        #     # Calculate the Gaussian distribution
        #     gaussian = torch.exp(-0.5 * (grid_x_stretch ** 2 + grid_y_stretch ** 2))
        #
        #     # Normalize the Gaussian so that integral (total prob) is 1
        #     gaussian = gaussian / (math.pi * sigma_major * sigma_minor)
        #
        #     # Add the Gaussian values to the heatmap for the corresponding cluster
        #     heatmap[cluster] += gaussian
        #     # TODO: remove
        #     giga_heatmap[0] += gaussian
        #
        # draw_activation(heatmap)
        # draw_activation(giga_heatmap)
        # FIXME
        break
    logger.info("Program execution completed.")


def main2():
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
    # Load an image
    for img, label in train_loader:
        # Find keypoints
        kps, descs = key_points_extractor.get_keypoints_and_descriptors(img)
        # --- Show img ---
        # plt.imshow(img.squeeze(), cmap='gray')
        # plt.show()
        # # --- Show kps ---
        # output_image = cv2.drawKeypoints(rescale_img(img.squeeze()), kps, None,
        #                                  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # plt.imshow(output_image)
        # plt.show()
        # -----------------
        # Predict clustering
        cluster_indexes = clusterer.predict(descs)
        # For now, create a tensor of 0 with as many layers as words
        giga_heatmap = torch.zeros((1, generic_config["img_size"], generic_config["img_size"]))
        heatmap = torch.zeros((clusterer.n_clusters(), generic_config["img_size"], generic_config["img_size"]))
        # Then, pair coordinates and cluster prediction to assign to layer
        for kp, cluster in zip(kps, cluster_indexes):
            y_coord, x_coord = kp.pt
            scale = kp.size  # Keypoint scale parameter
            angle = kp.angle  # Keypoint angle parameter

            # Generate a grid of coordinates corresponding to the heatmap indices
            x_indices, y_indices = torch.meshgrid(torch.arange(224), torch.arange(224), indexing='ij')

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
            heatmap[cluster] += gaussian
            # TODO: remove
            giga_heatmap[0] += gaussian

        draw_activation(heatmap)
        draw_activation(giga_heatmap)
        # FIXME
        break
    logger.info("Program execution completed.")


def printer():
    heatmap = torch.zeros((9, 224, 224))

    # Gaussian parameters
    stddev_major = 50  # standard deviation along the major axis
    stddev_minor = 25  # standard deviation along the minor axis
    # variance = stddev ** 2
    angle = math.pi / 4

    for i in range(9):
        y_coord, x_coord = 112, 112

        # Generate a grid of coordinates
        x_indices, y_indices = torch.meshgrid(torch.arange(224), torch.arange(224), indexing='ij')

        angle += math.pi / 4

        # stddev_major = 50  # standard deviation along the major axis
        # stddev_minor += 5  # standard deviation along the minor axis

        # Rotate the coordinates
        cos_theta = torch.cos(torch.tensor(angle))
        sin_theta = torch.sin(torch.tensor(angle))
        grid_x_rot = (x_indices - x_coord) * cos_theta - (y_indices - y_coord) * sin_theta
        grid_y_rot = (x_indices - x_coord) * sin_theta + (y_indices - y_coord) * cos_theta

        grid_x_stretch = grid_x_rot / stddev_major
        grid_y_stretch = grid_y_rot / stddev_minor

        # Calculate the Gaussian distri bution
        gaussian = torch.exp(-0.5 * (grid_x_stretch ** 2 + grid_y_stretch ** 2))

        # Normalize the Gaussian so that integral (total prob) is 1
        gaussian = gaussian / (2 * math.pi * stddev_major * stddev_minor)

        # Assign the Gaussian to the corresponding layer
        heatmap[i] = gaussian

    draw_activation(heatmap)


def save():
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
    # Load an image
    for img, label in train_loader:
        # Find keypoints
        kps, descs = key_points_extractor.get_keypoints_and_descriptors(img)
        # Predict clustering
        cluster_indexes = clusterer.predict(descs)
        # Then, pair coordinates and cluster prediction to assign to layer
        heatmap = kps_to_heatmaps(kps, cluster_indexes, (clusterer.n_clusters(), generic_config["img_size"], generic_config["img_size"]))

        draw_activation(heatmap)
        # FIXME
        break
    logger.info("Program execution completed.")


if __name__ == "__main__":
    # main()
    # main2()
    save()

    # printer()
