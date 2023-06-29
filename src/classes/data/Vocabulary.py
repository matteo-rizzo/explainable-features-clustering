from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch import Tensor

from src.classes.clustering.Clusterer import Clusterer
from src.classes.feature_extraction.FeatureExtractingAlgorithm import FeatureExtractingAlgorithm

"""
Representing an image as matches of visual words over a sliding window is a common technique in computer vision and is
often used for tasks such as object detection and image retrieval. This technique involves breaking an image down into
small, overlapping regions called "patches", and representing each patch as a histogram of visual word occurrences. 
"""


class Vocabulary:

    def __init__(self, words: List, clusterer: Clusterer,
                 feature_extractor: FeatureExtractingAlgorithm = FeatureExtractingAlgorithm()):
        self.words = words
        self.feature_extractor = feature_extractor
        self.clusterer = clusterer

    def match_words(self, found_kps: list) -> List | np.ndarray:

        matches = []
        if not found_kps:
            histogram_edge: int = len(self.words) + 1
            matches = [histogram_edge]  # Add 1 to "no keypoints" embedding
            return matches
        else:
            for kp in found_kps:
                a = self.clusterer.predict(np.expand_dims(kp, 0))
                matches.append(int(a))
            return matches

    def embed_unordered(self, descriptors):
        histogram = np.zeros(len(self.clusterer.get_centroids()))
        cluster_result = self.clusterer.predict(descriptors)
        for i in cluster_result:
            histogram[i] += 1.0
        return torch.tensor(histogram).float()

    def embed(self, keypoints: tuple[cv2.KeyPoint], descriptors: np.ndarray,
              image_size: tuple = (224, 224),
              window_size: tuple = (56, 56),
              stride: int = 56):
        centroids = self.clusterer.get_centroids()
        (win_w, win_h) = window_size
        img_width, img_height = image_size
        img_embedding = []
        for height in range(0, img_height - win_h + 1, stride):
            for width in range(0, img_width - win_w + 1, stride):
                found_kps = []
                # Embed the window
                for idx, kp in enumerate(keypoints):
                    x, y = kp.pt
                    if height < y < height + win_w and width < x < width + win_w:
                        found_kps.append(descriptors[idx])

                matches = self.match_words(found_kps)
                window_embedding, _ = np.histogram(matches, bins=range(len(self.words) + 1))
                # FIXME: test
                img_embedding.append(window_embedding.astype(float))
                # img_embedding.append(np.multiply(np.expand_dims(window_embedding, 1), centroids))

        # shape: number of windows (e.g. 224 with stride 24 = 64), number of words, 128 (SIFT)
        img_embedding = np.array(img_embedding)
        # FIXME: experimental
        # img_embedding = np.vstack(img_embedding)
        return torch.flatten(torch.tensor(img_embedding)).float()
        # return torch.tensor(img_embedding).float()

    def __embed(self, images: Tensor, window_size: Tuple = (56, 56), stride: int = 56) -> Tensor:
        # TODO: try fractional stride
        # Initialize an empty list to store embeddings for each image in the batch
        batched_embeddings, (win_w, win_h) = [], window_size

        # Iterate over each image in the batch
        for batch_idx in range(images.shape[0]):
            # Normalize and squeeze the image
            # img = normalize_img(images[batch_idx]).squeeze()
            img = images[batch_idx]
            # Initialize an empty list to store embeddings for each window in the image
            img_embedding = []
            # Prima estraggo le feature
            # Poi mi segno le coordinate
            # E poi le considero in ogni finestra
            # TODO: parametrize rgb
            # keypoints, descriptors = self.__feature_extractor.get_keypoints_and_descriptors(img, rgb=True)
            # # Sort the keypoints based on y-axis first and x-axis second

            # sorted_keypoints = sorted(keypoints, key=lambda kp: (kp.pt[1], kp.pt[0]))
            centroids = self.clusterer.get_centroids()
            keypoints, descriptors = self.feature_extractor.get_keypoints_and_descriptors(img)
            # Sort keypoints and descriptors together based on y-axis first and x-axis second
            sorted_data = sorted(zip(keypoints, descriptors), key=lambda data: (data[0].pt[1], data[0].pt[0]))
            # Unzip the sorted data back into separate keypoints and descriptors lists
            sorted_keypoints, sorted_descriptors = zip(*sorted_data)
            # Iterate over each window in the image
            for height in range(0, img.shape[1] - win_h + 1, stride):
                for width in range(0, img.shape[2] - win_w + 1, stride):
                    # Extract the window from the image
                    # window = img[:, height_index:height_index + win_w, width_index:width_index + win_w]
                    # left = width
                    # right = width + win_w
                    # top = height
                    # bottom = height + win_h
                    found_kps = []
                    # Embed the window
                    for idx, kp in enumerate(sorted_keypoints):
                        x, y = kp.pt
                        if height < y < height + win_w and width < x < width + win_w:
                            found_kps.append(sorted_descriptors[idx])

                    matches = self.match_words(found_kps)

                    # Calculate a histogram of the word indices with the highest similarity scores
                    # +2 because all but the last (righthand-most) bin is half-open
                    # The last one is for empty windows
                    # window_embedding, _ = np.histogram(np.argmax(matches, axis=1), bins=range(len(self.words) + 2))
                    # +1 for empty
                    window_embedding, _ = np.histogram(matches, bins=range(len(self.words) + 1))

                    # window_embedding = self.__embed_window(window, sorted_keypoints)
                    # Append the window embedding to the image embedding list
                    # FIXME: Test
                    # img_embedding.append(window_embedding)
                    img_embedding.append(np.dot(window_embedding, centroids))
            # Flatten the image embedding list, convert it to a tensor,
            # and append it to the batched_embeddings list
            img_embedding = np.array(img_embedding)
            batched_embeddings.append(torch.flatten(torch.tensor(img_embedding)))
        # Stack the batched embeddings along the batch dimension and add singleton dimensions
        return torch.stack(batched_embeddings).float()

# mappe di presenza delle SIFT
# dataset cartelli stradali
