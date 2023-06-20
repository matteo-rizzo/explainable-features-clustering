from typing import List, Tuple

import numpy as np
import torch
from torch import Tensor

from classes.feature_extraction.FeatureExtractingAlgorithm import FeatureExtractingAlgorithm
from classes.clustering.Clusterer import Clusterer

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
            matches = np.full(histogram_edge, -1)
            matches[-1] = histogram_edge  # Add 1 to "no keypoints" embedding
            return matches
        else:
            for kp in found_kps:
                # a = self.clusterer.predict(kp)
                a = self.clusterer.predict(np.expand_dims(kp, 0))
                matches.append(int(a))
            # for kp in found_kps:
                # word_match = []
                # for word_index, word in enumerate(self.words):
                #     self.words
                    # res = cv2.matchTemplate(word, kp, cv2.TM_CCOEFF_NORMED)
                    # word_match.append(-1 if res[0][0] < threshold else word_index)
                # matches.append(word_match)

            # matches = np.where(matches >= threshold)
            return matches

    # def __embed_window(self, window: np.ndarray, sorted_keypoints) -> np.ndarray:
    #     present_kps = []
    #     for kp in sorted_keypoints:
    #         coords = kp.pt
    #     # Check if the window is empty (contains only zeros)
    #     if not np.any(window != 0):
    #         # Return an embedding array filled with -1 values
    #         return np.full((len(self.__words),), -1)
    #
    #     # Run feature extraction on the window
    #     _, desc = self.__feature_extractor.run(window)
    #     # If feature extraction failed or didn't produce a valid descriptor,
    #     # return an embedding array filled with -1 values
    #     if desc is None:
    #         return np.full((len(self.__words),), -1)
    #
    #     # Match the descriptor with predefined words to get similarity scores or matching results
    #     matches = self.match_words(desc)
    #
    #     # Calculate a histogram of the word indices with the highest similarity scores
    #     hist, _ = np.histogram(np.argmax(matches, axis=1), bins=range(len(self.__words) + 1))
    #
    #     # Return the histogram as the final embedding for the window
    #     return hist

    def embed(self, images: Tensor, window_size: Tuple = (28, 28), stride: int = 28) -> Tensor:
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
            keypoints, descriptors = self.feature_extractor.get_keypoints_and_descriptors(img, rgb=True)
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
