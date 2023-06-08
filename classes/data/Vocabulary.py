from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch import Tensor

from classes.FeatureExtractingAlgorithm import FeatureExtractingAlgorithm
from functional.utils import normalize_img

"""
Representing an image as matches of visual words over a sliding window is a common technique in computer vision and is
often used for tasks such as object detection and image retrieval. This technique involves breaking an image down into
small, overlapping regions called "patches", and representing each patch as a histogram of visual word occurrences. 
"""


class Vocabulary:

    def __init__(self, words: List):
        self.__words = words
        self.__feature_extractor = FeatureExtractingAlgorithm()

    def match_words(self, x: np.ndarray, threshold: int = 0.8) -> List:
        matches = []
        if np.any(x != 0):
            for word in self.__words:
                res = cv2.matchTemplate(word, x, cv2.TM_CCOEFF_NORMED)
                res = np.where(res >= threshold)
                matches.append(res)
        return matches

    def __embed_window(self, window: np.ndarray) -> np.ndarray:
        # Check if the window is empty (contains only zeros)
        if not np.any(window != 0):
            # Return an embedding array filled with -1 values
            return np.full((len(self.__words),), -1)

        # Run feature extraction on the window
        _, desc = self.__feature_extractor.run(window)
        # If feature extraction failed or didn't produce a valid descriptor,
        # return an embedding array filled with -1 values
        if desc is None:
            return np.full((len(self.__words),), -1)

        # Match the descriptor with predefined words to get similarity scores or matching results
        matches = self.match_words(desc)

        # Calculate a histogram of the word indices with the highest similarity scores
        hist, _ = np.histogram(np.argmax(matches, axis=1), bins=range(len(self.__words) + 1))

        # Return the histogram as the final embedding for the window
        return hist

    def embed(self, images: Tensor, window_size: Tuple = (7, 7), stride: int = 1) -> Tensor:
        # Initialize an empty list to store embeddings for each image in the batch
        batched_embeddings, (win_w, win_h) = [], window_size

        # Iterate over each image in the batch
        for batch_idx in range(images.shape[0]):
            # Normalize and squeeze the image
            img = normalize_img(images[batch_idx]).squeeze()
            # Initialize an empty list to store embeddings for each window in the image
            img_embedding = []
            # Iterate over each window in the image
            for i in range(0, img.shape[1] - win_w, stride):
                for j in range(0, img.shape[0] - win_h, stride):
                    # Extract the window from the image
                    window = np.expand_dims(img[j:j + win_h, i:i + win_w], 0)
                    # Embed the window
                    window_embedding = self.__embed_window(window)
                    # Append the window embedding to the image embedding list
                    img_embedding.append(window_embedding)
            # Flatten the image embedding list, convert it to a tensor,
            # and append it to the batched_embeddings list
            batched_embeddings.append(torch.flatten(torch.tensor(img_embedding)))
        # Stack the batched embeddings along the batch dimension and add singleton dimensions
        return torch.stack(batched_embeddings).unsqueeze(1).unsqueeze(1)
