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
        if not np.any(window != 0):
            return np.full((len(self.__words),), -1)

        _, desc = self.__feature_extractor.run(window)
        if desc is None:
            return np.full((len(self.__words),), -1)

        matches = self.match_words(desc)
        hist, _ = np.histogram(np.argmax(matches, axis=1), bins=range(len(self.__words) + 1))
        return hist

    def embed(self, images: Tensor, window_size: Tuple = (7, 7), stride: int = 1) -> Tensor:
        batched_embeddings, (win_w, win_h) = [], window_size
        for batch_idx in range(images.shape[0]):
            img = normalize_img(images[batch_idx]).squeeze()
            img_embedding = []
            for i in range(0, img.shape[1] - win_w, stride):
                for j in range(0, img.shape[0] - win_h, stride):
                    window = np.expand_dims(img[j:j + win_h, i:i + win_w], 0)
                    window_embedding = self.__embed_window(window)
                    img_embedding.append(window_embedding)
            batched_embeddings.append(torch.flatten(torch.tensor(img_embedding)))
        return torch.stack(batched_embeddings).unsqueeze(1).unsqueeze(1)
