from typing import List

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
        # match the features with the visual words using cross correlation
        matches = []
        if np.any(x != 0):
            for word in self.__words:
                res = cv2.matchTemplate(word, x, cv2.TM_CCOEFF_NORMED)
                res = np.where(res >= threshold)
                matches.append(res)
        return matches

    def embed(self, img: Tensor, window_size=(7, 7), stride=1) -> torch.Tensor:
        embedding = []
        (win_w, win_h) = window_size
        img = normalize_img(img)
        for i in range(0, img.shape[1] - win_w, stride):
            for j in range(0, img.shape[0] - win_h, stride):
                window = np.expand_dims(img[j:j + win_h, i:i + win_w], 0)
                if np.any(window != 0):
                    _, desc = self.__feature_extractor.run(window)
                    if desc is None:
                        hist = np.full((len(self.__words),), -1)
                    else:
                        matches = self.match_words(desc)
                        hist, _ = np.histogram(np.argmax(matches, axis=1), bins=range(len(self.__words) + 1))
                else:
                    hist = np.full((len(self.__words),), -1)
                embedding.append(hist)

        embedding = torch.flatten(torch.Tensor(embedding))

        return embedding

    def weight_by_importance(self, x: np.ndarray, importance_weights: np.ndarray) -> np.ndarray:
        return np.dot(x, self.__words.T * importance_weights)
