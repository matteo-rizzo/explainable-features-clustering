from typing import List, Tuple

import cv2
import numpy as np
from torch import Tensor

from classes.FeautureExtractingAlgorithm import FeautureExtractingAlgorithm


class Vocabulary:

    def __init__(self, visual_words: List):
        self.__visual_words = visual_words
        self.__sift = FeautureExtractingAlgorithm()

    def match_words(self, x: np.ndarray) -> List:
        # match the features with the visual words using cross correlation
        return cv2.matchTemplate(self.__visual_words, x, cv2.TM_CCOEFF_NORMED)

    def embed(self, images: List[Tensor], labels: List, window_size=(50, 50), stride=25) -> Tuple:
        windows_embeddings, windows_labels = [], []
        (win_w, win_h) = window_size

        for img, label in zip(images, labels):
            for i in range(0, img.shape[1] - win_w, stride):
                for j in range(0, img.shape[0] - win_h, stride):
                    window = img[j:j + win_h, i:i + win_w]
                    _, desc = self.__sift.run(window)

                    matches = self.match_words(desc)
                    hist, _ = np.histogram(np.argmax(matches, axis=1), bins=range(len(self.__visual_words) + 1))

                    windows_embeddings.append(hist)
                    windows_labels.append(label)

        return windows_embeddings, windows_labels

    def weight_by_importance(self, x: np.ndarray, importance_weights: np.ndarray) -> np.ndarray:
        return np.dot(x, self.__visual_words.T * importance_weights)
