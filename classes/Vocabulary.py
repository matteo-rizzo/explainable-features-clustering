from typing import List, Tuple

import cv2
import numpy as np
from torch import Tensor


class Vocabulary:

    def __init__(self, visual_words: List):
        self.__visual_words = visual_words

    def embed(self, data: List[Tensor], labels: List) -> Tuple:
        # loop over all images in the dataset
        X, y = [], []
        for img, label in zip(data, labels):

            # initialize the sliding window
            window_size = (50, 50)
            stride = 25
            (win_w, win_h) = window_size
            for i in range(0, img.shape[1] - win_w, stride):
                for j in range(0, img.shape[0] - win_h, stride):
                    # extract SIFT features from the current window
                    window = img[j:j + win_h, i:i + win_w]
                    kp, desc = img.detectAndCompute(window, None)

                    # match the features with the visual words using cross correlation
                    matches = cv2.matchTemplate(self.__visual_words, desc, cv2.TM_CCOEFF_NORMED)

                    # compute the histogram of visual words for the current window
                    hist, _ = np.histogram(np.argmax(matches, axis=1), bins=range(len(self.__visual_words) + 1))

                    # append the histogram to the feature matrix
                    X.append(hist)
                    y.append(label)
        return X, y

    def match(self, descriptors, importance_weights) -> np.ndarray:
        return np.dot(descriptors, self.__visual_words.T * importance_weights)
