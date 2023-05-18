import logging
from typing import Callable

import cv2
import numpy as np


class CornerExtractingAlgorithm:
    def __init__(self, algorithm: str = "SHI-TOMASI",
                 multi_scale: bool = False,
                 logger: logging.Logger = logging.getLogger(__name__)):
        self.logger: logging.Logger = logger
        self.name: str = algorithm
        self.multi_scale: bool = multi_scale
        if algorithm.upper() == "SHI-TOMASI":
            # maxCorners  = None  # Maximum number of corners to detect
            # qualityLevel  = 0.01  # Quality level threshold
            # minDistance  = 2  # Minimum distance between detected corners
            # blockSize  = 3  # Size of the neighborhood considered for corner detection
            # useHarrisDetector  = False  # Whether to use the Harris corner detector or not
            # k = 0.04  # Free parameter for the Harris detector
            self._algorithm: Callable = cv2.goodFeaturesToTrack
        elif algorithm.upper() == "HARRIS":
            # maxCorners = None  # Maximum number of corners to detect
            # qualityLevel = 0.01  # Quality level threshold
            # minDistance = 1  # Minimum distance between detected corners
            self._algorithm: Callable = cv2.cornerHarris

    def __call__(self, image: np.ndarray, **kwargs):
        if not self.multi_scale:
            return self._algorithm(image=image, **kwargs)
        else:
            return self.__apply_multiscale(image, **kwargs)

    def __apply_multiscale(self, image: np.ndarray, num_levels: int = 4, scale_factor: float = 1.2, **kwargs):
        pyramid = [image]  # Default size
        for i in range(1, num_levels):
            scaled = cv2.resize(pyramid[i - 1], (0, 0), fx=1 / scale_factor, fy=1 / scale_factor)
            pyramid.append(scaled)

        # Perform corner detection at each level of the pyramid
        corners = []
        for level, img in enumerate(pyramid):
            # Apply corner detection
            level_corners = self._algorithm(image=img, **kwargs)
            if level_corners is not None:
                level_corners = level_corners.reshape(-1, 2)  # Reshape corner coordinates
                level_corners *= (scale_factor ** level)  # Scale the corners back to the original image size
                corners.extend(level_corners)
        return corners


def main():
    args = {
        "maxCorners": None,  # Maximum number of corners to detect
        "qualityLevel": 0.01,  # Quality level threshold
        "minDistance": 2,  # Minimum distance between detected corners
        "blockSize": 3,  # Size of the neighborhood considered for corner detection
        "useHarrisDetector": False,  # Whether to use the Harris corner detector or not
        "k": 0.04  # Free parameter for the Harris detector
    }
    image = cv2.imread("dataset/26.png", 0)  # Read the image in grayscale
    fea = CornerExtractingAlgorithm(multi_scale=True)
    print(fea(image, **args))


if __name__ == "__main__":
    main()
