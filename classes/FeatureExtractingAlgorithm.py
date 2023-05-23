import logging
from typing import Tuple, List, Optional

import cv2
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from classes.data.MNISTDataset import MNISTDataset

"""
Extracting key points from an image is a common task in computer vision and can be done using various techniques such 
as feature detection and extraction. One popular method for feature detection is using Scale-Invariant Feature 
Transform (SIFT) algorithm. To use the SIFT algorithm we leverage the the OpenCV library.

Tutorial: https://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html

In the code below, the kp variable contains the detected key points and des variable contains the computed descriptors
for each key point. You can use these key points and descriptors for various computer vision tasks such as object
recognition, image stitching, and so on.
"""


class FeatureExtractingAlgorithm:

    def __init__(self, algorithm: str = "SIFT", logger: logging.Logger = logging.getLogger(__name__), **kwargs):
        # nfeatures: The number of keypoints to detect.
        # This can be used to limit the number of keypoints returned by the algorithm.
        # By default, nfeatures is set to 0, which means that all keypoints are detected.

        # nOctaveLayers: The number of layers in each octave of the scale space.
        # Increasing this parameter can lead to more keypoints being detected,
        # but can also increase the computation time.
        # By default, nOctaveLayers is set to 3.

        # contrastThreshold: The threshold for the contrast of the detected keypoints.
        # Keypoints with a lower contrast than this value will be discarded.
        # By default, contrastThreshold is set to 0.04.

        # edgeThreshold: The threshold for the edge response of the detected keypoints.
        # Keypoints with an edge response lower than this value will be discarded.
        # By default, edgeThreshold is set to 10.

        # sigma: The initial Gaussian blur applied to the image before constructing the scale space.
        # By default, sigma is set to 1.6.
        self.name: str = algorithm
        self.logger: logging.Logger = logger
        if algorithm.upper() == "SIFT":
            # (Scale-Invariant Feature Transform) is a feature detection and description algorithm that detects
            # scale-invariant keypoints and computes descriptors based on gradient orientation histograms.
            self._algorithm = cv2.SIFT_create(**kwargs)
        elif algorithm.upper() == "ORB":
            # ORB (Oriented FAST and Rotated BRIEF) is a fast and robust feature detection and description algorithm
            # that combines the FAST corner detector and the BRIEF descriptor with additional orientation information.
            self._algorithm = cv2.ORB_create(**kwargs)
        elif algorithm.upper() == "KAZE":
            # (short for "KAZE Features") is a nonlinear scale-space corner detection and feature extraction algorithm
            # that is based on the nonlinear scale space theory and the concept of the Difference of Gaussian (DoG)
            # scale space. It is capable of detecting and describing both keypoints and scale-invariant local features
            # in images.
            self._algorithm = cv2.KAZE_create(**kwargs)
        elif algorithm.upper() == "AKAZE":
            # Accelerated kaze version - designed to be faster and more efficient than the original algorithm,
            # while maintaining its robustness to scale changes and other image transformations.
            self._algorithm = cv2.AKAZE_create(**kwargs)
        elif algorithm.upper() == "FREAK":
            raise NotImplementedError
            # (Fast Retina Keypoint): A fast keypoint detector and descriptor that
            # extracts features in a retina-like way.
            self._algorithm = cv2.xfeatures2d.FREAK_create(**kwargs)
        elif algorithm.upper() == "BRISK":
            # (Binary Robust Invariant Scalable Keypoints): A scale- and rotation-invariant detector
            # and descriptor that uses a binary descriptor instead of a floating-point one for efficiency.
            self._algorithm = cv2.BRISK_create(**kwargs)
        elif algorithm.upper() == "MSER":
            # (Maximally Stable Extremal Regions): A feature detection algorithm that finds regions
            # in an image which are stable under different transformations.
            self._algorithm = cv2.MSER_create(**kwargs)
        elif algorithm.upper() == "FAST":
            self._algorithm = cv2.FastFeatureDetector_create(**kwargs)
            #  (Binary Robust Invariant Scalable Keypoints): A scale- and rotation-invariant detector
            #  and descriptor that uses a binary descriptor instead of a floating-point one for efficiency.
        elif algorithm.upper() == "AGAST":
            # (Adaptive and Generic Accelerated Segment Test): A variant of FAST corner detector that is adaptive
            # to different image structures and performs well on noisy images.
            self._algorithm = cv2.AgastFeatureDetector_create(**kwargs)
        else:
            raise ValueError("Invalid algorithm selected. "
                             "Must be one of: [SIFT, ORB, KAZE, AKAZE, FREAK, BRISK, MSER, FAST, AGAST]")

    def run(self, img: np.ndarray) -> Tuple[Tuple, Optional[np.ndarray]]:
        if self.name.upper() in ["MSER", "FAST", "AGAST"]:
            return self._algorithm.detect(img, None), None
        else:
            return self._algorithm.detectAndCompute(img, None)

    @staticmethod
    def plot_keypoints(img: np.ndarray, keypoints: Tuple):
        plt.imshow(cv2.drawKeypoints(img, keypoints, None))
        plt.show()
        plt.clf()

    def get_keypoints_and_descriptors(self, dataloader: DataLoader) -> Tuple[List, List]:
        descriptors, keypoints = [], []
        for (x, _) in tqdm(dataloader, desc=f"Generating keypoints and descriptors using {self.name}"):
            # Make numpy -> Squeeze 1 (grayscale) dim -> go from float to 0-255 representation
            imgs = (x.numpy().squeeze(0) * 255).astype(np.uint8)
            for i in range(imgs.shape[0]):
                img_keypoints, img_descriptors = self.run(imgs[i])
                if img_descriptors is not None:
                    keypoints.append(img_keypoints)
                    descriptors.append(img_descriptors)
        return keypoints, descriptors


def main():
    dataset = MNISTDataset()
    dataloader = DataLoader(dataset, shuffle=True)
    sift = FeatureExtractingAlgorithm()

    for (x, _) in dataloader:
        img = x.squeeze(0).permute(1, 2, 0).numpy()
        kp, des = sift.run(img)
        sift.plot_keypoints(img, kp)


if __name__ == "__main__":
    main()
