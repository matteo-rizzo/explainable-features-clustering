from typing import Tuple, List

import cv2
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from classes.MNISTDataset import MNISTDataset

"""
Extracting key points from an image is a common task in computer vision and can be done using various techniques such 
as feature detection and extraction. One popular method for feature detection is using Scale-Invariant Feature 
Transform (SIFT) algorithm. To use the SIFT algorithm we leverage the the OpenCV library.

Tutorial: https://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html

In the code below, the kp variable contains the detected key points and des variable contains the computed descriptors
for each key point. You can use these key points and descriptors for various computer vision tasks such as object
recognition, image stitching, and so on.
"""


class SIFT:

    def __init__(self, **kwargs):
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
        self.__sift = cv2.SIFT_create(**kwargs)

    def run(self, img: np.ndarray) -> Tuple[Tuple, np.ndarray]:
        return self.__sift.detectAndCompute(img, None)

    @staticmethod
    def plot_keypoints(img: np.ndarray, keypoints: Tuple):
        plt.imshow(cv2.drawKeypoints(img, keypoints, None))
        plt.show()
        plt.clf()

    def get_descriptors(self, dataloader: DataLoader) -> List:
        descriptors = []
        for (x, _) in tqdm(dataloader, desc="Generating descriptors using SIFT"):
            # Make numpy -> Squeeze 1 (grayscale) dim -> go from float to 0-255 representation
            img = (x.numpy().squeeze() * 255).astype(np.uint8)
            # img = x.squeeze(0).permute(1, 2, 0).numpy()
            _, img_descriptors = self.run(img)
            if img_descriptors is not None:
                descriptors.append(img_descriptors)
        return descriptors

    def get_keypoints(self, dataloader: DataLoader) -> List:
        keypoints = []
        for (x, _) in tqdm(dataloader, desc="Generating keypoints using SIFT"):
            # Make numpy -> Squeeze 1 (grayscale) dim -> go from float to 0-255 representation
            img = (x.numpy().squeeze() * 255).astype(np.uint8)
            # img = x.squeeze(0).permute(1, 2, 0).numpy()
            img_keypoints, _ = self.run(img)
            if img_keypoints is not None:
                keypoints.append(img_keypoints)
        return keypoints

    def get_keypoints_and_descriptors(self, dataloader: DataLoader) -> Tuple[List, List]:
        descriptors, keypoints = [], []
        for (x, _) in tqdm(dataloader, desc="Generating keypoints and descriptors using SIFT"):
            # Make numpy -> Squeeze 1 (grayscale) dim -> go from float to 0-255 representation
            img = (x.numpy().squeeze() * 255).astype(np.uint8)
            # img = x.squeeze(0).permute(1, 2, 0).numpy()
            img_keypoints, img_descriptors = self.run(img)
            if img_descriptors is not None:
                keypoints.append(img_keypoints)
                descriptors.append(img_descriptors)
        return keypoints, descriptors


def main():
    dataset = MNISTDataset()
    dataloader = DataLoader(dataset, shuffle=True)
    sift = SIFT()

    for (x, _) in dataloader:
        img = x.squeeze(0).permute(1, 2, 0).numpy()
        kp, des = sift.run(img)
        sift.plot_keypoints(img, kp)


if __name__ == "__main__":
    main()
