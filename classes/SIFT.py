from typing import Tuple

import cv2
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

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

    def __init__(self):
        self.__sift = cv2.SIFT_create()

    def run(self, img: np.ndarray) -> Tuple[Tuple, np.ndarray]:
        return self.__sift.detectAndCompute(img, None)


if __name__ == "__main__":
    dataset = MNISTDataset()
    dataloader = DataLoader(dataset, shuffle=True)
    sift = SIFT()

    for (x, _, _) in dataloader:
        img = x.squeeze(0).permute(1, 2, 0).numpy()
        kp, des = sift.run(img)
        plt.imshow(cv2.drawKeypoints(img, kp, None))
        plt.show()
