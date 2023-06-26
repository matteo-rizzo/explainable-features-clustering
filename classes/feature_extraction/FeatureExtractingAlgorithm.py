import logging
from typing import Tuple, List, Optional

import cv2
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from classes.feature_extraction.CornerExtractingAlgorithm import CornerExtractingAlgorithm
from functional.utils import rescale_img
from torch import Tensor

class FeatureExtractingAlgorithm:

    def __init__(self, algorithm: str = "SIFT", logger: logging.Logger = logging.getLogger(__name__),
                 multi_scale: bool = False, **kwargs):
        """
        :param algorithm: name of feature extraction algorithm to be used must be one of the following:
        "SIFT", "ORB", "KAZE", "AKAZE", "FREAK", "BRISK", "AGAST"
        :param logger: tool for logging information throughout the framework
        :param kwargs:
            :param nfeatures: number of keypoints to detect. This can be used to limit the number of keypoints returned
            by the algorithm. By default, it is set to 0, which means that all keypoints are detected.
            :param nOctaveLayers: number of layers in each octave of the scale space. Increasing this parameter can lead
            to more keypoints being detected, but can also increase the computation time. By default, it is set to 3.
            :param contrastThreshold: threshold for the contrast of the detected keypoints. Keypoints with a lower 
            contrast than this value will be discarded. By default, it is set to 0.04.
            :param edgeThreshold: threshold for the edge response of the detected keypoints. Keypoints with an edge 
            response lower than this value will be discarded. By default, edgeThreshold is set to 10.
            :param sigma: initial Gaussian blur applied to the image before constructing the scale space. By default, it
            is set to 1.6.
        """
        self.__algorithm_name: str = algorithm.upper()
        self.__logger: logging.Logger = logger
        self.__algorithms = {
            # (Scale-Invariant Feature Transform) is a feature detection and description algorithm that detects
            # scale-invariant keypoints and computes descriptors based on gradient orientation histograms.
            "SIFT": cv2.SIFT_create,
            # (Oriented FAST and Rotated BRIEF) is a fast and robust feature detection and description algorithm
            # combining the FAST corner detector and the BRIEF descriptor with additional orientation information.
            "ORB": cv2.ORB_create,
            # ("KAZE features") is a nonlinear scale-space corner detection and feature extraction algorithm
            # that is based on the nonlinear scale space theory and the concept of the Difference of Gaussian (DoG)
            # scale space. It is capable of detecting and describing both keypoints and scale-invariant local features
            # in images.
            "KAZE": cv2.KAZE_create,
            # (Accelerated KAZE) is designed to be faster and more efficient than the original algorithm,
            # while maintaining its robustness to scale changes and other image transformations.
            "AKAZE": cv2.AKAZE_create,
            # (Fast Retina Keypoint): A fast keypoint detector and descriptor that extracts features in
            # a retina-like way.
            "FREAK": cv2.xfeatures2d.FREAK_create,
            # (Binary Robust Invariant Scalable Keypoints): A scale- and rotation-invariant detector and descriptor that
            # uses a binary descriptor instead of a floating-point one for efficiency.
            "BRISK": cv2.BRISK_create,
            # (Adaptive and Generic Accelerated Segment Test): A variant of FAST corner detector that is adaptive to
            # different image structures and performs well on noisy images.
            "AGAST": cv2.AgastFeatureDetector_create,
            # TODO: notes
            "SHI-TOMASI_BOX": CornerExtractingAlgorithm(algorithm="SHI-TOMASI", multi_scale=multi_scale, logger=logger,
                                                        **kwargs)
        }

        if self.__algorithm_name not in self.__algorithms.keys():
            raise ValueError(f"Invalid algorithm selected. Must be one of: {self.__algorithms.keys()}")
        if self.__algorithm_name == "SHI-TOMASI_BOX":
            self.__algorithm = self.__algorithms[self.__algorithm_name]
        else:
            self.__algorithm = self.__algorithms[self.__algorithm_name](**kwargs)

    def run(self, img: np.ndarray) -> Tuple[Tuple, Optional[np.ndarray]]:
        if self.__algorithm_name in ["MSER", "FAST", "AGAST"]:
            return self.__algorithm.detect(img, None), None
        elif self.__algorithm_name == "SHI-TOMASI_BOX":
            return self.__algorithm.run(img)
        else:
            return self.__algorithm.detectAndCompute(img, None)

    @staticmethod
    def plot_keypoints(img: np.ndarray, keypoints: Tuple):
        plt.imshow(cv2.drawKeypoints(img, keypoints, None))
        plt.show()
        plt.clf()

    def get_keypoints_and_descriptors(self, data: DataLoader | Tensor) \
            -> Tuple[List, List] | Tuple[Tuple, Optional[np.ndarray]]:
        descriptors, keypoints = [], []
        # ----------------------------------------------
        if isinstance(data, DataLoader):
            for (images, _) in tqdm(data, desc=f"Generating keypoints and descriptors using {self.__algorithm_name}"):
                for i in range(images.shape[0]):
                    img = rescale_img(images[i]).squeeze()
                    # NOTE: no, we're making them gray now
                    # if rgb:
                    #     img = cv2.cvtColor(img.transpose((1, 2, 0)), cv2.COLOR_RGB2BGR)
                    img_keypoints, img_descriptors = self.run(img)
                    if img_descriptors is not None:
                        keypoints.append(img_keypoints)
                        descriptors.append(img_descriptors)
            return keypoints, descriptors
        # ----------------------------------------------
        # Single image
        elif isinstance(data, Tensor):
            img = rescale_img(data).squeeze()
            # if rgb:
            #     img = cv2.cvtColor(img.transpose((1, 2, 0)), cv2.COLOR_RGB2BGR)
            return self.run(img)
        # ----------------------------------------------
        else:
            raise ValueError("Data should be either Tensor or Dataloader")



if __name__ == "__main__":
    FeatureExtractingAlgorithm("SHI-TOMASI")
