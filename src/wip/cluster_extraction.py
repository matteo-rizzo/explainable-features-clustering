import logging
import os
import time
from pathlib import Path

import cv2
import joblib
import numpy as np

from src.classes.clustering.Clusterer import Clusterer
from src.classes.feature_extraction.FeatureExtractingAlgorithm import FeatureExtractingAlgorithm


def extract_and_cluster(clustering_config: dict,
                        key_points_extractor: FeatureExtractingAlgorithm,
                        logger: logging.Logger,
                        data_loader,
                        train: bool):
    containing_folder: Path = Path("dumps/clustering")
    containing_folder.mkdir(exist_ok=True)
    keypoints_file = containing_folder / f'keypoints_{"train" if train else "test"}.joblib'
    descriptors_file = containing_folder / f'descriptors_{"train" if train else "test"}.joblib'
    if os.path.exists(keypoints_file) and os.path.exists(descriptors_file):
        # Load keypoints, descriptors, and clustering results from files
        t0 = time.perf_counter()
        logger.info(f"Loading {'train' if train else 'test'} keypoints, descriptors from file...")
        keypoints_list = joblib.load(keypoints_file)
        keypoints = list_to_keypoints(keypoints_list)
        descriptors = joblib.load(descriptors_file)
        logger.info(f"Loaded {'train' if train else 'test'} keypoints, descriptors from file in {(time.perf_counter() - t0):.2f}s.")
    else:
        t0 = time.perf_counter()
        keypoints, descriptors = key_points_extractor.get_keypoints_and_descriptors(data_loader)
        keypoints_list = keypoints_to_list(keypoints)
        joblib.dump(keypoints_list, keypoints_file)
        joblib.dump(descriptors, descriptors_file)
        logger.info(f"Saved keypoints and descriptors to files in {(time.perf_counter() - t0):.2f}s. ")
    # -- KMEANS Clustering --
    # Note: clustering should be done on train data only ? TODO: verify
    clustering_file = containing_folder / 'clustering.joblib'
    if os.path.exists(clustering_file):
        t0 = time.perf_counter()
        logger.info("Loading clustering from file...")
        # Load clustering results from file
        clusterer = joblib.load(clustering_file)
        logger.info(f"Loaded clustering results from file in {(time.perf_counter() - t0):.2f}s.")
    else:
        if not train:
            raise FileNotFoundError("Clustering should already exist for test and was not found.")
        t0 = time.perf_counter()
        flat_descriptors = np.concatenate(descriptors)
        clusterer = Clusterer(algorithm="KMEANS", logger=logger, **clustering_config["kmeans_args"])
        clusterer.fit_predict(flat_descriptors)
        joblib.dump(clusterer, clustering_file)
        logger.info(f"Saved clustering results to file in {(time.perf_counter() - t0):.2f}s.")
    return clusterer, descriptors, keypoints


def keypoints_to_list(per_img_kp_list: list):
    overall_kps: list = []
    for kp_list in per_img_kp_list:
        img_kp_list = []
        for kp in kp_list:
            kp_dict = {
                'pt': kp.pt,
                'size': kp.size,
                'angle': kp.angle,
                'response': kp.response,
                'octave': kp.octave,
                'class_id': kp.class_id
            }
            img_kp_list.append(kp_dict)
        overall_kps.append(img_kp_list)
    return overall_kps


def list_to_keypoints(listified_kps: list):
    overall_kps = []
    for per_image_list in listified_kps:
        keypoints = []
        for kp_dict in per_image_list:
            x, y = kp_dict['pt']
            kp = cv2.KeyPoint(x=x, y=y, size=kp_dict['size'],
                              angle=kp_dict['angle'],
                              response=kp_dict['response'],
                              octave=kp_dict['octave'],
                              class_id=kp_dict['class_id'],
                              )
            keypoints.append(kp)
        overall_kps.append(tuple(keypoints))
    return overall_kps
