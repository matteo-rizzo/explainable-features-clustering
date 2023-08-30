import logging
import os
import shutil
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
                        train: bool = True,
                        clean: bool = False,
                        clustering_algorithm: str = "kmeans"):
    # --- INIT ---
    containing_folder: Path = Path("dumps/clustering")
    # Delete content if clean flag is specified
    if clean and containing_folder.exists():
        shutil.rmtree(containing_folder)
    containing_folder.mkdir(exist_ok=True)
    n_clusters: int = clustering_config[f"{clustering_algorithm}_args"]["n_clusters"] \
        if clustering_algorithm.upper() != "HDBSCAN" else "NA"
    # --------------------------------------------------------------------------------------------------
    # --- KPS & DESCRIPTORS ---
    keypoints_file = containing_folder / f'keypoints_{"train" if train else "test"}.joblib'
    descriptors_file = containing_folder / f'descriptors_{"train" if train else "test"}.joblib'
    # --- LOAD FROM FILE ---
    if os.path.exists(keypoints_file) and os.path.exists(descriptors_file):
        # Load keypoints, descriptors, and clustering results from files
        t0 = time.perf_counter()
        logger.info(f"Loading {'train' if train else 'test'} keypoints, descriptors from file...")
        keypoints_list = joblib.load(keypoints_file)
        keypoints = list_to_keypoints(keypoints_list)
        descriptors = joblib.load(descriptors_file)
        logger.info(
            f"Loaded {'train' if train else 'test'} keypoints, descriptors from file in {(time.perf_counter() - t0):.2f}s.")
    # --- CREATE NEW & SAVE TO FILE ---
    else:
        t0 = time.perf_counter()
        keypoints, descriptors = key_points_extractor.get_keypoints_and_descriptors(data_loader)
        keypoints_list = keypoints_to_list(keypoints)
        joblib.dump(keypoints_list, keypoints_file)
        joblib.dump(descriptors, descriptors_file)
        logger.info(f"Saved keypoints and descriptors to files in {(time.perf_counter() - t0):.2f}s. ")
    # --------------------------------------------------------------------------------------------------
    # -- CLUSTERING --
    # Note: clustering should be done on train data only ? TODO: verify
    clustering_file = containing_folder / (f'clustering_'
                                           f'{clustering_algorithm}_'
                                           f'{n_clusters}'
                                           f'.joblib')
    # --- LOAD FROM FILE ---
    if os.path.exists(clustering_file):
        t0 = time.perf_counter()
        logger.info("Loading clustering from file...")
        # Load clustering results from file
        clusterer = joblib.load(clustering_file)
        logger.info(f"Loaded {clustering_algorithm} clustering results "
                    f'[k = {n_clusters}] '
                    f"from file in {(time.perf_counter() - t0):.2f}s.")
    # --- CREATE NEW & SAVE TO FILE ---
    else:
        if not train:
            raise FileNotFoundError("Clustering should already exist for test and was not found.")
        t0 = time.perf_counter()
        flat_descriptors = np.concatenate(descriptors)
        clusterer = Clusterer(algorithm=clustering_algorithm, logger=logger,
                              **clustering_config[f"{clustering_algorithm}_args"])
        clusterer.fit_predict(flat_descriptors)
        joblib.dump(clusterer, clustering_file)
        logger.info(f"Saved clustering results to file in {(time.perf_counter() - t0):.2f}s.")
    # --- Return everything ---
    return clusterer, descriptors, keypoints


def keypoints_to_list(per_img_kp_list: list):
    # This is necessary to load/save KPs from file, because CV2 has issues
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
    # This is necessary to load/save KPs from file, because CV2 has issues
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

# Che succede se apro/chiudo negozio/museo prima?
# Individuare fonte di dati
# Come utilizzare i dati per modellare il flusso dei turisti
# AlmaViva può già accedere alle celle telefoniche; tipo come si muovono i turisti
# Come poi utilizzarle?
# Sistemi che fanno queste cose qui; come modellare il flusso di turisti e
# come alterare i parametri per avere cose alternative
# - Rappresentazione dei turisti (propriet' utili)
# Dati rilevanti e quali NON rilevanti; dati meteo possono tornare utili ad esempio
# Modelli per la rappresentazione del movimento dei turisti. Secondo quali regole?
# Review di serie spazio temporali, traiettorie e clustering##

# Favorire turismo sostenibile
# Esempio Data Appeal

# Come si rappresenta il turista?
# Come si rappresenta un POI?
# Come si formalizzano un po' tutte le cose; POI

# La ricerca bibliografica riguarderà i seguenti temi:
# Rappresentazione dei turisti
# Dati rilevanti e non rilevanti per l’analisi dei turisti
# Modelli per la rappresentazione del movimento dei turisti
# Review delle serie spazio-temporali, traiettorie e loro clustering

# Dato as granular as possible
# Modelli generici per aggiungere fonti arbitrarie a fonti di base
# Risvolto commerciale
# Simulazione delle variabili
# Magari dataset

# Spreading out tourist: estimating parameters before moving manopole?
# Simulation?