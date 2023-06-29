import itertools
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.classes.feature_extraction.FeatureExtractingAlgorithm import FeatureExtractingAlgorithm
from src.classes.data.MNISTDataset import MNISTDataset


class KMeansClustering:

    def __init__(self, n_clusters: int = 10):
        self.__clustering = KMeans(n_clusters=n_clusters, n_init='auto', random_state=0)

    def get_centroids(self):
        return self.__clustering.cluster_centers_

    def get_labels(self):
        return self.__clustering.labels_

    def fit(self, data: np.ndarray) -> sklearn.cluster:
        return self.__clustering.fit(data)

    def predict(self, data: np.ndarray) -> np.ndarray:
        return self.__clustering.predict(data)

    def fit_predict(self, data: np.ndarray) -> np.ndarray:
        return self.__clustering.fit_predict(data)

    @staticmethod
    def rank_clusters(data: np.ndarray, centroids: np.ndarray, labels: List) -> List[Tuple]:
        clusters_ranking = []
        for i in tqdm(range(np.max(labels) + 1), desc="Ranking clusters"):
            cluster_variance = np.var(data[labels == i], axis=0)
            cluster_distance = np.linalg.norm(data[labels == i] - centroids[i], axis=1)
            clusters_ranking.append((i, np.sum(cluster_variance) / np.sum(cluster_distance)))

        for i, ranking in enumerate(clusters_ranking):
            print(f"Cluster {i}: Importance Score = {ranking}")

        return sorted(clusters_ranking, key=lambda x: x[1], reverse=True)

    @staticmethod
    def find_optimal_n_clusters(data: np.ndarray, start: int = 2, end: int = 100) -> int:
        distortions, silhouette_scores = [], []
        k_range = range(start, end)
        for k in k_range:
            clustering = KMeans(n_clusters=k, n_init='auto', random_state=0)
            clustering.fit(data)
            distortions.append(clustering.inertia_)
            silhouette_scores.append(silhouette_score(data, clustering.labels_))

        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(k_range, distortions, 'bx-')
        ax[0].set_xlabel('Number of Clusters')
        ax[0].set_ylabel('Distortion')
        ax[0].set_title('Elbow Method')
        ax[0].grid(True)

        ax[1].plot(k_range, silhouette_scores, 'bx-')
        ax[1].set_xlabel('Number of Clusters')
        ax[1].set_ylabel('Silhouette Score')
        ax[1].set_title('Silhouette Method')
        ax[1].grid(True)

        plt.show()

        optimal_n_clusters = np.argmax(silhouette_scores) + 2
        print(f"The optimal number of clusters is {optimal_n_clusters}")

        return optimal_n_clusters

    def plot_sample(self, data: np.ndarray, centroids: np.ndarray, labels: List, sample_size: int = 1000):
        sample = np.random.randint(0, len(data), size=sample_size)
        color_map = self.__get_color_map(len(centroids))
        for i in tqdm(sample, desc="Plotting sample data points"):
            plt.scatter(data[i][0], data[i][1], c=color_map(labels[i]), alpha=0.5)
        for i in tqdm(range(len(centroids)), desc="Plotting sample centroids"):
            plt.scatter(centroids[i][0], centroids[i][1], marker='x', s=100, linewidths=3, c='k')
        plt.show()

    @staticmethod
    def __get_color_map(n, name='hsv'):
        return plt.get_cmap(name, n)


def main():
    dataset = MNISTDataset()
    dataloader = DataLoader(dataset, shuffle=True)

    sample_size = 10
    dataloader = list(itertools.islice(dataloader, sample_size))

    sift = FeatureExtractingAlgorithm()
    descriptors = []
    for (x, y) in tqdm(dataloader, desc="Generating descriptors using SIFT"):
        img = cv2.normalize(x.squeeze(0).permute(1, 2, 0).numpy(), None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        kp, des = sift.run(img)
        if des is not None:
            descriptors.append(des)

    flat_descriptors = np.concatenate(descriptors)

    # optimal_n_clusters = KMeansClustering.find_optimal_n_clusters(flat_descriptors)
    # clustering = KMeansClustering(optimal_n_clusters)
    clustering = KMeansClustering()

    clusters = clustering.fit(flat_descriptors)
    clusters_labels, clusters_centroids = clusters.labels_, clusters.cluster_centers_
    clustering.plot_sample(flat_descriptors, clusters_centroids, clusters_labels)
    clusters_ranking = clustering.rank_clusters(flat_descriptors, clusters_centroids, clusters_labels)

    clustered_data, idx = [], 0
    for img_descriptors in tqdm(descriptors, desc="Packing data"):
        if img_descriptors is None:
            continue
        num_img_des = img_descriptors.shape[0]
        des_clusters_labels = clusters_labels[idx:idx + num_img_des]
        des_clusters_rankings = [rank for i in des_clusters_labels for (j, rank) in clusters_ranking if i == j]
        print(des_clusters_labels, des_clusters_rankings)
        img_data = (des_clusters_labels, des_clusters_rankings, img_descriptors)
        clustered_data.append(img_data)
        idx += num_img_des


if __name__ == "__main__":
    main()
