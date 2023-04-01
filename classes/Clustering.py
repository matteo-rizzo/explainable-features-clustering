from typing import List

import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from classes.MNISTDataset import MNISTDataset
from classes.SIFT import SIFT


class KMeansClustering:

    def __init__(self, n_clusters: int = 100):
        self.__clustering = KMeans(n_clusters=n_clusters, n_init='auto', random_state=0)

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

    def plot_sample(self, data: np.ndarray, centroids: np.ndarray, labels: List, sample_size: int = 100000):
        sample = np.random.randint(0, len(data), size=sample_size)
        color_map = self.__get_color_map(len(centroids))
        for i in tqdm(sample):
            plt.scatter(data[i][0], data[i][1], c=color_map(labels[i]), alpha=0.5)
        for i in range(len(centroids)):
            plt.scatter(centroids[i][0], centroids[i][1], marker='x', s=100, linewidths=3, c='k')
        plt.show()

    @staticmethod
    def __get_color_map(n, name='hsv'):
        return plt.get_cmap(name, n)

    def run(self, data: np.ndarray) -> sklearn.cluster:
        return self.__clustering.fit(data)

    @staticmethod
    def rank_clusters(data: np.ndarray, centroids: np.ndarray, labels: List) -> List[int]:
        cluster_variances = []
        for i in range(np.max(labels) + 1):
            cluster_variances.append(np.var(data[labels == i], axis=0))

        cluster_distances = []
        for i in range(np.max(labels) + 1):
            distances = np.linalg.norm(data[labels == i] - centroids[i], axis=1)
            cluster_distances.append(distances)

        cluster_importances = []
        for i in range(np.max(labels) + 1):
            importance = np.sum(cluster_variances[i]) / np.sum(cluster_distances[i])
            cluster_importances.append(importance)

        for i, importance in enumerate(cluster_importances):
            print(f"Cluster {i}: Importance Score = {importance}")

        return cluster_importances


if __name__ == "__main__":
    dataset = MNISTDataset()
    dataloader = DataLoader(dataset, shuffle=True)

    sift = SIFT()
    descriptors, paths_to_file = [], []
    for (x, _, path_to_file) in tqdm(dataloader):
        img = x.squeeze(0).permute(1, 2, 0).numpy()
        _, img_descriptors = sift.run(img)
        if img_descriptors is not None:
            descriptors.append(img_descriptors)
            paths_to_file.append(path_to_file)

    flat_descriptors = np.concatenate(descriptors)

    # optimal_n_clusters = KMeansClustering.find_optimal_n_clusters(flat_descriptors)
    # clustering = KMeansClustering(optimal_n_clusters)
    clustering = KMeansClustering()

    clusters = clustering.run(flat_descriptors)
    labels, centroids = clusters.labels_, clusters.cluster_centers_
    clustering.plot_sample(flat_descriptors, centroids, labels, sample_size=400000)
    ranking = clustering.rank_clusters(flat_descriptors, centroids, labels)

    clustered_data, idx = [], 0
    for des, path_to_file in tqdm(zip(descriptors, paths_to_file)):
        if des is None:
            continue
        num_img_des = des.shape[0]
        img_labels = labels[idx:idx + num_img_des]
        img_rankings = ranking[idx:idx + num_img_des]
        clustered_data.append((img_labels, img_rankings, path_to_file, des))
        idx += num_img_des
