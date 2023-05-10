import cv2
import hdbscan
import numpy as np
import torchmetrics
import umap
import yaml
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from tqdm import tqdm
import seaborn as sns

from classes.FeautureExtractingAlgorithm import FeautureExtractingAlgorithm
from classes.MNISTDataset import MNISTDataset


def reducer(descriptors):
    _reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.0,
        n_components=2,
        random_state=42069,
        metric="cosine"
    )
    # print("Running umap...")
    # reduced_vectors = _reducer.fit_transform(latent_vectors)
    # Create a progress bar using tqdm
    with tqdm(total=1, desc="Running umap...", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') as pbar:
        # Fit UMAP and update progress bar
        _reducer.fit(descriptors)
        pbar.update(1)

    reduced_vectors = _reducer.transform(descriptors)
    # -- Clustering ---
    cluster_labels = run_hdbscan(descriptors)
    plot(cluster_labels, reduced_vectors)

    cluster_labels = run_hac(descriptors)
    plot(cluster_labels, reduced_vectors)


def plot(cluster_labels, reduced_vectors):
    # Set the seaborn style to "darkgrid"
    sns.set(style="darkgrid")

    clustered = (cluster_labels >= 0)
    plt.figure(figsize=(10, 10), dpi=200)
    plt.scatter(reduced_vectors[~clustered, 0],
                reduced_vectors[~clustered, 1],
                color=(0.5, 0.5, 0.5),
                s=0.2,
                alpha=0.5)

    plt.scatter(reduced_vectors[clustered, 0],
                reduced_vectors[clustered, 1],
                c=cluster_labels[clustered],
                s=0.2,
                cmap="Spectral")

    # labels, centroids = clustering.get_labels(), clustering.get_centroids()
    #
    # Plot the data points and centroids
    # plt.scatter(flat_descriptors[:, 0], flat_descriptors[:, 1], c=labels)
    # plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=100, linewidths=3, c='k')

    plt.show()


def run_hdbscan(descriptors):
    print("Running HDBSCAN...")
    # -- Clustering ---
    clusterer = hdbscan.HDBSCAN(min_cluster_size=10,
                                min_samples=3,
                                cluster_selection_method="eom",
                                metric="euclidean")

    return clusterer.fit_predict(descriptors)


def run_hac(descriptors):
    print("Running HAC...")

    # Create and fit Agglomerative Clustering
    clusterer = AgglomerativeClustering(n_clusters=10,
                                        metric='euclidean',
                                        # linkage="average"
                                        # linkage='ward',
                                        # distance_threshold=2.0
                                        )
    return clusterer.fit_predict(descriptors)


def main():
    with open('config/training/training_configuration.yaml', 'r') as f:
        config = yaml.safe_load(f)

    train_loader = DataLoader(MNISTDataset(train=True),
                              batch_size=config["batch_size"],
                              shuffle=False,
                              num_workers=config["workers"])
    test_loader = DataLoader(MNISTDataset(train=False),
                             batch_size=config["batch_size"],
                             shuffle=False,
                             num_workers=config["workers"])
    # -----------------------------------------------------------------------------------
    # TODO: genetic algorithm to maximise these features?
    # -----------------------------------------------------------------------------------
    key_points_extractor = FeautureExtractingAlgorithm(algorithm="SIFT")
    keypoints, descriptors = key_points_extractor.get_keypoints_and_descriptors(train_loader)

    # TODO HDBScan
    # clustering = KMeansClustering(n_clusters=10)

    flat_descriptors = np.concatenate(descriptors)
    reducer(flat_descriptors)

    # clustering.fit(flat_descriptors)
    # labels, centroids = clustering.get_labels(), clustering.get_centroids()

    # Plot the data points and centroids
    # plt.scatter(flat_descriptors[:, 0], flat_descriptors[:, 1], c=labels)
    # plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=100, linewidths=3, c='k')
    # plt.show()


def show_4():
    with open('config/training/training_configuration.yaml', 'r') as f:
        config = yaml.safe_load(f)
    with open('config/training/hypeparameter_configuration.yaml', 'r') as f:
        hyp = yaml.safe_load(f)
    metric_collection = MetricCollection({
        'accuracy': torchmetrics.Accuracy(task="multiclass", num_classes=10),
        'precision': torchmetrics.Precision(task="multiclass", num_classes=10, average="macro"),
        'recall': torchmetrics.Recall(task="multiclass", num_classes=10, average="macro"),
        "F1": torchmetrics.F1Score(task="multiclass", num_classes=10, average="macro")
    })

    train_loader = DataLoader(MNISTDataset(train=True),
                              batch_size=config["batch_size"],
                              shuffle=False,
                              num_workers=config["workers"])
    key_points_extractor_1 = FeautureExtractingAlgorithm(nfeatures=200,
                                                         # (default = 0 = all) Small images, few features
                                                         nOctaveLayers=3,
                                                         # (default = 3) Default should be ok
                                                         contrastThreshold=0.04,
                                                         # (default = 0.04) Lower = Include kps with lower contrast
                                                         edgeThreshold=10,
                                                         # (default = 10) Higher = Include KPS with lower edge response
                                                         sigma=1.2)  # (default = 1.2) capture finer details in imgs
    key_points_extractor_2 = FeautureExtractingAlgorithm(nfeatures=200,
                                                         # (default = 0 = all) Small images, few features
                                                         nOctaveLayers=3,
                                                         # (default = 3) Default should be ok
                                                         contrastThreshold=0.04,
                                                         # (default = 0.04) Lower = Include kps with lower contrast
                                                         edgeThreshold=10,
                                                         # (default = 10) Higher = Include KPS with lower edge response
                                                         sigma=0.75)  # (default = 1.2) capture finer details in imgs

    key_points_extractor_3 = FeautureExtractingAlgorithm(nfeatures=200,
                                                         # (default = 0 = all) Small images, few features
                                                         nOctaveLayers=3,
                                                         # (default = 3) Default should be ok
                                                         contrastThreshold=0.04,
                                                         # (default = 0.04) Lower = Include kps with lower contrast
                                                         edgeThreshold=20,
                                                         # (default = 10) Higher = Include KPS with lower edge response
                                                         sigma=1.2)  # (default = 1.2) capture finer details in imgs

    key_points_extractor_4 = FeautureExtractingAlgorithm(nfeatures=200,
                                                         # (default = 0 = all) Small images, few features
                                                         nOctaveLayers=3,
                                                         # (default = 3) Default should be ok
                                                         contrastThreshold=0.04,
                                                         # (default = 0.04) Lower = Include kps with lower contrast
                                                         edgeThreshold=20,
                                                         # (default = 10) Higher = Include KPS with lower edge response
                                                         sigma=0.75)  # (default = 1.2) capture finer details in imgs        for imgs, _ in train_loader:
    for imgs, _ in train_loader:
        for img in imgs:
            img = (img.numpy().squeeze() * 255).astype(np.uint8)

            kp1, _ = key_points_extractor_1.run(img)
            kp2, _ = key_points_extractor_2.run(img)
            kp3, _ = key_points_extractor_3.run(img)
            kp4, _ = key_points_extractor_4.run(img)

            keypoints = [kp1, kp2, kp3, kp4]

            # Create a blank canvas to display the images
            canvas = np.zeros((28 * 2, 28 * 2, 3), dtype=np.uint8)

            # Loop through each digit and its keypoints
            for i in range(4):
                # Convert the image to uint8 and resize it
                # img = (digits[i].numpy().squeeze() * 255).astype(np.uint8)
                # img = cv2.resize(img, (28, 28))

                # Draw keypoints on the image
                img_kp = cv2.drawKeypoints(img, keypoints[i], None,
                                           flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

                # Compute the coordinates for placing the image on the canvas
                x = (i % 2) * 28
                y = (i // 2) * 28

                # Place the image with keypoints on the canvas
                canvas[y:y + 28, x:x + 28] = img_kp

            cv2.namedWindow('Canvas', cv2.WINDOW_NORMAL)
            scale = 4  # Adjust this to change the size of the canvas
            resized_canvas = cv2.resize(canvas, (canvas.shape[1] * scale, canvas.shape[0] * scale))
            cv2.imshow('Canvas', resized_canvas)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
    # show_4()
