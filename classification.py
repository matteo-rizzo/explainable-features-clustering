import logging

import cv2
import numpy as np
import torchmetrics
import yaml
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection

from classes.Clustering import KMeansClustering
from classes.MNISTDataset import MNISTDataset
from classes.SIFT import SIFT


def main():
    # -----------------------------------------------------------------------------------
    logger = logging.getLogger(__name__)
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
    test_loader = DataLoader(MNISTDataset(train=False),
                             batch_size=config["batch_size"],
                             shuffle=False,
                             num_workers=config["workers"])
    # -----------------------------------------------------------------------------------
    # TODO: genetic algorithm to maximise these features?
    key_points_extractor = SIFT(nfeatures=150,  # (default = 0 = all) Small images, few features
                                nOctaveLayers=3,  # (default = 3) Default should be ok
                                contrastThreshold=0.04,  # (default = 0.04) Lower = Include kps with lower contrast
                                edgeThreshold=20,  # (default = 10) Higher = Include keypoints with lower edge response
                                sigma=1.1)  # (default = 1.2) capture finer details in the images
    keypoints, descriptors = key_points_extractor.get_keypoints_and_descriptors(train_loader)

    for imgs, _ in train_loader:
        for img in imgs:
            img = (img.numpy().squeeze() * 255).astype(np.uint8)
            kp, _ = key_points_extractor.run(img)
            img_kp = cv2.drawKeypoints(img,
                                       kp,
                                       None,
                                       flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            # Display image
            cv2.namedWindow('SIFT Image', cv2.WINDOW_NORMAL)

            # Display image
            scale = 4  # Adjust this to change the size of the image
            resized_img = cv2.resize(img_kp, (img_kp.shape[1] * scale, img_kp.shape[0] * scale))
            cv2.imshow('SIFT Image', resized_img)
            # cv2.imshow('SIFT Keypoints', img_kp)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    # TODO HDBScan
    clustering = KMeansClustering()

    flat_descriptors = np.concatenate(descriptors)
    clusters = clustering.run(flat_descriptors)
    labels, centroids = clusters.labels_, clusters.cluster_centers_

    # TODO FROM HERE ON ------
    # clustering.plot_sample(flat_descriptors, centroids, labels, sample_size=400000)
    # ranking = clustering.rank_clusters(flat_descriptors, centroids, labels)
    # vocabulary = Vocabulary()
    # X, y = vocabulary.embed(ranking)
    # # -----------------------------------------------------------------------------------
    # svm = LinearSVC()
    # # train the machine learning model on the feature matrix and label vector
    # svm.fit(X, y)
    #
    # # predict the labels of the training data
    # y_pred = svm.predict(X)
    #
    # # evaluate the accuracy of the model
    # acc = accuracy_score(y, y_pred)
    #
    # # compute the importance weights of the visual words using the learned coefficients
    # importance_weights = np.abs(svm.coef_).sum(axis=0)
    # # -----------------------------------------------------------------------------------
    # # match the visual words with the features using dot product weighted by importance weights
    # matches = Vocabulary.match(descriptors, importance_weights)
    # # -----------------------------------------------------------------------------------
    # trainer = Trainer(ImportanceWeightedCNN, config=config, hyperparameters=hyp,
    #                   metric_collection=metric_collection, logger=logger)
    # trainer.train(train_dataloader=train_loader, test_dataloader=test_loader)
    # # -----------------------------------------------------------------------------------


if __name__ == "__main__":
    main()
