import logging

import numpy as np
import torchmetrics
import yaml
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection

from classes.Clustering import KMeansClustering
from classes.MNISTDataset import MNISTDataset
from classes.SIFT import SIFT
from classes.Vocabulary import Vocabulary
from classes.core.Trainer import Trainer
from classes.deep_learning.architectures.ImportanceWeightedCNN import ImportanceWeightedCNN


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

    train_data = DataLoader(MNISTDataset(train=True))
    test_data = DataLoader(MNISTDataset(train=False))
    # -----------------------------------------------------------------------------------
    key_points_extractor = SIFT()
    key_points, descriptors = [], []

    for (imgs, _) in train_data:
        kp, des = key_points_extractor.run(imgs)
        key_points.append(kp)
        descriptors.append(des)

    clusterizer = KMeansClustering()
    clusters = clusterizer.run(descriptors)
    ranking = clusterizer.rank_clusters(clusters)

    vocabulary = Vocabulary()
    X, y = vocabulary.embed(ranking)
    # -----------------------------------------------------------------------------------
    svm = LinearSVC()
    # train the machine learning model on the feature matrix and label vector
    svm.fit(X, y)

    # predict the labels of the training data
    y_pred = svm.predict(X)

    # evaluate the accuracy of the model
    acc = accuracy_score(y, y_pred)

    # compute the importance weights of the visual words using the learned coefficients
    importance_weights = np.abs(svm.coef_).sum(axis=0)
    # -----------------------------------------------------------------------------------
    # match the visual words with the features using dot product weighted by importance weights
    matches = Vocabulary.match(descriptors, importance_weights)
    # -----------------------------------------------------------------------------------
    trainer = Trainer(ImportanceWeightedCNN, config=config, hyperparameters=hyp,
                      metric_collection=metric_collection, logger=logger)
    trainer.train(train_dataloader=train_data, test_dataloader=test_data)
    # -----------------------------------------------------------------------------------


if __name__ == "__main__":
    main()
