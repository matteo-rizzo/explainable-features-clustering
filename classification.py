import numpy as np
from classes.Clustering import KMeansClustering
from classes.core.Trainer import Trainer
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader

from classes.MNISTDataset import MNISTDataset
from classes.SIFT import SIFT
from classes.Vocabulary import Vocabulary


def main():
    data = {
        "train": DataLoader(MNISTDataset(train=True)),
        "test": DataLoader(MNISTDataset(train=False))
    }

    key_points_extractor = SIFT()
    key_points, descriptors = [], []

    for (x, _, _) in data["train"]:
        kp, des = key_points_extractor.run(x)
        key_points.append(kp)
        descriptors.append(des)

    clusterizer = KMeansClustering()
    clusters = clusterizer.run(descriptors)
    ranking = clusterizer.rank_clusters(clusters)

    vocabulary = Vocabulary()
    X, y = vocabulary.embed(ranking)

    svm = LinearSVC()

    # train the machine learning model on the feature matrix and label vector
    svm.fit(X, y)

    # predict the labels of the training data
    y_pred = svm.predict(X)

    # evaluate the accuracy of the model
    acc = accuracy_score(y, y_pred)

    # compute the importance weights of the visual words using the learned coefficients
    importance_weights = np.abs(svm.coef_).sum(axis=0)

    # match the visual words with the features using dot product weighted by importance weights
    matches = Vocabulary.match(descriptors, importance_weights)

    Trainer().train(data)


if __name__ == "__main__":
    main()
