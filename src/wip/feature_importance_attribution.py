import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.classes.feature_extraction.FeatureExtractingAlgorithm import FeatureExtractingAlgorithm
from __to_delete.KMeansClustering import KMeansClustering
from src.classes.data.MNISTDataset import MNISTDataset
from __to_delete.models.ModelImportanceWeightedCNN import ModelImportanceWeightedCNN
from src.functional.torch_utils import get_device

DEVICE_TYPE = "cpu"
OPTIMIZER = "sgd"
LEARNING_RATE = 0.01
CRITERION = "CrossEntropyLoss"
EPOCHS = 15


def main():
    train_loader = DataLoader(MNISTDataset(train=True),
                              batch_size=64, shuffle=True, num_workers=2)
    test_loader = DataLoader(MNISTDataset(train=False),
                             batch_size=64, shuffle=False, num_workers=2)

    sift = FeatureExtractingAlgorithm()
    _, descriptors = sift.get_keypoints_and_descriptors(train_loader)
    flat_descriptors = np.concatenate(descriptors)

    clustering = KMeansClustering()
    clusters = clustering.fit(flat_descriptors)
    labels, centroids = clusters.labels_, clusters.cluster_centers_
    clustering.plot_sample(flat_descriptors, centroids, labels, sample_size=400000)
    ranking = clustering.rank_clusters(flat_descriptors, centroids, labels)

    device = get_device(DEVICE_TYPE)

    model = ModelImportanceWeightedCNN(device)
    model.set_optimizer(OPTIMIZER, LEARNING_RATE)
    model.set_criterion(CRITERION)
    model.train_mode()

    for epoch in range(EPOCHS):

        running_loss, correct, total = 0.0, 0, 0
        for i, (x, y) in tqdm(enumerate(train_loader), desc="Training epoch: {}".format(epoch)):
            x, y = x.to(device), y.to(device)
            o = model.predict(x).to(device)
            loss = model.update_weights(o, y)
            running_loss += loss
            total, correct = model.get_accuracy(o, y, total, correct)

        train_loss, train_accuracy = running_loss / len(train_loader), 100 * correct / total

        running_loss, correct, total = 0.0, 0, 0
        for i, (x, y) in tqdm(enumerate(test_loader), desc="Testing epoch: {}".format(epoch)):
            x, y = x.to(device), y.to(device)
            o = model.predict(x).to(device)
            loss = model.get_loss(o, y)
            running_loss += loss
            total, correct = model.get_accuracy(o, y, total, correct)

        test_loss, test_accuracy = running_loss / len(test_loader), 100 * correct / total

        print(f'Epoch [{epoch + 1:d}], '
              f'train loss: {train_loss:.3f}, '
              f'train accuracy: {train_accuracy:.3f}, '
              f'test loss: {test_loss:.3f}, '
              f'test accuracy: {test_accuracy:.3f}')


if __name__ == "__main__":
    main()
