from torch.utils.data import DataLoader

from classes.Vocabulary import Vocabulary
from classes.clusterizers.Clusterizer import Clusterizer
from classes.core.Trainer import Trainer
from classes.dataset.MNISTDataset import MNISTDataset
from classes.feature_descriptors.FeatureDescriptor import FeatureDescriptor
from classes.key_points_extractors.KeyPointsExtractor import KeyPointsExtractor


def main():
    data = {
        "train": DataLoader(MNISTDataset(train=True)),
        "test": DataLoader(MNISTDataset(train=False))
    }

    key_points_extractor = KeyPointsExtractor()
    feature_descriptor = FeatureDescriptor()
    key_points, features = [], []

    for (x, _, _) in data["train"]:
        key_points.append(key_points_extractor.extract(x))
        features.append(feature_descriptor.describe(x))

    clusterizer = Clusterizer()
    clusters = clusterizer.cluster(features)
    ranking = clusterizer.rank_clusters(clusters)

    vocabulary = Vocabulary()
    vocabulary.build(ranking)

    Trainer().train(data)


if __name__ == "__main__":
    main()
