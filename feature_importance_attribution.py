from torch.utils.data import DataLoader

from classes.Clusterizer import Clusterizer
from classes.FeatureDescriptor import FeatureDescriptor
from classes.KeyPointsExtractor import KeyPointsExtractor
from classes.Plotter import Plotter
from classes.dataset.MNISTDataset import MNISTDataset


def main():
    dataset = MNISTDataset()
    dataloader = DataLoader(dataset)

    key_points_extractor = KeyPointsExtractor()
    feature_descriptor = FeatureDescriptor()
    data, features = [], []

    for (x, _, _) in dataloader:
        data.append(key_points_extractor.extract(x))
        features.append(feature_descriptor.describe(x))

    clusterizer = Clusterizer()
    clusters = clusterizer.cluster(features)
    ranking = clusterizer.rank_clusters(clusters)

    plotter = Plotter()
    plotter.plot_clusters(ranking[:10])

    ranked_features = clusterizer.rank_features(features, ranking)
    for key_points, (_, importance) in zip(data, ranked_features):
        plotter.plot_importance(key_points, importance)


if __name__ == "__main__":
    main()
