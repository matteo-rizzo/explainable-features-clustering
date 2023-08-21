import torch
import yaml
from torch.utils.data import Dataset
from tqdm import tqdm

from classes.data.MaskHeatmapPetDataset import MaskHeatmapPetDataset
from functional.utilities.utils import default_logger
from functional.utilities.cluster_utilities import prepare_clusters_and_features
from visualization.image_visualization import draw_9x9_activation, draw_1x1_activations


def main():
    # --- Config ---
    with open('config/training/training_configuration.yaml', 'r') as f:
        config: dict = yaml.safe_load(f)
    with open('config/clustering/clustering_params.yaml', 'r') as f:
        clustering_config: dict = yaml.safe_load(f)
    with open('config/training/hypeparameter_configuration.yaml', 'r') as f:
        hyperparameters: dict = yaml.safe_load(f)
    # --- Logger ---
    logger = default_logger(config["logger"])
    # --- TRAIN DS ---
    clusterer_train, descriptors_train, keypoints_train = prepare_clusters_and_features(config, clustering_config,
                                                                                        logger, train=True)
    # train_ds = HeatmapPetDataset(keypoints, descriptors, clusterer, train=True)
    train_ds = MaskHeatmapPetDataset(keypoints_train, descriptors_train, clusterer_train, train=True)

    loader_ds = torch.utils.data.DataLoader(train_ds,
                                            batch_size=5,
                                            shuffle=False,
                                            num_workers=config["workers"],
                                            drop_last=False)

    for heatmap, label in tqdm(loader_ds, total=len(train_ds), desc=f"Drawing masks"):
        draw_1x1_activations(heatmap)
        break


if __name__ == "__main__":
    main()
