import matplotlib.pyplot as plt
import numpy as np
import torchmetrics
import yaml
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from torchvision import datasets, transforms

from classes.core.Trainer import Trainer
from classes.deep_learning.CNNs.ConvNextWrapper import ConvNextWrapper
from classes.deep_learning.CNNs.SmarterCNN import SmarterCNN
# from classes.deep_learning.architectures.modules.ExponentialMovingAverage import ExponentialMovingAverageModel
from functional.utilities.utils import default_logger

from torchvision import transforms as T

def test_trainer():
    # --- Config ---
    with open('config/other/dtd_simple_trainer_config.yaml', 'r') as f:
        config: dict = yaml.safe_load(f)
    with open('config/training/hypeparameter_configuration.yaml', 'r') as f:
        hyperparameters: dict = yaml.safe_load(f)
    # --- Logger ---
    logger = default_logger(config["logger"])
    # Define the transformation to apply to the data
    transform = transforms.Compose([
        T.ToTensor(),
        T.Resize(224, antialias=True),
        T.CenterCrop((224, 224)),
        # T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        # T.Grayscale()cre
    ])
    # --- TRAIN DS ---
    # Download and load the MNIST training dataset
    train_dataset = datasets.DTD(root="dataset", split="train", transform=transform, download=True)
    # Create a data loader for the training dataset
    train_loader_ds = DataLoader(train_dataset, batch_size=32, shuffle=True)

    for batch_imgs, batch_labels in train_loader_ds:
        for idx, (img, label) in enumerate(zip(batch_imgs, batch_labels)):
            # Transpose image data for plt.imshow
            img = np.transpose(img, (1, 2, 0))
            # Display the image
            plt.imshow(img)
            # Add label as title
            plt.title(f"Label: {train_dataset.classes[label]}")
            # Display the plot
            plt.show()
            if idx == 10:
                break
        break
    # --- VAL DS ---
    # Download and load the MNIST test dataset
    val_dataset = datasets.DTD(root="dataset", split="val", transform=transform, download=True)
    # Create a data loader for the test dataset
    val_loader_ds = DataLoader(val_dataset, batch_size=32, shuffle=False)
    # --- TEST DS ---
    # Download and load the MNIST test dataset
    test_dataset = datasets.DTD(root="dataset", split="test", transform=transform, download=True)
    # Create a data loader for the test dataset
    test_loader_ds = DataLoader(test_dataset, batch_size=32, shuffle=False)
    # --- Metrics for training ---
    metric_collection = MetricCollection({
        'accuracy': torchmetrics.Accuracy(task="multiclass",
                                          num_classes=config["num_classes"]),
        'macro_F1': torchmetrics.F1Score(task="multiclass",
                                         average="micro",
                                         num_classes=config["num_classes"]),
        'micro_F1': torchmetrics.F1Score(task="multiclass",
                                         average="micro",
                                         num_classes=config["num_classes"]),
    })
    # # # --- Training ---
    trainer = Trainer(ConvNextWrapper,
                      config=config,
                      hyperparameters=hyperparameters,
                      metric_collection=metric_collection,
                      logger=logger)
    trainer.train(train_dataloader=train_loader_ds,
                  val_dataloader=val_loader_ds,
                  test_dataloader=test_loader_ds)


if __name__ == "__main__":
    test_trainer()
