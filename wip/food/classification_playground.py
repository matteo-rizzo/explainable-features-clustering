import logging

import torch
import torchmetrics
import yaml
from torchmetrics import MetricCollection
from tqdm import tqdm

from classes.core.Trainer import Trainer
from classes.data.Food101Dataset import Food101Dataset
from classes.deep_learning.architectures.ConvNextWrapper import ConvNextWrapper
from classes.factories.CriterionFactory import CriterionFactory
from classes.factories.OptimizerFactory import OptimizerFactory
from functional.data_utils import create_stratified_splits
from functional.torch_utils import get_device


def main():
    logger = logging.getLogger(__name__)

    with open('config/training/training_configuration.yaml', 'r') as f:
        train_config = yaml.safe_load(f)
    with open('config/training/hypeparameter_configuration.yaml', 'r') as f:
        hyp = yaml.safe_load(f)

    # train = torch.utils.data.DataLoader(MNISTDataset(train=True),
    #                                     batch_size=train_config["batch_size"],
    #                                     shuffle=True,
    #                                     num_workers=train_config["workers"])
    # test = torch.utils.data.DataLoader(MNISTDataset(train=False),
    #                                    batch_size=train_config["batch_size"],
    #                                    shuffle=True,
    #                                    num_workers=train_config["workers"])

    train = torch.utils.data.DataLoader(Food101Dataset(train=True, augment=True),
                                        batch_size=train_config["batch_size"],
                                        shuffle=True,
                                        num_workers=train_config["workers"],
                                        drop_last=True)
    test = torch.utils.data.DataLoader(Food101Dataset(train=False),
                                       batch_size=train_config["batch_size"],
                                       shuffle=True,
                                       num_workers=train_config["workers"],
                                       drop_last=True)

    # train_subset, test_subset = create_stratified_splits(Food101Dataset(train=True, augment=False),
    #                                                      n_splits=1,
    #                                                      train_size=4040,
    #                                                      test_size=1010, )
    # train = torch.utils.data.DataLoader(train_subset,
    #                                     batch_size=train_config["batch_size"],
    #                                     shuffle=True,
    #                                     num_workers=train_config["workers"],
    #                                     drop_last=True)
    # test = torch.utils.data.DataLoader(test_subset,
    #                                    batch_size=train_config["batch_size"],
    #                                    shuffle=False,
    #                                    num_workers=train_config["workers"],
    #                                    drop_last=True)

    metric_collection = MetricCollection({
        'accuracy': torchmetrics.Accuracy(task="multiclass",
                                          num_classes=train_config["num_classes"]),
        # 'micro_precision': torchmetrics.Precision(task="multiclass",
        #                                           num_classes=train_config["num_classes"],
        #                                           average="micro"),
        # 'micro_recall': torchmetrics.Recall(task="multiclass",
        #                                     num_classes=train_config["num_classes"],
        #                                     average="micro"),
        # "micro_F1": torchmetrics.F1Score(task="multiclass",
        #                                  num_classes=train_config["num_classes"],
        #                                  average="micro")
    })

    trainer = Trainer(ConvNextWrapper, config=train_config, hyperparameters=hyp,
                      metric_collection=metric_collection, logger=logger)
    trainer.train(train, test)


PLOT = False
NUM_WORDS = 10
DEVICE_TYPE = "cuda:0"
OPTIMIZER = "AdamW"
LEARNING_RATE = 0.0001
CRITERION = "CrossEntropyLoss"
EPOCHS = 5
from torchvision.transforms import ToTensor, Resize, Compose, CenterCrop, Normalize

def get_accuracy(logits, gt, total: int, correct: int):
    _, predicted = torch.max(logits.data, 1)
    total += gt.size(0)
    correct += (predicted == gt).sum().item()
    return total, correct

def simple_for():
    with open('config/training/training_configuration.yaml', 'r') as f:
        train_config = yaml.safe_load(f)
    with open('config/training/hypeparameter_configuration.yaml', 'r') as f:
        hyp = yaml.safe_load(f)

    train = torch.utils.data.DataLoader(Food101Dataset(train=True, augment=True),
                                        batch_size=train_config["batch_size"],
                                        shuffle=True,
                                        num_workers=train_config["workers"],
                                        drop_last=True)
    test = torch.utils.data.DataLoader(Food101Dataset(train=False),
                                       batch_size=train_config["batch_size"],
                                       shuffle=True,
                                       num_workers=train_config["workers"],
                                       drop_last=True)

    device = get_device(DEVICE_TYPE)

    model = ConvNextWrapper(train_config).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    for epoch in range(EPOCHS):

        running_loss, correct, total = 0.0, 0, 0
        for i, (x, y) in tqdm(enumerate(train), desc="Training epoch: {}".format(epoch), total=len(train)):
            optimizer.zero_grad()

            x, y = x.to(device), y.to(device)
            # x = norm(x)
            o = model(x).to(device)
            # print(o.shape)
            loss = criterion(o, y)
            loss.backward()
            optimizer.step()
            loss = loss.item()
            # print(loss)
            running_loss += loss
            total, correct = get_accuracy(o, y, total, correct)

        train_loss, train_accuracy = running_loss / len(train), (correct / total)

        running_loss, correct, total = 0.0, 0, 0
        for i, (x, y) in tqdm(enumerate(test), desc="Testing epoch: {}".format(epoch)):
            x, y = x.to(device), y.to(device)
            o = model(x).to(device)
            # results = metric_collection(nn.functional.softmax(o), y)
            # print(results)
            # loss = criterion(o, y).item()
            # running_loss += loss
            total, correct = get_accuracy(o, y, total, correct)

        test_loss, test_accuracy = running_loss / len(test), (correct / total)

        print(f'Epoch [{epoch + 1:d}], '
              f'train loss: {train_loss:.3f}, '
              f'train accuracy: {train_accuracy:.3f}, '
              f'test loss: {test_loss:.3f}, '
              f'test accuracy: {test_accuracy:.3f}')


if __name__ == "__main__":
    # main()
    simple_for()
