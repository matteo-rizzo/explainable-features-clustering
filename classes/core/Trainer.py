import sys
from typing import Dict, Tuple

from tqdm import tqdm

from classes.core.Evaluator import Evaluator
from classes.models.ModelCNN import ModelCNN
from functional.setup import get_device
from settings import OPTIMIZER, LEARNING_RATE, EPOCHS, CRITERION


class Trainer:

    def __init__(self):

        self.__device = get_device()
        self.__epochs = EPOCHS

        self.model = ModelCNN(self.__device)
        self.model.set_optimizer(OPTIMIZER, LEARNING_RATE)
        self.model.set_criterion(CRITERION)

        self.evaluator = Evaluator(self.__device, num_classes=10)

    running_loss, running_accuracy = 0.0, 0.0

    def train_one_epoch(self, epoch, training_loader):
        print(f"\n *** Epoch {epoch + 1}/{self.__epochs} *** ")

        self.model.train_mode()
        running_loss, running_accuracy = 0.0, 0.0

        tqdm_bar = tqdm(training_loader, total=len(training_loader), unit="batch", file=sys.stdout)
        tqdm_bar.set_description_str(" Training  ")

        for i, (x, y, _) in enumerate(training_loader):
            tqdm_bar.update(1)

            self.model.reset_gradient()

            y = y.long().to(self.__device)
            o = self.model.predict(x).to(self.__device)

            running_loss += self.model.update_weights(o, y)
            running_accuracy += Evaluator.batch_accuracy(o, y)

            progress: str = f"[ Loss: {running_loss:.4f} | Batch accuracy: {running_accuracy:.4f} ]"
            tqdm_bar.set_postfix_str(progress)

        tqdm_bar.close()
        print(" ...........................................................")

    def train(self, data: Dict) -> Tuple:
        """
        Trains the model according to the established parameters and the given data
        :param data: a dictionary of data loaders containing train, val and test data
        :return: the evaluation metrics of the training and the trained model
        """
        print("\n Training the model...")

        self.model.print_model_overview()

        evaluations = []
        training_loader = data["train"]

        for epoch in range(self.__epochs):
            self.train_one_epoch(epoch, training_loader)
            evaluations += [self.evaluator.evaluate(data, self.model)]

        print("\n Finished training!")
        print("----------------------------------------------------------------")
        return self.model, evaluations
