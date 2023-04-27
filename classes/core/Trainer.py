from pathlib import Path
from typing import Optional

import torch
import yaml

from functional.setup import get_device
from functional.utils import intersect_dicts


# from classifiers.deep_learning.classes.core.Evaluator import Evaluator
# from classifiers.deep_learning.classes.factories.ModelFactory import ModelFactory
# from classifiers.deep_learning.classes.utils.Params import Params


class Trainer:

    def __init__(self, config: dict, hyperparameters: dict, logger):
        self.config = config
        self.hyperparameters = hyperparameters
        self.logger = logger
        # --- Training stuff ---
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.ema = None  # Exponential Moving Average
        # ---
        # self.compute_loss_ota = None
        # self.compute_loss = None
        # ---
        self.device: Optional[torch.device] = None
        # ---
        self.model = None
        self.checkpoint = None

        # self.path_to_best_model = path_to_best_model
        #
        # self.device = config["device"]
        # self.epochs = config["epochs"]
        # # self.optimizer_type = config["optimizer"]
        # self.lr = config["learning_rate"]
        #
        # self.log_every = config["log_every"]
        # self.evaluate_every = config["evaluate_every"]
        #
        # self.patience = config["early_stopping"]["patience"]
        # self.es_metric = config["early_stopping"]["metrics"]
        # self.es_metric_trend = config["early_stopping"]["metrics_trend"]
        # self.es_metric_best_value = 0.0 if self.__es_metric_trend == "increasing" else 1000
        # self.epochs_no_improvement = 0
        #
        # # network_type, criterion_type = train_params["network_type"], train_params["criterion"]
        #
        # network_params = Params.load_network_params(network_type)
        # network_params["device"] = self.__device
        #
        # self.model = ModelFactory().get(network_type, network_params)
        # self.model.set_optimizer(self.__optimizer_type, self.__lr)
        # self.model.set_criterion(criterion_type)
        #
        # self.evaluator = Evaluator(self.__device, config["num_classes"]

    def train_one_epoch(self, epoch, training_loader):
        pass
        # print(f"\n *** Epoch {epoch + 1}/{self.__epochs} *** ")
        #
        # self.model.train_mode()
        # running_loss, running_accuracy = 0.0, 0.0
        #
        # # Visual progress bar
        # tqdm_bar = tqdm(training_loader, total=len(training_loader), unit="batch", file=sys.stdout)
        # tqdm_bar.set_description_str(" Training  ")
        #
        # # Process batches
        # for i, (x, y) in enumerate(training_loader):
        #     tqdm_bar.update(1)
        #     # Zero gradients
        #     self.model.reset_gradient()
        #
        #     # Forward pass
        #     y = y.long().to(self.__device)
        #     o = self.model.predict(x).to(self.__device)
        #
        #     # Loss, backward pass, step
        #     running_loss += self.model.update_weights(o, y)
        #     running_accuracy += Evaluator.batch_accuracy(o, y)
        #
        #     # Log current epoch result
        #     if not (i + 1) % self.__log_every:
        #         avg_loss, avg_accuracy = running_loss / self.__log_every, running_accuracy / self.__log_every
        #         running_loss, running_accuracy = 0.0, 0.0
        #         # Update progress bar
        #         progress: str = f"[ Loss: {avg_loss:.4f} | Batch accuracy: {avg_accuracy:.4f} ]"
        #         tqdm_bar.set_postfix_str(progress)
        # # Close progress bar for this epoch
        # tqdm_bar.close()
        # print(" ...........................................................")

    def __init_dump_folder(self):
        save_dir: Path = Path(self.config["save_dir"])
        weights_dir: Path = save_dir / 'weights'
        weights_dir.mkdir(parents=True, exist_ok=True)  # make dir
        last_ckpt: Path = weights_dir / 'last.pt'
        best_ckpt: Path = weights_dir / 'best.pt'
        results_file: Path = save_dir / 'results.txt'

        # --- Save run settings ---
        with open(save_dir / 'hyp.yaml', 'w') as f:
            yaml.dump(self.hyperparameters, f, sort_keys=False)
        with open(save_dir / 'opt.yaml', 'w') as f:
            yaml.dump(vars(self.config), f, sort_keys=False)
        return save_dir, weights_dir, last_ckpt, best_ckpt, results_file

    def __setup_model(self, pretrained, class_number):
        if pretrained:
            self.checkpoint = torch.load(self.config["weights"], map_location=self.device)  # load checkpoint
            self.model = Model(self.config["cfg"] or self.checkpoint['model'].yaml,
                               ch=3,
                               nc=class_number,
                               anchors=self.hyperparameters.get('anchors'),
                               logger=self.logger).to(self.device)
            state_dict = self.checkpoint['model'].float().state_dict()  # to FP32
            state_dict = intersect_dicts(state_dict, self.model.state_dict(), exclude=[])  # intersect
            self.model.load_state_dict(state_dict, strict=False)  # load
            self.logger.info(
                f'Transferred {len(state_dict):g}/{len(self.model.state_dict()):g} '
                f'items from {self.config["weights"]}')  # report
        else:
            self.model = Model(self.config["cfg"],
                               ch=3,
                               nc=class_number,
                               anchors=self.hyperparameters.get('anchors'),
                               logger=self.logger).to(self.device)

    def train(self):
        # --- Directories, initialize where to save things ---
        save_dir, weights_dir, last_ckpt, best_ckpt, results_file = self.__init_dump_folder()

        self.device = get_device(self.config["device"])
        # --- Model ---
        pretrained: bool = self.config["weights"].endswith('.pt')
        self.__setup_model(pretrained=pretrained, class_number=self.config["class_number"])

        pass
        # """
        # Trains the model according to the established parameters and the given data
        # :param data: a dictionary of data loaders containing train, val and test data
        # :return: the evaluation metrics of the training and the trained model
        # """
        # print("\n Training the model...")
        #
        # self.model.print_model_overview()
        #
        # evaluations = []
        # training_loader = data["train"]
        #
        # # --- Single epoch ---
        # for epoch in range(self.__epochs):
        #     # Train epoch
        #     self.train_one_epoch(epoch, training_loader)
        #
        #     # Perform evaluation
        #     if not (epoch + 1) % self.__evaluate_every:
        #         evaluations += [self.evaluator.evaluate(data, self.model)]
        #         if self.__early_stopping_check(evaluations[-1]["metrics"]["val"][self.__es_metric]):
        #             break
        #
        # print("\n Finished training!")
        # print("----------------------------------------------------------------")
        # return self.model, evaluations

    def __early_stopping_check(self, metric_value: float) -> bool:
        pass
        # """
        # Decides whether to early stop the train based on the early stopping conditions
        # @param metric_value: the monitored val metrics (e.g. auc, loss)
        # @return: a flag indicating whether the training should be early stopped
        # """
        # if self.__es_metric_trend == "increasing":
        #     metrics_check = metric_value > self.__es_metric_best_value
        # else:
        #     metrics_check = metric_value < self.__es_metric_best_value
        #
        # if metrics_check:
        #     print(f"\n\t Old best val {self.__es_metric}: {self.__es_metric_best_value:.4f} "
        #           f"| New best {self.__es_metric}: {metric_value:.4f}\n")
        #
        #     print("\t Saving new best model...")
        #     self.model.save(self.__path_to_best_model)
        #     print("\t -> New best model saved!")
        #
        #     self.__es_metric_best_value = metric_value
        #     self.__epochs_no_improvement = 0
        # else:
        #     self.__epochs_no_improvement += 1
        #     if self.__epochs_no_improvement == self.__patience:
        #         print(f" ** No decrease in val {self.__es_metric} "
        #               f"for {self.__patience} evaluations. Early stopping! ** ")
        #         return True
        #
        # print(" Epochs without improvement: ", self.__epochs_no_improvement)
        # print(" ........................................................... ")
        # return False
