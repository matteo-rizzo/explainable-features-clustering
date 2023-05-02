import logging
import os
from pathlib import Path
from typing import Optional

import colorlog
import torch
import yaml
from torch.cuda import amp
from torch.utils.data import Dataset

from classes.deep_learning.architectures.CNN import CNN
from classes.deep_learning.architectures.TorchModel import TorchModel
from classes.deep_learning.architectures.modules.ExponentialMovingAverage import ExponentialMovingAverageModel
from functional.lr_schedulers import linear_lrs, one_cycle_lrs
from functional.setup import get_device
from functional.utils import intersect_dicts, increment_path, check_file_exists, get_latest_run, colorstr


# from classifiers.deep_learning.classes.core.Evaluator import Evaluator
# from classifiers.deep_learning.classes.factories.ModelFactory import ModelFactory
# from classifiers.deep_learning.classes.utils.Params import Params


class Trainer:

    def __init__(self, model_class: TorchModel, config: dict, hyperparameters: dict, logger):
        self.config = config
        self.hyperparameters = hyperparameters
        self.logger = logger
        # --- Training stuff ---
        self.optimizer = None
        self.gradient_scaler = None
        self.exponential_moving_average = None
        self.scheduler = None
        self.sca = None  # Exponential Moving Average
        # ---
        # self.compute_loss_ota = None
        # self.compute_loss = None
        # ---
        self.device: Optional[torch.device] = get_device(config["device"])
        # ---
        self.model_class = model_class
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

    def train(self, train_dataloader, val_dataloader):
        # --- Directories, initialize where to save things ---
        self.__start_or_resume_config()
        save_dir, weights_dir, last_ckpt, best_ckpt, results_file = self.__init_dump_folder()

        # --- Model ---
        pretrained: bool = self.config["weights"].endswith('.pt')
        self.__setup_model(pretrained=pretrained)

        # --- Gradient accumulation ---
        accumulate: int = self.__setup_gradient_accumulation()

        # --- Optimization ---
        # TODO: make optional / modularize
        self.optimizer: torch.optim.Optimizer = self.__setup_optimizer()
        self.scheduler: torch.optim.lr_scheduler = self.__setup_scheduler()

        # --- Exponential moving average ---
        self.exponential_moving_average = ExponentialMovingAverageModel(self.model)

        # --- Resume pretrained if necessary ---
        if pretrained:
            start_epoch, best_fitness = self.__resume_pretrained(results_file=results_file)
            self.checkpoint = None
        else:
            start_epoch, best_fitness = 0, 0.0

        # Grid size (max stride)
        # grid_size = max(int(self.model.stride.max()), 32)
        # Verify img_size are grid_size-multiples
        # img_size, img_size_test = [check_img_size(x, grid_size, logger=self.logger) for x in self.config["img_size"]]

        batch_number: int = len(train_dataloader)
        # Number of warmup iterations, max(3 epochs, 1k iterations)
        warmup_number: int = max(round(self.hyperparameters['warmup_epochs'] * batch_number), 1000)

        results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
        self.scheduler.last_epoch = start_epoch - 1  # do not move
        self.gradient_scaler = amp.GradScaler(enabled=self.device == "cuda:0")
        # self.compute_loss_ota = ComputeLossOTA(self.model)  # init losses class
        # self.compute_loss = ComputeLoss(self.model)  # init losses class
        self.logger.info(f'{colorstr("bright_green", "Image sizes")}: TRAIN [{self.config["img_size"][0]}] , '
                         f'TEST [{self.config["img_size"][1]}]\t'
                         f'{colorstr("bright_green", "Dataloader workers")}: {train_dataloader.num_workers}\t'
                         f'{colorstr("bright_green", "Saving results to")}: {save_dir}\n'
                         f'{" "*31}{colorstr("bright_green", "Starting training for")} {self.config["epochs"]} epochs...')

        torch.save(self.model, weights_dir / 'init.pt')
        epoch: int = -1

    def __start_or_resume_config(self):
        # --- Resume if a training had already started ---
        if self.config["resume"]:
            # Specified or most recent path
            checkpoint = self.config["resume"] if isinstance(self.config["resume"], str) else get_latest_run()
            assert os.path.isfile(checkpoint), 'ERROR: checkpoint does not exist'
            # Open the parameters of the resumed configuration
            with open(Path(checkpoint).parent.parent / 'opt.yaml') as f:
                self.config = yaml.safe_load(f)
            # Re-configure from loaded configuration
            self.config["architecture_config"] = ''
            self.config["weights"] = checkpoint
            self.config["resume"] = True
            self.logger.info(f'Resuming training from {checkpoint}')

        else:
            # Check files exist. The return is either the same path (if it was correct)
            # Or an updated path if it was found (uniquely) in the path's subdirectories
            self.config["data"] = check_file_exists(self.config["data"])
            self.config["architecture_config"] = check_file_exists(self.config["architecture_config"])
            self.config["hyperparameters"] = check_file_exists(self.config["hyperparameters"])
            assert len(self.config["architecture_config"]) or len(self.config["weights"]), \
                'either architecture_config or weights must be specified'
            # Increment run
            self.config["save_dir"] = increment_path(Path(self.config["project"]) / self.config["name"],
                                                     exist_ok=self.config["exist_ok"])
            self.logger.info(f'Starting a new training')

    def __init_dump_folder(self) -> [Path, Path, Path, Path, Path]:
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
            yaml.dump(self.config, f, sort_keys=False)
        return save_dir, weights_dir, last_ckpt, best_ckpt, results_file

    def __setup_model(self, pretrained: bool) -> None:
        if pretrained:
            self.checkpoint = torch.load(self.config["weights"], map_location=self.device)  # load checkpoint
            self.model = self.model_class(
                config_path=self.config["architecture_config"] or self.checkpoint['model'].yaml,
                logger=self.logger).to(self.device)
            state_dict = self.checkpoint['model'].float().state_dict()  # to FP32
            state_dict = intersect_dicts(state_dict, self.model.state_dict(), exclude=[])  # intersect
            self.model.load_state_dict(state_dict, strict=False)  # load
            self.logger.info(
                f'Transferred {len(state_dict):g}/{len(self.model.state_dict()):g} '
                f'items from {self.config["weights"]}')  # report
        else:
            self.model = self.model_class(config_path=self.config["architecture_config"], logger=self.logger).to(
                self.device)

    def __setup_gradient_accumulation(self) -> int:
        # If the total batch size is less than or equal to the nominal batch size, then accumulate is set to 1.
        # Accumulate losses before optimizing
        accumulate = max(round(self.config["nominal_batch_size"] / self.config["batch_size"]), 1)
        # Scale weight_decay
        self.hyperparameters['weight_decay'] *= (self.config["batch_size"] * accumulate
                                                 / self.config["nominal_batch_size"])
        self.logger.info(f"Scaled weight_decay = {self.hyperparameters['weight_decay']}")
        return accumulate

    def __setup_scheduler(self) -> torch.optim.lr_scheduler:
        # TODO: replace with scheduler factory
        if self.config["linear_lr"]:
            lr_schedule_fn = linear_lrs(steps=self.config["epochs"],
                                        lrf=self.hyperparameters['lrf'])
        else:
            lr_schedule_fn = one_cycle_lrs(y1=1, y2=self.hyperparameters['lrf'],
                                           steps=self.config["epochs"])  # cosine 1->hyp['lrf']
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_schedule_fn)

    def __setup_optimizer(self) -> torch.optim.Optimizer:
        # TODO: replace with optimizer factory
        if self.config["adam"]:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hyperparameters['lr0'],
                                         betas=(self.hyperparameters['momentum'], 0.999))  # adjust beta1 to momentum
        else:
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.hyperparameters['lr0'],
                                        momentum=self.hyperparameters['momentum'],
                                        nesterov=True)
        # TODO: look up different optimization for different parameter groups
        return optimizer

    def __resume_pretrained(self, results_file) -> [int, float]:
        # --- Optimizer ---
        best_fitness: float = 0.0
        if self.checkpoint['optimizer'] is not None:
            self.optimizer.load_state_dict(self.checkpoint['optimizer'])
            best_fitness = self.checkpoint['best_fitness']

        # --- EMA ---
        if self.exponential_moving_average and self.checkpoint.get('ema'):
            self.exponential_moving_average.ema_model.load_state_dict(self.checkpoint['ema'].float().state_dict())
            self.exponential_moving_average.updates = self.checkpoint['updates']

        # --- Results ---
        if self.checkpoint.get('training_results') is not None:
            results_file.write_text(self.checkpoint['training_results'])  # write results.txt

        # --- Epochs ---
        start_epoch = self.checkpoint['epoch'] + 1
        if self.config["resume"]:
            assert start_epoch > 0, f'{self.config["weights"]} training to ' \
                                    f'{self.config["epochs"]:g} epochs is finished, nothing to resume.'
        if self.config["epochs"] < start_epoch:
            self.logger.info(
                f'{self.config["weights"]} has been trained for {self.checkpoint["epoch"]:g} epochs. '
                f'Fine-tuning for {self.config["epochs"]:g} additional epochs.')
            self.config["epochs"] += self.checkpoint['epoch']  # finetune additional epoch
        return start_epoch, best_fitness

    # def __early_stopping_check(self, metric_value: float) -> bool:
    #     pass
    #     # """
    #     # Decides whether to early stop the train based on the early stopping conditions
    #     # @param metric_value: the monitored val metrics (e.g. auc, loss)
    #     # @return: a flag indicating whether the training should be early stopped
    #     # """
    #     # if self.__es_metric_trend == "increasing":
    #     #     metrics_check = metric_value > self.__es_metric_best_value
    #     # else:
    #     #     metrics_check = metric_value < self.__es_metric_best_value
    #     #
    #     # if metrics_check:
    #     #     print(f"\n\t Old best val {self.__es_metric}: {self.__es_metric_best_value:.4f} "
    #     #           f"| New best {self.__es_metric}: {metric_value:.4f}\n")
    #     #
    #     #     print("\t Saving new best model...")
    #     #     self.model.save(self.__path_to_best_model)
    #     #     print("\t -> New best model saved!")
    #     #
    #     #     self.__es_metric_best_value = metric_value
    #     #     self.__epochs_no_improvement = 0
    #     # else:
    #     #     self.__epochs_no_improvement += 1
    #     #     if self.__epochs_no_improvement == self.__patience:
    #     #         print(f" ** No decrease in val {self.__es_metric} "
    #     #               f"for {self.__patience} evaluations. Early stopping! ** ")
    #     #         return True
    #     #
    #     # print(" Epochs without improvement: ", self.__epochs_no_improvement)
    #     # print(" ........................................................... ")
    #     # return False


class DummyDataset(Dataset):
    def __getitem__(self, index):
        return torch.zeros(3, 224, 224), torch.zeros(1, dtype=torch.long)

    def __len__(self):
        return 1


def main():
    logger = logging.getLogger(__name__)
    logger.setLevel("INFO")
    ch = logging.StreamHandler()
    ch.setLevel("INFO")
    formatter = colorlog.ColoredFormatter(
        "%(log_color)s[%(asctime)s] - %(levelname)s - %(white)s%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    with open('config/training/training_configuration.yaml', 'r') as f:
        config = yaml.safe_load(f)
    with open('config/training/hypeparameter_configuration.yaml', 'r') as f:
        hyp = yaml.safe_load(f)

    train, val = torch.utils.data.DataLoader(DummyDataset()), torch.utils.data.DataLoader(DummyDataset())
    trainer = Trainer(CNN, config=config, hyperparameters=hyp, logger=logger)
    trainer.train(train, val)


if __name__ == "__main__":
    main()
