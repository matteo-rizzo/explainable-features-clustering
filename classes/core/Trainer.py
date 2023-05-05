import logging
import os
import time
from copy import deepcopy
from pathlib import Path
from typing import Optional, Type, Callable

import colorlog
import numpy as np
import torch
import torch.nn as nn
import torchmetrics
import yaml
from torch.cuda import amp
from torch.utils.data import Dataset
from tqdm import tqdm

from classes.MNISTDataset import MNISTDataset
from classes.deep_learning.architectures.CNN import CNN
from classes.factories.CriterionFactory import CriterionFactory
# from classes.deep_learning.architectures.modules.ExponentialMovingAverage import ExponentialMovingAverageModel
from classes.factories.OptimizerFactory import OptimizerFactory
from functional.lr_schedulers import linear_lrs, one_cycle_lrs
from functional.setup import get_device
from functional.torch_utils import strip_optimizer
from functional.utils import intersect_dicts, increment_path, check_file_exists, get_latest_run, colorstr


def fitness(x):
    # FIXME: to change
    # IDEA: could change these weights
    # Model fitness as a weighted combination of metrics
    w = [0.0, 0.0, 0.1, 0.9]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
    return (x[:, :4] * w).sum(1)


class Trainer:

    def __init__(self, model_class: Type[nn.Module], config: dict, hyperparameters: dict, logger: logging.Logger):
        self.config: dict = config
        self.hyperparameters: dict = hyperparameters
        self.logger: logging.Logger = logger
        # --- Training stuff ---
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.gradient_scaler: Optional[amp.GradScaler] = None
        # self.exponential_moving_average = None
        self.lr_schedule_fn: Optional[Callable] = None  # Scheduling function
        self.scheduler: torch.optim.lr_scheduler = None  # Torch scheduler
        self.criterion: torch.nn.modules.loss = None  # Loss
        self.accumulate: int = -1
        # ---
        self.device: Optional[torch.device] = get_device(config["device"])
        self.model_class: Type[nn.Module] = model_class
        self.model: Optional[nn.Module] = None
        self.checkpoint = None

    def train_one_epoch(self, train_dataloader: torch.utils.data.DataLoader, epoch: int):
        # --- Enable training ---
        self.model.train()
        # --- Console logging ---
        epoch_description: str = ""
        batch_number: int = len(train_dataloader)
        progress_bar = tqdm(enumerate(train_dataloader), total=batch_number)
        # --- Zero gradient once and train batches ---
        self.optimizer.zero_grad()

        # Number of warmup iterations, max(config epochs (e.g., 3), 1k iterations)
        warmup_number: int = max(round(self.hyperparameters['warmup_epochs'] * batch_number), 1000)
        for idx, (imgs, targets) in progress_bar:

            imgs, n_integrated_batches = self.__warmup_batch(imgs, batch_number, epoch, idx, warmup_number)
            with amp.autocast(enabled=self.device.type[:4] == "cuda"):
                # --- Forward pass ---
                preds = self.model(imgs)
                loss = self.criterion(preds, targets.to(self.config["device"]))

                # --- Backward ---
                self.gradient_scaler.scale(loss).backward()

                # --- Optimization ---
                if n_integrated_batches % self.accumulate == 0:
                    # Optimizer.step
                    self.gradient_scaler.step(self.optimizer)
                    self.gradient_scaler.update()
                    # Gradient reset
                    self.optimizer.zero_grad()
                    # if self.exponential_moving_average:
                    #     self.exponential_moving_average.update(self.model)

                # --- Console logging ---
                mem: str = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.2g}G'
                max_mem: float = torch.cuda.get_device_properties(self.device.index).total_memory
                total_mem: str = f"{(max_mem / 1E9) if torch.cuda.is_available() else 0:.2g}G"
                s = (f"{colorstr('bold', 'magenta', '[TRAIN]')}"
                     f"\t{colorstr('bold', 'magenta', 'Epoch')}: {epoch}/{self.config['epochs'] - 1}"
                     f"\t{colorstr('bold', 'magenta', 'gpu_mem')}: {mem}/{total_mem}"
                     f"\t{colorstr('bold', 'magenta', 'loss')}: {loss:.4f}"
                     f"\t{colorstr('bold', 'magenta', 'img_size')}: {imgs.shape[-2]}x{imgs.shape[-1]}"
                     )
                progress_bar.set_description(s)

        return epoch_description

    def __warmup_batch(self, imgs: torch.Tensor, batch_number: int,
                       epoch: int, i: int, warmup_number: int) -> [torch.Tensor, int]:

        n_integrated_batches: int = i + batch_number * epoch  # number integrated batches (since train start)
        imgs = imgs.to(self.device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0
        if n_integrated_batches <= warmup_number:
            x_interpolated = [0, warmup_number]
            self.accumulate = max(1, np.interp(n_integrated_batches, x_interpolated,
                                               [1, self.config["nominal_batch_size"]
                                                / self.config["batch_size"]]).round())
            for j, x in enumerate(self.optimizer.param_groups):
                # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                x['lr'] = np.interp(n_integrated_batches, x_interpolated,
                                    [self.hyperparameters['warmup_bias_lr'] if j == 2 else 0.0,
                                     x['initial_lr'] * self.lr_schedule_fn(epoch)])
                if 'momentum' in x:
                    x['momentum'] = np.interp(n_integrated_batches, x_interpolated,
                                              [self.hyperparameters['warmup_momentum'],
                                               self.hyperparameters['momentum']])
        return imgs, n_integrated_batches

    def train(self, train_dataloader: torch.utils.data.DataLoader, test_dataloader: torch.utils.data.DataLoader = None):
        # --- Directories, initialize where things are saved ---
        self.__start_or_resume_config()
        save_dir, weights_dir, last_ckpt, best_ckpt, results_file = self.__init_dump_folder()

        # --- Model ---
        pretrained: bool = self.config["weights"].endswith('.pt')
        self.__setup_model(pretrained=pretrained)
        self.__print_model()
        # --- Gradient accumulation ---
        self.accumulate: int = self.__setup_gradient_accumulation()

        # --- Optimization ---
        self.optimizer: torch.optim.Optimizer = self.__setup_optimizer()
        self.criterion: torch.nn.modules.loss = self.__setup_criterion()
        # TODO: make optional / modularize
        self.scheduler: torch.optim.lr_scheduler = self.__setup_scheduler()

        # --- Exponential moving average ---
        # self.exponential_moving_average = ExponentialMovingAverageModel(self.model)

        # --- Resume pretrained if necessary ---
        if pretrained:
            start_epoch, best_fitness = self.__resume_pretrained(results_file=results_file)
            self.checkpoint = None
        else:
            start_epoch, best_fitness = 0, 0.0

        # TODO: change
        results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
        self.scheduler.last_epoch = start_epoch - 1  # do not move
        self.gradient_scaler = amp.GradScaler(enabled=self.device.type[:4] == "cuda")
        self.logger.info(f'{colorstr("bright_green", "Image sizes (square)")}: TRAIN [{self.config["img_size"][0]}] , '
                         f'TEST [{self.config["img_size"][1]}]\t'
                         f'{colorstr("bright_green", "Batch size")}: {self.config["batch_size"]} '
                         f'({self.config["nominal_batch_size"]} nominal)\t'
                         f'{colorstr("bright_green", "Dataloader workers")}: {train_dataloader.num_workers}\t'
                         f'{colorstr("bright_green", "Saving results to")}: {save_dir}\n'
                         f'{" " * 31}{colorstr("bright_green", "Starting training for")} '
                         f'{self.config["epochs"]} epochs...')

        torch.save(self.model, weights_dir / 'init.pt')
        epoch: int = -1
        t0 = time.time()
        # Start training ------------------------------------------------------------------------
        for epoch in range(start_epoch, self.config["epochs"]):
            # --- Forward, backward, optimization ---
            progress_description: str = self.train_one_epoch(train_dataloader=train_dataloader, epoch=epoch)

            # --- Scheduler ---
            self.scheduler.step()
            # self.exponential_moving_average.update_attr(self.model, include=[.....])
            is_final_epoch: bool = epoch + 1 == self.config["epochs"]
            if (not self.config["notest"] or is_final_epoch) and test_dataloader:  # Calculate mAP
                self.test(test_dataloader)
            if is_final_epoch and not test_dataloader:
                self.logger.info("Test dataset was not given: skipping test...")
            # --- Write results ---
            with open(results_file, 'a') as ckpt:
                ckpt.write(progress_description + '%10.4g' * 7 % results + '\n')  # append metrics, val_loss

            # weighted combination of metrics
            fitness_value = fitness(np.array(results).reshape(1, -1))
            if fitness_value > best_fitness:
                best_fitness = fitness_value

            # Save model
            if (not self.config["nosave"]) or is_final_epoch:
                self.__save_model(best_fitness, epoch, fitness_value, best_ckpt, last_ckpt, results_file, weights_dir)
                self.checkpoint = None
        # End training --------------------------------------------------------------------------

        self.logger.info(f'{epoch - start_epoch + 1:g} epochs completed in {(time.time() - t0) / 3600:.3f} hours.\n')

        # --- Strip optimizers ---
        for ckpt in last_ckpt, best_ckpt:
            if ckpt.exists():
                strip_optimizer(ckpt)
        torch.cuda.empty_cache()
        return results

    def test(self, test_dataloader: torch.utils.data.DataLoader):
        # TODO: WIP
        # --- Disable training ---
        self.model.eval()
        # --- Console logging ---
        batch_number: int = len(test_dataloader)
        progress_bar = tqdm(enumerate(test_dataloader), total=batch_number)
        rolling_acc: float = 0
        for idx, (imgs, targets) in progress_bar:
            imgs = imgs.to(self.device, non_blocking=True)
            targets = targets.to(self.device)
            with torch.no_grad():
                preds = self.model(imgs)
                acc = torchmetrics.functional.accuracy(preds, targets, task="multiclass", num_classes=10)
                rolling_acc += acc
                s = (
                    f"{colorstr('bold', 'cyan', '[TEST]')}"
                    f"\t{colorstr('bold', 'cyan', 'Batch Acc')}: {acc/self.config['batch_size']:.3f}"
                    # f"\t{colorstr('bold', 'cyan', 'Rolling Acc')}: {rolling_acc/(idx+1):.3f}"
                )
                progress_bar.set_description(s)

        self.logger.info(f"Final metrics: Acc: {rolling_acc/batch_number:.3f}")

    def __start_or_resume_config(self) -> None:
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
            # Either check for configuration or pick default
            if not self.config["architecture_config"] == "default":
                self.config["architecture_config"] = check_file_exists(self.config["architecture_config"])
            self.config["hyperparameters"] = check_file_exists(self.config["hyperparameters"])
            # Increment run
            self.config["save_dir"] = increment_path(Path(self.config["project"]) / self.config["name"],
                                                     exist_ok=self.config["exist_ok"])
            self.logger.info(f'Starting a new training')

    def __init_dump_folder(self) -> [Path, Path, Path, Path, Path]:
        save_dir: Path = Path(f'{self.config["save_dir"]}')
        weights_dir: Path = save_dir / 'weights'
        weights_dir.mkdir(parents=True, exist_ok=True)
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
        # If pretrained, load checkpoint
        if pretrained:
            self.checkpoint = torch.load(self.config["weights"], map_location=self.device)
            self.model = self.model_class(
                config_path=self.config["architecture_config"] or self.checkpoint['model'].yaml,
                logger=self.logger).to(self.device)
            state_dict = self.checkpoint['model'].float().state_dict()  # to FP32
            state_dict = intersect_dicts(state_dict, self.model.state_dict(), exclude=[])  # intersect
            self.model.load_state_dict(state_dict, strict=False)
            self.logger.info(
                f'Transferred {len(state_dict):g}/{len(self.model.state_dict()):g} '
                f'items from {self.config["weights"]}')
        # Else initialize a new model
        else:
            self.model = self.model_class(config_path=self.config["architecture_config"],
                                          logger=self.logger).to(self.device)

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
        if self.config["linear_lr"]:
            self.lr_schedule_fn = linear_lrs(steps=self.config["epochs"],
                                             lrf=self.hyperparameters['lrf'])
        else:
            self.lr_schedule_fn = one_cycle_lrs(y1=1, y2=self.hyperparameters['lrf'],
                                                steps=self.config["epochs"])  # cosine 1->hyp['lrf']
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lr_schedule_fn)

    def __setup_optimizer(self) -> torch.optim.Optimizer:
        # TODO: look up different optimization for different parameter groups
        return OptimizerFactory(nn.ParameterList(self.model.parameters()),
                                hyperparameters=self.hyperparameters).get(self.config["optimizer"])

    def __setup_criterion(self) -> torch.nn.modules.loss:
        return CriterionFactory().get(self.config["criterion"])

    def __resume_pretrained(self, results_file) -> [int, float]:
        # --- Optimizer ---
        best_fitness: float = 0.0
        if self.checkpoint['optimizer'] is not None:
            self.optimizer.load_state_dict(self.checkpoint['optimizer'])
            best_fitness = self.checkpoint['best_fitness']

        # --- EMA ---
        # if self.exponential_moving_average and self.checkpoint.get('ema'):
        #     self.exponential_moving_average.ema_model.load_state_dict(self.checkpoint['ema'].float().state_dict())
        #     self.exponential_moving_average.updates = self.checkpoint['updates']

        # --- Results ---
        if self.checkpoint.get('training_results') is not None:
            results_file.write_text(self.checkpoint['training_results'])  # write results.txt

        # --- Epochs ---
        # TODO: check
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

    def __save_model(self, best_fitness: float, epoch: int, fitness_value: float,
                     best_ckpt: Path, last_ckpt: Path, results_file: Path,
                     weights_dir: Path):

        checkpoint = {'epoch': epoch,
                      'best_fitness': best_fitness,
                      'training_results': results_file.read_text(),
                      'model': deepcopy(self.model).half(),
                      # 'ema': deepcopy(self.exponential_moving_average.ema_model).half(),
                      # 'updates': self.exponential_moving_average.updates,
                      'optimizer': self.optimizer.state_dict(),
                      }
        # Save last, best and delete
        torch.save(checkpoint, last_ckpt)
        if best_fitness == fitness_value:
            torch.save(checkpoint, best_ckpt)
        if (best_fitness == fitness_value) and (epoch >= 200):
            torch.save(checkpoint, weights_dir / f'best_{epoch:03d}.pt')
        if epoch == 0:
            torch.save(checkpoint, weights_dir / f'epoch_{epoch:03d}.pt')
        elif ((epoch + 1) % 25) == 0:
            torch.save(checkpoint, weights_dir / f'epoch_{epoch:03d}.pt')
        elif epoch >= (self.config["epochs"] - 5):
            torch.save(checkpoint, weights_dir / f'epoch_{epoch:03d}.pt')

    def __print_model(self):
        self.logger.info(f'{"idx":>1}{"params":>10}  {"module":<40}{"parameters":<30}')
        self.logger.info(f'{"-" * 95}')
        for idx, module in enumerate(list(self.model.modules())[1:]):
            module_type = str(module.__class__)[8:-2]
            parameters = sum(dict((p.data_ptr(), p.numel()) for p in module.parameters()).values())
            arguments = str(module)[len(module_type.split(".")[-1]):]
            self.logger.info(f'{idx:>3}{parameters:10.0f}  {module_type:<40}{arguments}')
        self.logger.info(f'{"-" * 95}')
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

    train = torch.utils.data.DataLoader(MNISTDataset(train=True),
                                        batch_size=config["batch_size"],
                                        shuffle=True,
                                        num_workers=config["workers"])
    test = torch.utils.data.DataLoader(MNISTDataset(train=False),
                                       batch_size=config["batch_size"],
                                       shuffle=True,
                                       num_workers=config["workers"])
    trainer = Trainer(CNN, config=config, hyperparameters=hyp, logger=logger)
    trainer.train(train, test)


if __name__ == "__main__":
    main()
