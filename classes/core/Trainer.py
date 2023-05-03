import logging
import os
import time
from copy import deepcopy
from pathlib import Path
from typing import Optional, Type

import colorlog
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.cuda import amp
from torch.utils.data import Dataset
from tqdm import tqdm

from classes.deep_learning.architectures.CNN import CNN
from classes.deep_learning.architectures.TorchModel import TorchModel
# from classes.deep_learning.architectures.modules.ExponentialMovingAverage import ExponentialMovingAverageModel
from classes.factories.OptimizerFactory import OptimizerFactory
from functional.lr_schedulers import linear_lrs, one_cycle_lrs
from functional.setup import get_device
from functional.torch_utils import strip_optimizer
from functional.utils import intersect_dicts, increment_path, check_file_exists, get_latest_run, colorstr


# from classifiers.deep_learning.classes.core.Evaluator import Evaluator
# from classifiers.deep_learning.classes.factories.ModelFactory import ModelFactory
# from classifiers.deep_learning.classes.utils.Params import Params

def fitness(x):
    # FIXME: to change
    # IDEA: could change these weights
    # Model fitness as a weighted combination of metrics
    w = [0.0, 0.0, 0.1, 0.9]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
    return (x[:, :4] * w).sum(1)


class Trainer:

    def __init__(self, model_class: Type[TorchModel], config: dict, hyperparameters: dict, logger):
        self.config = config
        self.hyperparameters = hyperparameters
        self.logger = logger
        # --- Training stuff ---
        self.optimizer = None
        self.gradient_scaler = None
        # self.exponential_moving_average = None
        self.lr_schedule_fn = None  # Scheduling function
        self.scheduler = None  # Torch scheduler
        self.accumulate: int = -1
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

    def train_one_epoch(self, train_dataloader: torch.utils.data.DataLoader, epoch: int):
        # --- Enable training ---
        self.model.train()
        # --- Console logging ---
        epoch_description: str = ""
        batch_number: int = len(train_dataloader)
        progress_bar = tqdm(enumerate(train_dataloader), total=batch_number)
        # --- Zero gradient and train batch ---
        self.optimizer.zero_grad()

        # Number of warmup iterations, max(e.g. 3 epochs, 1k iterations)
        warmup_number: int = max(round(self.hyperparameters['warmup_epochs'] * batch_number), 1000)
        for i, (imgs, targets, paths, _) in progress_bar:

            imgs, n_integrated_batches = self.__warmup_batch(imgs, batch_number, epoch, i, warmup_number)
            # --- Forward ---
            with amp.autocast(enabled=self.device[:4] == "cuda:0"[:4]):
                # --- Forward pass ---
                pred = self.model(imgs)

                if 'loss_ota' not in self.hyperparameters or self.hyperparameters['loss_ota'] == 1:
                    loss, loss_items = self.compute_loss_ota(pred, targets.to(self.config["device"]),
                                                             imgs)  # losses scaled by batch_size
                else:
                    # losses scaled by batch_size
                    loss, loss_items = self.compute_loss(pred, targets.to(self.config["device"]))

                # --- Backward ---
                self.gradient_scaler.scale(loss).backward()

                # --- Optimization ---
                if n_integrated_batches % self.accumulate == 0:
                    self.gradient_scaler.step(self.optimizer)  # optimizer.step
                    self.gradient_scaler.update()
                    self.optimizer.zero_grad()
                    # if self.exponential_moving_average:
                    #     self.exponential_moving_average.update(self.model)

                # --- Console logging ---
                mean_loss = (mean_loss * i + loss_items) / (i + 1)  # update mean losses
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)

                s = (f"\t{colorstr('bold', 'magenta', 'Epoch')}: {epoch}/{self.config['epochs'] - 1}"
                     f"\t{colorstr('bold', 'magenta', 'gpu_mem')}: {mem}"
                     # f"\t{colorstr('bold', 'magenta', 'box')}: {box_loss:.4f}"
                     # f"\t{colorstr('bold', 'magenta', 'obj')}: {obj_loss:.4f}"
                     # f"\t{colorstr('bold', 'magenta', 'cls')}: {cls_loss:.4f}"
                     f"\t{colorstr('bold', 'magenta', 'mean loss')}: {mean_loss:.4f}"
                     f"\t{colorstr('bold', 'magenta', 'labels')}: {targets.shape[0]}"
                     f"\t{colorstr('bold', 'magenta', 'img_size')}: {imgs.shape[-1]}"
                     )
                progress_bar.set_description(s)

        return epoch_description
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

    def __warmup_batch(self, imgs, batch_number: int, epoch: int, i: int, warmup_number: int):
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

    def train(self, train_dataloader, val_dataloader):
        # --- Directories, initialize where things are saved ---
        self.__start_or_resume_config()
        save_dir, weights_dir, last_ckpt, best_ckpt, results_file = self.__init_dump_folder()

        # --- Model ---
        pretrained: bool = self.config["weights"].endswith('.pt')
        self.__setup_model(pretrained=pretrained)

        # --- Gradient accumulation ---
        self.accumulate: int = self.__setup_gradient_accumulation()

        # --- Optimization ---
        # TODO: make optional / modularize
        self.optimizer: torch.optim.Optimizer = self.__setup_optimizer()
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
        self.gradient_scaler = amp.GradScaler(enabled=self.device[:4] == "cuda:0"[:4])
        # self.compute_loss_ota = ComputeLossOTA(self.model)  # init losses class
        # self.compute_loss = ComputeLoss(self.model)  # init losses class
        self.logger.info(f'{colorstr("bright_green", "Image sizes")}: TRAIN [{self.config["img_size"][0]}] , '
                         f'TEST [{self.config["img_size"][1]}]\t'
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
            s = self.train_one_epoch(train_dataloader=train_dataloader, epoch=epoch)

            # --- Scheduler ---
            self.scheduler.step()
            # mAP
            # self.exponential_moving_average.update_attr(self.model,
            # include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride', 'class_weights'])
            is_final_epoch: bool = epoch + 1 == self.config["epochs"]
            # TODO: test?

            # --- Write results ---
            with open(results_file, 'a') as ckpt:
                ckpt.write(s + '%10.4g' * 7 % results + '\n')  # append metrics, val_loss

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
