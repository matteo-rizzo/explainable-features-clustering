from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import yaml

from classes.deep_learning.architectures.TorchModel import TorchModel
from functional.lr_schedulers import linear, one_cycle
from functional.setup import get_device
from functional.utils import intersect_dicts


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
        self.scheduler = None
        self.scaler = None
        self.ema = None  # Exponential Moving Average
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

    def __setup_model(self, pretrained: bool):
        if pretrained:
            self.checkpoint = torch.load(self.config["weights"], map_location=self.device)  # load checkpoint
            self.model = self.model_class(config_path=self.config["cfg"] or self.checkpoint['model'].yaml,
                                          logger=self.logger).to(self.device)
            state_dict = self.checkpoint['model'].float().state_dict()  # to FP32
            state_dict = intersect_dicts(state_dict, self.model.state_dict(), exclude=[])  # intersect
            self.model.load_state_dict(state_dict, strict=False)  # load
            self.logger.info(
                f'Transferred {len(state_dict):g}/{len(self.model.state_dict()):g} '
                f'items from {self.config["weights"]}')  # report
        else:
            self.model = self.model_class(config_path=self.config["cfg"], logger=self.logger).to(self.device)

    def __setup_gradient_accumulation(self):
        # If the total batch size is less than or equal to the nominal batch size, then accumulate is set to 1.
        # Accumulate losses before optimizing
        accumulate = max(round(self.config["nominal_batch_size"] / self.config["batch_size"]), 1)
        # Scale weight_decay
        self.hyperparameters['weight_decay'] *= (self.config["batch_size"] * accumulate
                                                 / self.config["nominal_batch_size"])
        self.logger.info(f"Scaled weight_decay = {self.hyperparameters['weight_decay']}")
        return accumulate

    def __setup_scheduler(self):
        if self.config["linear_lr"]:
            lr_schedule_fn = linear(steps=self.config["epochs"], lrf=self.hyperparameters['lrf'])
        else:
            lr_schedule_fn = one_cycle(y1=1, y2=self.hyperparameters['lrf'],
                                       steps=self.config["epochs"])  # cosine 1->hyp['lrf']
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_schedule_fn)

    def __setup_optimizer_parameters(self):
        pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
        for module_name, module in self.model.named_modules():
            if hasattr(module, 'bias') and isinstance(module.bias, nn.Parameter):
                pg2.append(module.bias)  # biases
            if isinstance(module, nn.BatchNorm2d):
                pg0.append(module.weight)  # no decay
            elif hasattr(module, 'weight') and isinstance(module.weight, nn.Parameter):
                pg1.append(module.weight)  # apply decay
            if hasattr(module, 'im'):
                if hasattr(module.im, 'implicit'):
                    pg0.append(module.im.implicit)
                else:
                    for iv in module.im:
                        pg0.append(iv.implicit)
            if hasattr(module, 'imc'):
                if hasattr(module.imc, 'implicit'):
                    pg0.append(module.imc.implicit)
                else:
                    for iv in module.imc:
                        pg0.append(iv.implicit)
            if hasattr(module, 'imb'):
                if hasattr(module.imb, 'implicit'):
                    pg0.append(module.imb.implicit)
                else:
                    for iv in module.imb:
                        pg0.append(iv.implicit)
            if hasattr(module, 'imo'):
                if hasattr(module.imo, 'implicit'):
                    pg0.append(module.imo.implicit)
                else:
                    for iv in module.imo:
                        pg0.append(iv.implicit)
            if hasattr(module, 'ia'):
                if hasattr(module.ia, 'implicit'):
                    pg0.append(module.ia.implicit)
                else:
                    for iv in module.ia:
                        pg0.append(iv.implicit)
            if hasattr(module, 'attn'):
                if hasattr(module.attn, 'logit_scale'):
                    pg0.append(module.attn.logit_scale)
                if hasattr(module.attn, 'q_bias'):
                    pg0.append(module.attn.q_bias)
                if hasattr(module.attn, 'v_bias'):
                    pg0.append(module.attn.v_bias)
                if hasattr(module.attn, 'relative_position_bias_table'):
                    pg0.append(module.attn.relative_position_bias_table)
            if hasattr(module, 'rbr_dense'):
                if hasattr(module.rbr_dense, 'weight_rbr_origin'):
                    pg0.append(module.rbr_dense.weight_rbr_origin)
                if hasattr(module.rbr_dense, 'weight_rbr_avg_conv'):
                    pg0.append(module.rbr_dense.weight_rbr_avg_conv)
                if hasattr(module.rbr_dense, 'weight_rbr_pfir_conv'):
                    pg0.append(module.rbr_dense.weight_rbr_pfir_conv)
                if hasattr(module.rbr_dense, 'weight_rbr_1x1_kxk_idconv1'):
                    pg0.append(module.rbr_dense.weight_rbr_1x1_kxk_idconv1)
                if hasattr(module.rbr_dense, 'weight_rbr_1x1_kxk_conv2'):
                    pg0.append(module.rbr_dense.weight_rbr_1x1_kxk_conv2)
                if hasattr(module.rbr_dense, 'weight_rbr_gconv_dw'):
                    pg0.append(module.rbr_dense.weight_rbr_gconv_dw)
                if hasattr(module.rbr_dense, 'weight_rbr_gconv_pw'):
                    pg0.append(module.rbr_dense.weight_rbr_gconv_pw)
                if hasattr(module.rbr_dense, 'vector'):
                    pg0.append(module.rbr_dense.vector)
        if self.config["adam"]:
            optimizer = torch.optim.Adam(pg0, lr=self.hyperparameters['lr0'],
                                         betas=(self.hyperparameters['momentum'], 0.999))  # adjust beta1 to momentum
        else:
            optimizer = torch.optim.SGD(pg0, lr=self.hyperparameters['lr0'], momentum=self.hyperparameters['momentum'],
                                        nesterov=True)

        optimizer.add_param_group(
            {'params': pg1, 'weight_decay': self.hyperparameters['weight_decay']})  # add pg1 with weight_decay
        optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
        self.logger.info(f'Optimizer groups: {len(pg2):g} .bias, {len(pg1):g} conv.weight, {len(pg0):g} other')
        # del pg0, pg1, pg2
        return optimizer

    def train(self):
        # --- Directories, initialize where to save things ---
        save_dir, weights_dir, last_ckpt, best_ckpt, results_file = self.__init_dump_folder()

        # --- Model ---
        pretrained: bool = self.config["weights"].endswith('.pt')
        self.__setup_model(pretrained=pretrained)

        # --- Gradient accumulation ---
        accumulate = self.__setup_gradient_accumulation()

        self.optimizer = self.__setup_optimizer_parameters()
        self.scheduler = self.__setup_scheduler()

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
