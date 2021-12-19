import copy
import os
import utils.loss_utils as module_loss
import utils.metric_utils as module_metric
from utils import prepare_device
from experiment.config.default_config import arch_default_config
import torch
import torch.distributed
import pandas as pd
from abc import abstractmethod
from numpy import inf
from logger import TensorboardWriter
from pathlib import Path


class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, model, config):
        self.config = config.config
        cfg_trainer = config["trainer_config"]
        self.logger = config.get_logger("trainer", cfg_trainer["verbosity"])
        # prepare for (multi-device) GPU training
        self.device, device_ids = prepare_device(config["n_gpu"])
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)
        # set up model parameters
        self.best_model = model
        # get function handles of loss and metrics
        self.criterion = getattr(module_loss, config["loss"])
        self.metric_ftns = [getattr(module_metric, met) for met in config["metrics"]]
        # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        self.optimizer = config.init_obj("optimizer_config", torch.optim, trainable_params)
        self.lr_scheduler = config.init_obj("scheduler_config", torch.optim.lr_scheduler, self.optimizer)
        # set up trainer parameters
        self.epochs = cfg_trainer["epochs"]
        self.save_model = config["save_model"]
        self.monitor = cfg_trainer.get("monitor", "off")
        self.last_best_path = None
        self.not_improved_count = 0

        # configuration to monitor model performance and save best
        if self.monitor == "off":
            self.mnt_mode = "off"
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ["min", "max"]

            self.mnt_best = inf if self.mnt_mode == "min" else -inf
            self.early_stop = cfg_trainer.get("early_stop", inf)
            if self.early_stop <= 0:
                self.early_stop = inf

        self.start_epoch = 1
        self.checkpoint_dir = config.save_dir

        # setup visualization writer instance                
        self.writer = TensorboardWriter(config.log_dir, self.logger, cfg_trainer["tensorboard"])

        if config["resume"] is not None:
            self._resume_checkpoint(config["resume"])

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def _log_info(self, log):
        # print logged information to the screen
        for key, value in log.items():
            self.logger.info("    {:15s}: {}".format(str(key), value))

    def save_log(self, log, **kwargs):
        log["seed"] = self.config["seed"]
        arch_config = self.config["arch_config"]
        default_config = arch_default_config(arch_config.get("type"))
        for key in arch_config.keys():
            if default_config.get(key, None) != arch_config.get(key):
                log[key] = arch_config.get(key)
        log["run_id"] = self.config["run_name"]
        saved_path = kwargs.get("saved_path", Path(self.checkpoint_dir) / "model_best.csv")
        log_df = pd.DataFrame(log, index=[0])
        if os.path.exists(saved_path):
            log_df = log_df.append(pd.read_csv(saved_path, float_precision="round_trip"), ignore_index=True)
        log_df = log_df.loc[:, ~log_df.columns.str.contains("^Unnamed")]
        log_df.to_csv(saved_path)

    def _monitor(self, log, epoch):
        # evaluate model performance according to configured metric, save best checkpoint as model_best with score
        if self.mnt_mode != "off":
            try:
                # check whether model performance improved or not, according to specified metric(mnt_metric)
                improved = (self.mnt_mode == "min" and log[self.mnt_metric] <= self.mnt_best) or \
                           (self.mnt_mode == "max" and log[self.mnt_metric] >= self.mnt_best)
            except KeyError:
                err_msg = f"Warning:Metric {self.mnt_metric} is not found.Model performance monitoring is disabled."
                self.logger.warning(err_msg)
                self.mnt_mode = "off"
                improved = False
            log["split"] = "valid"
            self.save_log(log)

            if improved:
                self.mnt_best = log[self.mnt_metric]
                self.not_improved_count = 0
                self.best_model = copy.deepcopy(self.model)
                if self.save_model:
                    self._save_checkpoint(epoch, log[self.mnt_metric])
            else:
                self.not_improved_count += 1

    def train(self):
        """
        Full training logic


        """
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            # save logged information into log dict
            log = {"epoch": epoch}
            log.update(result)
            self._log_info(log)
            self._monitor(log, epoch)
            if self.not_improved_count > self.early_stop:
                self.logger.info(f"Validation performance did not improve for {self.early_stop} epochs. "
                                 "Training stops.")
                break

    def _save_checkpoint(self, epoch, score=0.0):
        """
        Saving checkpoints
        :param epoch: current epoch number
        :param score: current score of monitor metric
        """
        arch = type(self.model).__name__
        state = {
            "arch": arch,
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "monitor_best": self.mnt_best,
            "config": self.config
        }
        best_path = str(self.checkpoint_dir / f"{round(score, 4)}_model_best-epoch{epoch}.pth")
        if self.last_best_path:
            if os.path.exists(self.last_best_path):
                os.remove(self.last_best_path)
        torch.save(state, best_path)
        self.logger.info(f"Saving current best: {best_path}")
        self.last_best_path = best_path

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info(f"Loading checkpoint: {resume_path} ...")
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint["epoch"] + 1
        self.mnt_best = checkpoint["monitor_best"]

        # load architecture params from checkpoint.
        if checkpoint["config"]["arch_config"] != self.config["arch_config"]:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        # if torch.distributed.is_initialized():
        #     self.model.load_state_dict(checkpoint["state_dict"])
        # else:
        #     self.model.load_state_dict(checkpoint["state_dict"])
        self.model.load_state_dict(checkpoint["state_dict"])
        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint["config"]["optimizer_config"] != self.config["optimizer_config"]:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

        self.logger.info(f"Checkpoint loaded. Resume training from epoch {self.start_epoch}")
