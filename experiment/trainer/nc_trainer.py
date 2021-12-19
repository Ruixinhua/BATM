import numpy as np
import torch
from base import BaseTrainer
from utils import MetricTracker
from tqdm import tqdm


class NCTrainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, model, config, data_loader, **kwargs):
        super().__init__(model, config)
        self.config = config
        self.data_loader = data_loader.train_loader
        arch_config = self.config["arch_config"]
        self.return_entropy = arch_config.get("return_entropy", False)
        self.alpha = arch_config.get("alpha", 0.001)
        self.len_epoch = len(self.data_loader)
        self.valid_loader = data_loader.valid_loader
        self.do_validation = self.valid_loader is not None
        self.log_step = int(np.sqrt(self.data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def load_batch_data(self, batch_dict):
        """
        load batch data to default device
        """
        return {k: v.to(self.device) for k, v in batch_dict.items()}

    def run_model(self, batch_dict, model=None):
        """
        run model with the batch data
        :param batch_dict: the dictionary of data with format like {"data": Tensor(), "label": Tensor()}
        :param model: by default we use the self model
        :return: the output of running, label used for evaluation, and loss item
        """
        batch_dict = self.load_batch_data(batch_dict)
        output = model(batch_dict) if model is not None else self.model(batch_dict)
        if self.return_entropy:
            loss = self.criterion(output[0], batch_dict["label"]) + self.alpha * output[-1]
            output = output[0]
        else:
            output = output[0]
            loss = self.criterion(output, batch_dict["label"])
        return output, batch_dict["label"], loss

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        bar = tqdm(enumerate(self.data_loader), total=len(self.data_loader))
        for batch_idx, batch_dict in bar:
            self.optimizer.zero_grad()  # setup gradient to zero
            output, label, loss = self.run_model(batch_dict, self.model)  # run model
            loss.backward()  # backpropagation
            self.optimizer.step()  # gradient descent
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item(), n=len(batch_dict["data"]))  # update metrix
            for met in self.metric_ftns:  # run metric functions
                self.train_metrics.update(met.__name__, met(output, label), n=len(batch_dict["data"]))
            if batch_idx % self.log_step == 0:  # set bar
                bar.set_description(f"Train Epoch: {epoch} Loss: {loss.item()}")
            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()
        if self.do_validation:
            log.update(self._valid_epoch(epoch))  # update validation log

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def evaluate(self, loader, model, epoch=0, prefix="val"):
        model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, batch_dict in tqdm(enumerate(loader), total=len(loader)):
                output, label, loss = self.run_model(batch_dict, model)
                self.writer.set_step((epoch - 1) * len(loader) + batch_idx, 'evaluate')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, label))
        for name, p in model.named_parameters():  # add histogram of model parameters to the tensorboard
            self.writer.add_histogram(name, p, bins='auto')
        return {f"{prefix}_{k}": v for k, v in self.valid_metrics.result().items()}  # return log with prefix

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        return self.evaluate(self.valid_loader, self.model, epoch)
