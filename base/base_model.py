import torch.nn as nn
import numpy as np
from abc import abstractmethod


class BaseModel(nn.Module):
    """
    Base class for all models
    """
    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        self.trained_params = sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, self.parameters())])
        self.freeze_params = sum([np.prod(p.size()) for p in filter(lambda p: not p.requires_grad, self.parameters())])
        self.trained_params = format(self.trained_params, ",")
        return super().__str__() + f"\nTrainable params: {self.trained_params}\nFreeze params: {self.freeze_params}"
