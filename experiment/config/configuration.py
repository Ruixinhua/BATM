import copy
import os
from pathlib import Path
from typing import Dict, Union, Any
from utils import write_json, read_json, get_project_root


class Configuration:
    """
    This is the base class for all configuration class. Deal with the common hyper-parameters to all models'
    configuration, and include the methods for loading/saving configurations.
    For each sub configuration, a variable named 'type' is defined to indicate which class it belongs to

    """

    def __init__(self, **kwargs):
        # parameters in general
        self.n_gpu = kwargs.pop("n_gpu", 1)  # default using gpu for training
        self.embedding_type = kwargs.pop("embedding_type", "glove")
        self.max_length = kwargs.pop("max_length", 100)
        self.loss = kwargs.pop("loss", "cross_entropy")
        self.metrics = kwargs.pop("metrics", ["accuracy", "macro_f"])
        self.save_model = kwargs.pop("save_model", False)
        self.resume = kwargs.pop("resume", None)
        # setup default relative project path
        self.project_name = kwargs.pop("project_name", "bi_attention")
        self.project_root = kwargs.pop("project_root", get_project_root(project_name=self.project_name))
        self.data_root = os.path.join(self.project_root, "dataset")
        self.save_dir = kwargs.pop("save_dir", os.path.join(self.project_root, "saved"))
        self.seed = kwargs.pop("seed", 42)
        self.sub_configs = ["arch_config", "data_config", "trainer_config", "optimizer_config", "scheduler_config"]

        # parameters for architecture by default
        self.arch_config = {
            "type": "Baseline", "dropout_rate": 0.2, "embedding_type": self.embedding_type,
            "max_length": self.max_length,
        }
        self.arch_config.update(kwargs.pop("arch_config", {}))

        # parameters for loading data
        self.data_config = {
            "type": "NewsDataLoader", "batch_size": 32, "num_workers": 1, "name": "News26/keep_all",
            "max_length": self.max_length, "data_root": self.data_root, "embedding_type": self.embedding_type
        }
        self.data_config.update(kwargs.pop("data_config", {}))
        # identifier of experiment, default is identified by dataset name and architecture type.
        self.run_name = kwargs.pop("run_name", f"{self.data_config['name']}/{self.arch_config['type']}")

        # parameters for optimizer
        self.optimizer_config = {"type": "Adam", "lr": 1e-3, "weight_decay": 0}
        self.optimizer_config.update(kwargs.pop("optimizer_config", {}))

        # parameters for scheduler
        self.scheduler_config = {"type": "StepLR", "step_size": 50, "gamma": 0.1}
        self.scheduler_config.update(kwargs.pop("scheduler_config", {}))

        # parameters for trainer
        self.trainer_config = {
            "epochs": 3, "early_stop": 3, "monitor": "max val_accuracy", "verbosity": 2, "tensorboard": False
        }
        self.trainer_config.update(kwargs.pop("trainer_config", {}))

        # Additional attributes without default values
        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                raise err

    def get(self, key, default=None):
        if hasattr(self, key):
            return getattr(self, key)
        else:
            for sub in self.sub_configs:
                sub_config = getattr(self, sub)
                if key in sub_config:
                    return sub_config[key]
            return default

    def set(self, key, value):
        if hasattr(self, key):
            setattr(self, key, value)
        for sub in self.sub_configs:
            sub_config = getattr(self, sub)
            if key in sub_config:
                sub_config[key] = value

    def update(self, config_dict: Dict[str, Any]):
        """
        Updates attributes of this class with attributes from ``config_dict``.

        Args:
            config_dict (:obj:`Dict[str, Any]`): Dictionary of attributes that should be updated for this class.
        """
        for key, value in config_dict.items():
            if isinstance(value, dict):
                self.update_sub_config(key, **value)
            else:
                setattr(self, key, value)

    def update_sub_config(self, sub_name: str, **kwargs):
        """
        update corresponding sub configure dictionary
        :param sub_name: the name of sub-configuration, such as arch_config
        """
        getattr(self, sub_name).update(kwargs)

    def save_config(self, save_dir: Union[str, os.PathLike], config_name: str = "config.json"):
        """
        Save configuration with the saved directory with corresponding configuration name in a json file
        :param config_name: default is config.json, should be a json filename
        :param save_dir: the directory to save the configuration
        """
        if os.path.isfile(save_dir):
            raise AssertionError(f"Provided path ({save_dir}) should be a directory, not a file")
        os.makedirs(save_dir, exist_ok=True)
        config_file = Path(save_dir) / config_name
        write_json(copy.deepcopy(self.__dict__), config_file)

    @classmethod
    def from_json_file(cls, json_file: Union[str, os.PathLike]):
        """
        load configuration from a json file
        :param json_file: the path to the json file
        :return: a configuration object
        """
        return cls(**read_json(json_file))
