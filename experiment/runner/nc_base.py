import os
import numpy as np
import models as module_arch
import experiment.data_loader as module_data
from typing import Union
from torch.backends import cudnn
from scipy.stats import entropy
from experiment.data_loader import NewsDataLoader
from experiment.trainer import NCTrainer
from experiment.config import ConfigParser, init_args, customer_args, set_seed
from utils.topic_utils import get_topic_dist, save_topic_info


def init_default_model(config_parser: ConfigParser, data_loader: NewsDataLoader):
    # build a default model architecture
    model_params = {"num_classes": len(data_loader.label_dict), "word_dict": data_loader.word_dict}
    if hasattr(data_loader, "embeds"):
        model_params.update({"embeds": data_loader.embeds})
    model = config_parser.init_obj("arch_config", module_arch, **model_params)
    return model


def init_data_loader(config_parser: ConfigParser):
    # setup data_loader instances
    data_loader = config_parser.init_obj("data_config", module_data)
    return data_loader


def run(config_parser: ConfigParser, data_loader: NewsDataLoader):
    cudnn.benchmark = False
    cudnn.deterministic = True
    logger = config_parser.get_logger("train")
    model = init_default_model(config_parser, data_loader)
    logger.info(model)
    trainer = NCTrainer(model, config_parser, data_loader)
    trainer.train()
    return trainer


def test(trainer: NCTrainer, data_loader: NewsDataLoader):
    log = {}
    # run validation
    log.update(trainer.evaluate(data_loader.valid_loader, trainer.best_model, prefix="val"))
    # run test
    log.update(trainer.evaluate(data_loader.test_loader, trainer.best_model, prefix="test"))
    return log


def topic_evaluation(trainer: NCTrainer, data_loader: NewsDataLoader, path: Union[str, os.PathLike]):
    # statistic topic distribution of Topic Attention network
    reverse_dict = {v: k for k, v in data_loader.word_dict.items()}
    topic_dist = get_topic_dist(trainer, list(data_loader.word_dict.values()))
    topic_result = save_topic_info(path, topic_dist, reverse_dict, data_loader)
    topic_result.update({"token_entropy": np.mean(entropy(topic_dist, axis=1))})
    return topic_result


if __name__ == "__main__":
    args, options = init_args(), customer_args()
    main_config = ConfigParser.from_args(args, options)
    set_seed(main_config["seed"])
    run(main_config, init_data_loader(main_config))
