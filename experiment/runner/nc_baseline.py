import os
import ast
from pathlib import Path
from itertools import product
from experiment.config import ConfigParser
from experiment.config import init_args, customer_args, set_seed
from experiment.runner.nc_base import run, test, init_data_loader

default_seeds = [42, 2020, 2021, 25, 4]  # setup default seeds


if __name__ == "__main__":
    # setup seeds used to run baseline models
    baseline_args = [
        {"flags": ["-ss", "--seeds"], "type": str, "target": None},
        {"flags": ["-aa", "--arch_attr"], "type": str, "target": None},
        {"flags": ["-va", "--values"], "type": str, "target": None},
    ]
    args, options = init_args(), customer_args(baseline_args)
    config_parser = ConfigParser.from_args(args, options)
    data_loader = init_data_loader(config_parser)
    config = config_parser.config
    saved_dir = Path(config.project_root) / "saved" / "performance"  # init saved directory
    os.makedirs(saved_dir, exist_ok=True)  # create empty directory
    arch_attr = config.get("arch_attr", "base")  # test an architecture attribute
    saved_path = saved_dir / f'{config.data_config["name"].replace("/", "_")}_{arch_attr}.csv'
    test_values = config.get("values", "baseline").split(",")  # acquires test values for a given arch attribute
    seeds = [int(s) for s in config.seeds.split(",")] if hasattr(config, "seeds") else default_seeds
    for seed, value in product(seeds, test_values):
        try:
            config.arch_config[arch_attr] = ast.literal_eval(value)
        except ValueError:
            config.arch_config[arch_attr] = value
        setattr(config, "seed", seed)
        log = {"arch_type": config.arch_config["type"], "seed": config.seed, arch_attr: value}
        set_seed(log["seed"])
        trainer = run(config_parser, data_loader)
        log.update(test(trainer, data_loader))
        trainer.save_log(log, saved_path=saved_path)
