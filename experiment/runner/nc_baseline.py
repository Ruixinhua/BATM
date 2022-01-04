import os
import ast
from pathlib import Path
from itertools import product
from experiment.config import ConfigParser
from experiment.config import init_args, customer_args, set_seed
from experiment.runner.nc_base import run, test, init_data_loader

# setup default values
DEFAULT_VALUES = {
    "seeds": [42, 2020, 2021, 25, 4],
    "head_num": [10, 30, 50, 70, 100, 150, 180, 200],
    "embedding_type": ["distilbert-base-uncased", "bert-base-uncased", "roberta-base", "allenai/longformer-base-4096",
                       "xlnet-base-cased"]
}


if __name__ == "__main__":
    # setup seeds used to run baseline models
    baseline_args = [
        {"flags": ["-ss", "--seeds"], "type": str, "target": None},
        {"flags": ["-aa", "--arch_attr"], "type": str, "target": None},
        {"flags": ["-va", "--values"], "type": str, "target": None},
    ]
    args, options = init_args(), customer_args(baseline_args)
    config_parser = ConfigParser.from_args(args, options)
    config = config_parser.config
    saved_dir = Path(config.project_root) / "saved" / "performance"  # init saved directory
    os.makedirs(saved_dir, exist_ok=True)  # create empty directory
    arch_attr = config.get("arch_attr", "base")  # test an architecture attribute
    saved_path = saved_dir / f'{config.data_config["name"].replace("/", "_")}_{arch_attr}.csv'
    # acquires test values for a given arch attribute
    test_values = config.get("values").split(",") if hasattr(config, "values") else DEFAULT_VALUES.get(arch_attr, [0])
    seeds = [int(s) for s in config.seeds.split(",")] if hasattr(config, "seeds") else DEFAULT_VALUES.get("seeds")
    for value, seed in product(test_values, seeds):
        try:
            config.set(arch_attr, ast.literal_eval(value))
        except ValueError:
            config.set(arch_attr, value)
        config.set("seed", seed)
        log = {"arch_type": config.arch_config["type"], "seed": config.seed, arch_attr: value,
               "variant_name": config.arch_config.get("variant_name", None)}
        set_seed(log["seed"])
        data_loader = init_data_loader(config_parser)
        trainer = run(config_parser, data_loader)
        log.update(test(trainer, data_loader))
        trainer.save_log(log, saved_path=saved_path)
