import argparse
import collections
import random
import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_args():
    args = argparse.ArgumentParser(description="Define Argument")
    args.add_argument("-r", "--resume", default=None, type=str,
                      help="path to latest checkpoint (default: None)")
    return args


def customer_args(args=None):
    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    # set default arguments
    options = [
        # global variables
        CustomArgs(["-ng", "--n_gpu"], type=int, target=None),
        CustomArgs(["-s", "--seed"], type=int, target=None),
        CustomArgs(["-ml", "--max_length"], type=int, target=None),
        CustomArgs(["-sm", "--save_model"], type=int, target=None),
        CustomArgs(["-et", "--embedding_type"], type=str, target=None),
        CustomArgs(["-sd", "--save_dir"], type=str, target=None),
        CustomArgs(["-rn", "--run_name"], type=str, target=None),
        # architecture params
        CustomArgs(["-at", "--arch_type"], type=str, target="arch_config"),
        CustomArgs(["-p", "--pooling"], type=str, target="arch_config"),
        CustomArgs(["-em", "--entropy_method"], type=str, target="arch_config"),
        CustomArgs(["-vn", "--variant_name"], type=str, target="arch_config"),
        CustomArgs(["-an", "--act_name"], type=str, target="arch_config"),
        CustomArgs(["-up", "--use_pretrained"], type=int, target="arch_config"),
        CustomArgs(["-nl", "--n_layers"], type=int, target="arch_config"),
        CustomArgs(["-ed", "--embedding_dim"], type=int, target="arch_config"),
        CustomArgs(["-ec", "--entropy_constraint"], type=int, target="arch_config"),
        CustomArgs(["-ap", "--add_pos"], type=int, target="arch_config"),
        CustomArgs(["-hn", "--head_num"], type=int, target="arch_config"),
        CustomArgs(["-hd", "--head_dim"], type=int, target="arch_config"),
        CustomArgs(["-al", "--alpha"], type=float, target="arch_config"),
        CustomArgs(["-dr", "--dropout_rate"], type=float, target="arch_config"),
        # dataloader params
        CustomArgs(["-na", "--name"], type=str, target="data_config"),
        CustomArgs(["-eb", "--embed_method"], type=str, target="data_config"),
        CustomArgs(["-bs", "--batch_size"], type=int, target="data_config"),
        # trainer params
        CustomArgs(["-ep", "--epochs"], type=int, target="trainer_config"),
        # optimizer
        CustomArgs(["-lr", "--lr"], type=float, target="optimizer_config"),
    ]
    if args:
        options.extend([CustomArgs(**ca) for ca in args])
    return options
