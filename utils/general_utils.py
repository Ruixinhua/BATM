import json
import os
from collections import OrderedDict
from pathlib import Path
from typing import Union, Dict

import torch


def read_json(file: Union[str, os.PathLike]):
    """
    Read json from file
    :param file: the path to the json file
    :return: ordered dictionary content
    """
    file = Path(file)
    with file.open("rt") as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content: Dict, file: Union[str, os.PathLike]):
    """
    Write content to a json file
    :param content: the content dictionary
    :param file: the path to save json file
    """
    file = Path(file)
    with file.open("wt") as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def write_to_file(file: Union[str, os.PathLike], text: Union[str, list]):
    with open(file, "w", encoding="utf-8") as w:
        if isinstance(text, str):
            w.write(text)
        elif isinstance(text, list):
            w.write("\n".join(text))


def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids
