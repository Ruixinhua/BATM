import os
import pandas as pd
import numpy as np

from pathlib import Path
from itertools import product
from utils import get_project_root, del_index_column


def get_mean(values):
    return f"{np.round(np.mean(values), 2)}"+u"\u00B1"+f"{np.round(np.std(values), 2)}"


if __name__ == "__main__":
    names = ["News26", "MIND15"]
    datasets = ["keep_all", "aggressive", "alphabet_only"]
    test_args = ["head_num", "embedding_type"]
    for name, set_type, arg in product(names, datasets, test_args):
        root_path = Path(get_project_root()) / "saved"
        file_name = f"{name}_{set_type}_{arg}.csv"
        df_file = root_path / "performance" / file_name
        saved_path = root_path / "stat"
        os.makedirs(saved_path, exist_ok=True)
        if os.path.exists(df_file):
            stat_df = pd.DataFrame()
            per_df = del_index_column(pd.read_csv(df_file))
            for (arch_type, arg_value), group in per_df.groupby(["arch_type", arg]):
                metrics = [f"{d}_{m}" for d, m in product(["val", "test"], ["loss", "accuracy", "macro_f"])]
                mean_values = [get_mean(group[m].values*100) for m in metrics]
                group = group.drop(columns=metrics+["seed", "run_id", "dropout_rate"]).drop_duplicates()
                group[metrics] = pd.DataFrame([mean_values], index=group.index)
                stat_df = stat_df.append(group, ignore_index=True)
            del_index_column(stat_df).to_csv(saved_path / file_name)
