import pandas as pd
import random
import os
import numpy as np
from pathlib import Path
from datasets import load_dataset
from utils.preprocess_utils import clean_text, text2index
from utils.general_utils import read_json, write_json


def clean_df(data_df):
    data_df.dropna(subset=["title", "body"], inplace=True, how="all")
    data_df.fillna("", inplace=True)
    data_df["title"] = data_df.title.apply(lambda s: clean_text(s))
    data_df["body"] = data_df.body.apply(lambda s: clean_text(s))
    return data_df


def split_df(df, split=0.1, split_test=False):
    indices = df.index.values
    random.Random(42).shuffle(indices)
    split_len = round(split * len(df))
    df.loc[indices[:split_len], "split"] = "valid"
    if split_test:
        df.loc[indices[split_len:split_len*2], "split"] = "test"
        df.loc[indices[split_len*2:], "split"] = "train"
    else:
        df.loc[indices[split_len:], "split"] = "train"
    return df


def load_set_by_type(dataset, set_type: str) -> pd.DataFrame:
    df = {k: [] for k in ["data", "category"]}
    for text, label in zip(dataset[set_type]["text"], dataset[set_type]["label"]):
        for c, v in zip(["data", "category"], [text, label]):
            df[c].append(v)
    df["split"] = set_type
    return pd.DataFrame(df)


def load_dataset_df(dataset_name, data_path):
    if dataset_name in ["MIND15", "News26"]:
        df = clean_df(pd.read_csv(data_path, encoding="utf-8"))
        df["data"] = df.title + "\n" + df.body
    elif dataset_name in ["ag_news", "yelp_review_full", "imdb"]:
        # load corresponding dataset from datasets library
        dataset = load_dataset(dataset_name)
        train_set, test_set = split_df(load_set_by_type(dataset, "train")), load_set_by_type(dataset, "test")
        df = train_set.append(test_set)
    else:
        raise ValueError("dataset name should be in one of MIND15, IMDB, News26, and ag_news...")
    labels = df["category"].values.tolist()
    label_dict = dict(zip(sorted(set(labels)), range(len(set(labels)))))
    return df, label_dict


def load_word_dict(data_root, dataset_name, process_method, **kwargs):
    embed_method = kwargs.get("embed_method", "use_all")
    wd_path = Path(data_root) / "utils" / "word_dict" / f"{dataset_name}_{process_method}_{embed_method}.json"
    if os.path.exists(wd_path):
        word_dict = read_json(wd_path)
    else:
        word_dict = {}
        data_path = kwargs.get("data_path", Path(data_root) / "data" / f"{dataset_name}.csv")
        df = kwargs.get("df", load_dataset_df(dataset_name, data_path)[0])
        df.data.apply(lambda s: text2index(s, word_dict, process_method, False))
        os.makedirs(wd_path.parent, exist_ok=True)
        write_json(word_dict, wd_path)
    return word_dict


def load_glove_embedding(glove_path=None):
    if not glove_path:
        glove_path = "E:\\glove.840B.300d.txt"
    glove = pd.read_csv(glove_path, sep=" ", quoting=3, header=None, index_col=0)
    return {key: val.values for key, val in glove.T.items()}


def load_embeddings(data_root, dataset_name, process_method, word_dict, glove_path=None, embed_method="use_all"):
    embed_path = Path(data_root) / "utils" / "embed_dict" / f"{dataset_name}_{process_method}_{embed_method}.npy"
    wd_path = Path(data_root) / "utils" / "word_dict" / f"{dataset_name}_{process_method}_{embed_method}.json"
    if os.path.exists(embed_path):
        embeddings = np.load(embed_path.__str__())
        word_dict = read_json(wd_path)
    else:
        new_wd = {"[UNK]": 0}
        embedding_dict = load_glove_embedding(glove_path)
        embeddings, exclude_words = [np.zeros(300)], []
        for i, w in enumerate(word_dict.keys()):
            if w in embedding_dict:
                embeddings.append(embedding_dict[w])
                new_wd[w] = len(new_wd)
            else:
                exclude_words.append(w)
        if embed_method == "use_all":
            mean, std = np.mean(embeddings), np.std(embeddings)
            # append random embedding
            for i, w in enumerate(exclude_words):
                new_wd[w] = len(new_wd)
                embeddings.append(np.random.normal(loc=mean, scale=std, size=300))
        os.makedirs(embed_path.parent, exist_ok=True)
        np.save(embed_path.__str__(), np.array(embeddings))
        word_dict = new_wd
        write_json(word_dict, wd_path)
    return np.array(embeddings), word_dict
