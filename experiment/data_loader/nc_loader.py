from pathlib import Path
from torch.utils.data import DataLoader
from utils import load_dataset_df, load_word_dict, load_embeddings
from base.base_dataset import BaseDatasetBert, BaseDataset


class NewsDataLoader:
    def load_dataset(self, df):
        pretrained_models = ["distilbert-base-uncased", "bert-base-uncased", "xlnet-base-cased", "roberta-base",
                             "allenai/longformer-base-4096", "transfo-xl-wt103"]
        if self.embedding_type in pretrained_models:
            dataset = BaseDatasetBert(texts=df["data"].values.tolist(), labels=df["category"].values.tolist(),
                                      label_dict=self.label_dict, max_length=self.max_length,
                                      embedding_type=self.embedding_type)
            if self.embedding_type == "transfo-xl-wt103":
                self.word_dict = dataset.tokenizer.sym2idx
            else:
                self.word_dict = dataset.tokenizer.vocab
        elif self.embedding_type in ["glove", "init"]:
            # if we use glove embedding, then we ignore the unknown words
            dataset = BaseDataset(df["data"].values.tolist(), df["category"].values.tolist(), self.label_dict,
                                  self.max_length, self.word_dict, self.method)
        else:
            raise ValueError(f"Embedding type should be one of {','.join(pretrained_models)} or glove and init")
        return dataset

    def __init__(self, batch_size=32, shuffle=True, num_workers=1, max_length=128, name="MIND15/keep", **kwargs):
        self.set_name, self.method = name.split("/")[0], name.split("/")[1]
        self.max_length, self.embedding_type = max_length, kwargs.get("embedding_type", "glove")
        self.data_root = kwargs.get("data_root", "../../dataset")
        data_path = Path(self.data_root) / "data" / f"{self.set_name}.csv"
        df, self.label_dict = load_dataset_df(self.set_name, data_path)
        train_set, valid_set, test_set = df["split"] == "train", df["split"] == "valid", df["split"] == "test"
        if self.embedding_type in ["glove", "init"]:
            # setup word dictionary for glove or init embedding
            self.word_dict = load_word_dict(self.data_root, self.set_name, self.method, df=df)
        if self.embedding_type == "glove":
            self.embeds, self.word_dict = load_embeddings(self.data_root, self.set_name, self.method, self.word_dict,
                                                          embed_method=kwargs.get("embed_method", "use_all"))
        self.init_params = {'batch_size': batch_size, 'shuffle': shuffle, 'num_workers': num_workers}
        # initialize train loader
        self.train_loader = DataLoader(self.load_dataset(df[train_set]), **self.init_params)
        # initialize validation loader
        self.valid_loader = DataLoader(self.load_dataset(df[valid_set]), **self.init_params)
        # initialize test loader
        self.test_loader = DataLoader(self.load_dataset(df[test_set]), **self.init_params)
