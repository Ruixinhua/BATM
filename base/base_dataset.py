import logging
from typing import List, Mapping

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from utils import text2index


class BaseDataset(Dataset):
    def __init__(self, texts, labels, label_dict, max_length, word_dict, process_method="keep_all"):
        super().__init__()
        self.texts, self.labels, self.label_dict, self.max_length = texts, labels, label_dict, max_length
        self.word_dict = word_dict
        self.process_method = process_method

        if self.label_dict is None and labels is not None:
            self.label_dict = dict(zip(sorted(set(labels)), range(len(set(labels)))))

    def __getitem__(self, i):
        data = text2index(self.texts[i], self.word_dict, self.process_method, True)[:self.max_length]
        data.extend([0 for _ in range(max(0, self.max_length - len(data)))])
        data = torch.tensor(data, dtype=torch.long)
        label = torch.tensor(self.label_dict.get(self.labels[i], -1), dtype=torch.long).squeeze(0)
        mask = torch.tensor(np.where(data == 0, 0, 1), dtype=torch.long)
        return {"data": data, "label": label, "mask": mask}

    def __len__(self):
        return len(self.labels)


class BaseDatasetBert(Dataset):
    def __init__(self, texts: List[str], labels: List[str] = None, label_dict: Mapping[str, int] = None,
                 max_length: int = 512, embedding_type: str = 'distilbert-base-uncased'):

        self.texts = texts
        self.labels = labels
        self.label_dict = label_dict
        self.max_length = max_length

        if self.label_dict is None and labels is not None:
            self.label_dict = dict(zip(sorted(set(labels)), range(len(set(labels)))))

        self.tokenizer = AutoTokenizer.from_pretrained(embedding_type)
        logging.getLogger("transformers.tokenization_utils").setLevel(logging.FATAL)

        # self.sep_vid = self.tokenizer.sep_token_id
        # self.cls_vid = self.tokenizer.cls_token_id
        if embedding_type == "transfo-xl-wt103":
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.pad_vid = self.tokenizer.pad_token_id
        else:
            self.pad_vid = self.tokenizer.pad_token_id

    def __len__(self):

        return len(self.texts)

    def __getitem__(self, index) -> Mapping[str, torch.Tensor]:

        x = self.texts[index]
        x_encoded = self.tokenizer.encode(x, add_special_tokens=True, max_length=self.max_length, truncation=True,
                                          return_tensors="pt").squeeze(0)

        true_seq_length = x_encoded.size(0)
        pad_size = self.max_length - true_seq_length
        pad_ids = torch.Tensor([self.pad_vid] * pad_size).long()
        x_tensor = torch.cat((x_encoded, pad_ids))

        mask = torch.ones_like(x_encoded, dtype=torch.int8)
        mask_pad = torch.zeros_like(pad_ids, dtype=torch.int8)
        mask = torch.cat((mask, mask_pad))

        output_dict = {"data": x_tensor, 'mask': mask}

        if self.labels is not None:
            y = self.labels[index]
            y_encoded = torch.Tensor([self.label_dict.get(y, -1)]).long().squeeze(0)
            output_dict["label"] = y_encoded

        return output_dict
