import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoModel

from base.base_model import BaseModel


class BaseClassifyModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__()
        self.att_weight = None
        self.output_hidden_states = kwargs.pop("output_hidden_states", True)
        self.return_attention = kwargs.pop("output_attentions", True)
        self.embed_dim = kwargs.pop("embed_dim", 300)
        self.__dict__.update(kwargs)
        if self.embedding_type == "glove":
            self.embed_dim = self.embeds.shape[1]
            self.embedding = nn.Embedding(len(self.word_dict), self.embed_dim, padding_idx=0)
            self.embedding = self.embedding.from_pretrained(torch.FloatTensor(self.embeds), freeze=False)
        elif self.embedding_type == "init":
            self.embedding = nn.Embedding(len(self.word_dict), self.embed_dim, padding_idx=0)
        else:
            # load weight and model from pretrained model
            self.config = AutoConfig.from_pretrained(self.embedding_type, num_labels=self.num_classes,
                                                     output_hidden_states=self.output_hidden_states,
                                                     output_attentions=self.return_attention)
            add_weight = self.add_weight if hasattr(self, "add_weight") else False
            layer_mapping = {"distilbert-base-uncased": "n_layers", "xlnet-base-cased": "n_layer",
                             "bert-base-uncased": "num_hidden_layers", "roberta-base": "num_hidden_layers",
                             "allenai/longformer-base-4096": "num_hidden_layers",
                             "transfo-xl-wt103": "n_layers"}
            self.config.__dict__.update({"add_weight": add_weight, layer_mapping[self.embedding_type]: self.n_layers})
            if self.embedding_type == "allenai/longformer-base-4096":
                self.config.attention_window = self.config.attention_window[:self.n_layers]
            embedding = AutoModel.from_pretrained(self.embedding_type, config=self.config)
            self.embedding = kwargs.get("bert")(self.config) if "bert" in kwargs else embedding
            self.embed_dim = self.config.dim if hasattr(self.config, "dim") else self.config.hidden_size
        self.classifier = nn.Linear(self.embed_dim, self.num_classes)

    def embedding_layer(self, input_feat):
        if self.embedding_type in ["glove", "init"]:
            embedding = self.embedding(input_feat["data"])
        else:
            input_feat["embedding"] = input_feat["embedding"] if "embedding" in input_feat else None
            output = self.embedding(input_feat["data"], input_feat["mask"], inputs_embeds=input_feat["embedding"])
            self.att_weight = output[-1]
            embedding = output[0]
        embedding = nn.Dropout(self.dropout_rate)(embedding)
        return embedding

    def classify_layer(self, latent, weight=None, return_attention=None):
        output = (self.classifier(latent),)
        return_attention = return_attention if return_attention else self.return_attention
        if return_attention:
            output = output + (weight,)
        return output

    def forward(self, input_feat, **kwargs):
        input_feat["embedding"] = input_feat.get("embedding", kwargs.get("inputs_embeds"))
        embedding = self.embedding_layer(input_feat)
        if self.embedding_type == "glove" or self.embedding_type == "init":
            embedding = torch.mean(embedding, dim=1)
        else:
            embedding = embedding[0][:, 0]  # shape of last hidden: (N, L, D), take the CLS for classification
        self.return_attention = kwargs.get("return_attention", False)
        return self.classify_layer(embedding, self.att_weight)


class PretrainedBaseline(BaseModel):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        super(PretrainedBaseline, self).__init__()
        self.use_pretrained = kwargs.get("use_pretrained", True)
        layer_mapping = {"distilbert-base-uncased": "n_layers", "xlnet-base-cased": "n_layer",
                         "bert-base-uncased": "num_hidden_layers", "roberta-base": "num_hidden_layers",
                         "allenai/longformer-base-4096": "num_hidden_layers",
                         "transfo-xl-wt103": "n_layer"}
        max_layers = {"bert-base-uncased": 12, "distilbert-base-uncased": 6, "allenai/longformer-base-4096": 12,
                      "xlnet-base-cased": 12, "roberta-base": 12}
        config = AutoConfig.from_pretrained(self.embedding_type, num_labels=self.num_classes)
        n_layers = min(self.n_layers, max_layers[self.embedding_type])
        if self.embedding_type == "allenai/longformer-base-4096":
            config.attention_window = config.attention_window[:n_layers]
        config.__dict__.update({layer_mapping[self.embedding_type]: n_layers, "pad_token_id": 0})
        if self.use_pretrained:
            self.model = AutoModelForSequenceClassification.from_pretrained(self.embedding_type, config=config)
        else:
            self.model = AutoModelForSequenceClassification.from_config(config=config)

    def forward(self, input_feat, **kwargs):
        feat_dict = {"input_ids": input_feat["data"], "attention_mask": input_feat["mask"]}
        if self.embedding_type == "transfo-xl-wt103":
            outputs = self.model(input_feat["data"])
        else:
            outputs = self.model(**feat_dict)
        outputs = (outputs.logits,)
        return outputs
