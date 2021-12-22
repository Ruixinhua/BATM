import torch
import torch.nn as nn
from transformers import BertConfig
from models.fastformer import FastformerEncoder
from models.nc_models import BaseClassifyModel


class FastformerClassifyModel(BaseClassifyModel):
    """
    Fastformer model by Wu, C., Wu, F., Qi, T., & Huang, Y. (n.d.). Fastformer: Additive Attention Can Be All You Need.
    and a part of https://github.com/wuch15/Fastformer
    """

    def __init__(self, **kwargs):
        super(FastformerClassifyModel, self).__init__(**kwargs)
        kwargs.update({
            "intermediate_size": self.embed_dim, "hidden_size": self.embed_dim, "embedding_dim": self.embed_dim,
            "max_position_embeddings": self.max_length, "hidden_dropout_prob": self.dropout_rate,
        })
        kwargs["max_position_embeddings"] = self.max_length
        kwargs["hidden_dropout_prob"] = self.dropout_rate
        self.word_dict, self.embeds = kwargs.pop("word_dict"), kwargs.pop("embeds")
        self.config = BertConfig.from_dict(kwargs)
        self.fastformer_model = FastformerEncoder(self.config)
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Embedding) and module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, input_feat, **kwargs):
        mask = input_feat["data"].bool().float()
        embedding = self.embedding_layer(input_feat)
        text_vec = self.fastformer_model(embedding, mask)
        output = self.classify_layer(text_vec)
        return output
