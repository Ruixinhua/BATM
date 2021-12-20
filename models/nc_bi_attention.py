import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from models.nc_models import BaseClassifyModel
# from model.distill_bert import DistilBertModel
from models.layers import AttLayer, MultiHeadedAttention


class BiAttentionClassifyModel(BaseClassifyModel):
    def __init__(self, **kwargs):
        super(BiAttentionClassifyModel, self).__init__(**kwargs)
        self.variant_name = kwargs.pop("variant_name", "base")
        topic_dim = self.head_num * self.head_dim
        # the structure of basic model
        self.final = nn.Linear(self.embed_dim, self.embed_dim)
        if self.variant_name == "base":
            self.topic_layer = nn.Sequential(nn.Linear(self.embed_dim, topic_dim), nn.Tanh(),
                                             nn.Linear(topic_dim, self.head_num))
        elif self.variant_name == "raw":
            self.topic_layer = nn.Linear(self.embed_dim, self.head_num)
        self.projection = AttLayer(self.embed_dim, 128)

    def extract_topic(self, input_feat):
        embedding = self.embedding_layer(input_feat)  # (N, S, E) topic layer -> (N, S, H)
        topic_weight = self.topic_layer(embedding).transpose(1, 2)  # (N, H, S)
        # expand mask to the same size as topic weights
        mask = input_feat["mask"].expand(self.head_num, embedding.size(0), -1).transpose(0, 1) == 0
        topic_weight = torch.softmax(topic_weight.masked_fill(mask, -1e9), dim=-1)  # fill zero entry with -INF
        topic_vec = torch.matmul(topic_weight, embedding)  # (N, H, E)
        return topic_vec, topic_weight

    def forward(self, input_feat, inputs_embeds=None, return_attention=False, **kwargs):
        input_feat["embedding"] = input_feat.get("embedding", inputs_embeds)
        topic_vec, topic_weight = self.extract_topic(input_feat)
        doc_embedding, doc_topic = self.projection(topic_vec)
        output = self.classify_layer(doc_embedding, topic_weight, return_attention)
        if self.return_entropy:
            entropy_sum = torch.sum(-topic_weight * torch.log(1e-6 + topic_weight)).squeeze() / self.head_num
            output = output + (entropy_sum,)
        return output
