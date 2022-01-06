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
        if self.variant_name in ["base", "reuse"]:
            self.topic_layer = nn.Sequential(nn.Linear(self.embed_dim, topic_dim), nn.Tanh(),
                                             nn.Linear(topic_dim, self.head_num))
        elif self.variant_name == "topic_embed":
            self.topic_layer = nn.Embedding(len(self.word_dict), self.head_num)
        else:  # default setting
            self.topic_layer = nn.Linear(self.embed_dim, self.head_num)
        if self.variant_name == "gru" or self.variant_name == "combined_gru":
            self.gru = nn.GRU(self.embed_dim, self.embed_dim, 2, batch_first=True)
        if self.variant_name == "weight_mha":
            head_dim = self.embed_dim // 12
            self.sentence_encoder = MultiHeadedAttention(12, head_dim, self.embed_dim)
        if self.variant_name == "combined_mha":
            self.query = nn.Linear(self.embed_dim, topic_dim)
            self.key = nn.Linear(self.embed_dim, topic_dim)
        if self.variant_name == "reuse":
            self.projection = self.topic_layer
        else:
            self.projection = AttLayer(self.embed_dim, 128)

    def run_gru(self, embedding, length):
        try:
            embedding = pack_padded_sequence(embedding, lengths=length.cpu(), batch_first=True, enforce_sorted=False)
        except RuntimeError:
            raise RuntimeError()
        y, _ = self.gru(embedding)  # extract interest from history behavior
        y, _ = pad_packed_sequence(y, batch_first=True, total_length=self.max_length)
        return y

    def extract_topic(self, input_feat):
        embedding = self.embedding_layer(input_feat)  # (N, S, E) topic layer -> (N, S, H)
        if self.variant_name == "topic_embed":
            topic_weight = self.topic_layer(input_feat["data"]).transpose(1, 2)  # (N, H, S)
        else:
            topic_weight = self.topic_layer(embedding).transpose(1, 2)  # (N, H, S)
        # expand mask to the same size as topic weights
        mask = input_feat["mask"].expand(self.head_num, embedding.size(0), -1).transpose(0, 1) == 0
        topic_weight = torch.softmax(topic_weight.masked_fill(mask, -1e9), dim=-1)  # fill zero entry with -INF
        if self.variant_name == "combined_mha":
            # context_vec = torch.matmul(topic_weight, embedding)  # (N, H, E)
            query, key = [linear(embedding).view(embedding.size(0), -1, self.head_num, self.head_dim).transpose(1, 2)
                          for linear in (self.query, self.key)]
            # topic_vec, _ = self.mha(context_vec, context_vec, context_vec)  # (N, H, H*D)
            scores = torch.matmul(query, key.transpose(-2, -1)) / self.head_num ** 0.5  # (N, H, S, S)
            context_weight = torch.mean(scores, dim=-1)  # (N, H, S)
            topic_weight = context_weight * topic_weight  # (N, H, S)
        elif self.variant_name == "combined_gru":
            length = torch.sum(input_feat["mask"], dim=-1)
            embedding = self.run_gru(embedding, length)
        elif self.variant_name == "weight_mha":
            embedding = self.sentence_encoder(embedding, embedding, embedding)[0]
        topic_vec = self.final(torch.matmul(topic_weight, embedding))  # (N, H, E)
        return topic_vec, topic_weight

    def forward(self, input_feat, inputs_embeds=None, return_attention=False, **kwargs):
        input_feat["embedding"] = input_feat.get("embedding", inputs_embeds)
        topic_vec, topic_weight = self.extract_topic(input_feat)
        if self.variant_name == "reuse":
            doc_topic = torch.mean(self.topic_layer(topic_vec), -1).unsqueeze(-1)  # (N, H)
            doc_embedding = torch.sum(topic_vec * doc_topic, dim=1)  # (N, E)
        else:
            doc_embedding, doc_topic = self.projection(topic_vec)  # (N, E), (N, H)
        output = self.classify_layer(doc_embedding, topic_weight, return_attention)
        if self.entropy_constraint or self.calculate_entropy:
            entropy_sum = torch.sum(-topic_weight * torch.log(1e-6 + topic_weight)).squeeze() / self.head_num
            output = output + (entropy_sum,)
        return output
