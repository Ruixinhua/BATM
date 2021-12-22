import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers import MultiHeadedAttention, AttLayer
from models.nc_models import BaseClassifyModel


class TextCNNClassifyModel(BaseClassifyModel):
    """Time-consuming and the performance is not good, score is about 0.67 in News26 with 1 CNN layer"""
    def __init__(self, **kwargs):
        super(TextCNNClassifyModel, self).__init__(**kwargs)
        self.num_filters, self.filter_sizes = kwargs.get("num_filters", 256), kwargs.get("filter_sizes", (2, 3, 4))
        self.conv_layers = nn.ModuleList(
            [nn.Conv2d(1, self.num_filters, (k, self.embed_dim)) for k in self.filter_sizes])
        self.classifier = nn.Linear(self.num_filters * len(self.filter_sizes), self.num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, input_feat, **kwargs):
        x = self.embedding_layer(input_feat).unsqueeze(1)
        x = torch.cat([self.conv_and_pool(x, conv) for conv in self.conv_layers], 1)
        x = nn.Dropout(self.dropout_rate)(x)
        return self.classify_layer(x)


class NRMSNewsEncoderModel(BaseClassifyModel):
    def __init__(self, **kwargs):
        super(NRMSNewsEncoderModel, self).__init__(**kwargs)
        head_num = self.embed_dim // 20
        self.mha_encoder = MultiHeadedAttention(head_num, 20, self.embed_dim)
        self.news_att = AttLayer(self.embed_dim, 128)

    def forward(self, input_feat, **kwargs):
        x = self.embedding_layer(input_feat)
        if self.variant_name == "one_att":
            x = self.news_att(x)[0]
        else:
            x = self.mha_encoder(x, x, x)[0]
            x = nn.Dropout(self.dropout_rate)(x)
            x = self.news_att(x)[0]
        return self.classify_layer(x)
