import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForSequenceClassification, BertConfig, AutoModel

from base import BaseModel
# from model.distill_bert import DistilBertModel
from models.fastformer import FastformerEncoder
from models.layers import AttLayer, MultiHeadedAttention, activation_layer


class BaseClassifyModel(BaseModel):
    def __init__(self, bert=None, **kwargs):
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
            self.embedding = bert(self.config) if bert else embedding
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

    def forward(self, input_feat, inputs_embeds=None, return_attention=False, **kwargs):
        input_feat["embedding"] = input_feat.get("embedding", inputs_embeds)
        embedding = self.embedding_layer(input_feat)
        if self.embedding_type == "glove" or self.embedding_type == "init":
            embedding = torch.mean(embedding, dim=1)
        else:
            embedding = embedding[0][:, 0]  # shape of last hidden: (N, L, D), take the CLS for classification
        self.return_attention = return_attention
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
        outputs = (outputs.logits, )
        return outputs


class BertAvgClassifyModel(BaseClassifyModel):
    def __init__(self, bert=None, **kwargs):
        super(BertAvgClassifyModel, self).__init__(bert, **kwargs)
        self.att_layer = AttLayer(self.embed_dim, 256)

    def forward(self, input_feat, return_attention=False, inputs_embeds=None, **kwargs):
        input_feat["embedding"] = input_feat.get("embedding", inputs_embeds)
        embedding = self.embedding_layer(input_feat)
        hidden_avg = self.att_layer(embedding)[0]
        # shape of hidden average: (N, D) which compute the average output of each token
        self.return_attention = return_attention
        return self.classify_layer(hidden_avg, self.att_weight)


class TopicExtractorClassifyModel(BaseClassifyModel):
    def __init__(self, bert=None, **kwargs):
        super().__init__(bert, **kwargs)
        self.__dict__.update(kwargs)
        self.act_name = self.act_name if hasattr(self, "act_name") else None
        self.return_entropy = self.return_entropy if hasattr(self, "return_entropy") else False
        self.entropy_method = self.entropy_method if hasattr(self, "entropy_method") else None
        self.variant_name = self.variant_name if hasattr(self, "variant_name") else "share_topic"
        self.encoder_name = self.encoder_name if hasattr(self, "encoder_name") else "MHA"
        self.pooling = self.pooling if hasattr(self, "pooling") else "topic"
        # using multi-head attention to extract the topical information of the sentence
        if self.encoder_name == "MHA":
            self.sentence_encoder = MultiHeadedAttention(self.head_num, self.head_dim, self.embed_dim)
        elif self.encoder_name == "Fastformer":
            kwargs["intermediate_size"] = self.embed_dim
            kwargs["hidden_size"] = self.embed_dim
            self.config = BertConfig.from_dict(kwargs)
            self.sentence_encoder = FastformerEncoder(self.config, pooler_count=0)
            self.head_dim = int(self.embed_dim / self.head_num)

        if self.act_name:
            self.activation_layer = activation_layer(self.act_name)
            self.projection = nn.Sequential(
                nn.Linear(self.embed_dim, 1),
                self.activation_layer
            )
        else:
            self.projection = nn.Linear(self.embed_dim, 1)
        if self.variant_name == "distribute_topic":
            self.token_weight = nn.Linear(self.head_dim * self.head_num, self.head_num)
        else:
            self.token_weight = nn.Linear(self.head_dim, 1)
        self.classifier = nn.Linear(self.head_num, self.num_classes)
        if self.variant_name == "additive":
            self.projection = AttLayer(self.embed_dim, 128)
            self.classifier = nn.Linear(self.embed_dim, self.num_classes)
        if self.pooling == "additive":
            self.att_layer = AttLayer(self.head_num * self.head_dim, 128)
            self.classifier = nn.Linear(self.head_num * self.head_dim, self.num_classes)

    def cal_entropy(self, topic_weight):
        epsilon = 1e-6
        entropy_sum = torch.sum(-topic_weight * torch.log(epsilon + topic_weight), -1)  # (bs, n_heads)
        p_head = self.head_num * torch.softmax(1 / (epsilon + entropy_sum), -1)  # (bs, q_length, n_heads)
        return p_head.unsqueeze(-1)

    def extract_topic(self, topic_vector, input_feat):
        if self.variant_name == "distribute_topic":
            topic_vector = self.token_weight(topic_vector)  # (N, S, H)
            topic_vector = topic_vector.transpose(1, 2)     # (N, H, S)
        else:
            topic_vector = topic_vector.view(-1, topic_vector.shape[1], self.head_num, self.head_dim)
            topic_vector = topic_vector.transpose(1, 2)  # (N, H, S, D)
            topic_vector = self.token_weight(topic_vector).squeeze(-1)  # (N, H, S)
        if self.entropy_method == "without_softmax":
            return topic_vector
        mask = input_feat["mask"].expand(self.head_num, topic_vector.size(0), -1).transpose(0, 1) == 0
        topic_weight = topic_vector.masked_fill(mask, -1e9)
        topic_weight = torch.softmax(topic_weight, dim=-1)  # (N, H, S)
        if self.entropy_method == "weight_entropy":
            topic_weight = topic_weight * self.cal_entropy(topic_weight)
        return topic_weight

    def forward(self, input_feat, inputs_embeds=None, return_attention=False, **kwargs):
        input_feat["embedding"] = input_feat.get("embedding", inputs_embeds)
        embedding = self.embedding_layer(input_feat)

        if self.encoder_name == "MHA":
            hidden_score, _ = self.sentence_encoder(embedding, embedding, embedding)
        else:
            mask = input_feat["data"].bool().float()
            hidden_score = self.sentence_encoder(embedding, mask)  # (N, S, H*D)
        if self.pooling == "additive":
            output, topic_weight = self.att_layer(hidden_score)
        else:
            topic_weight = self.extract_topic(hidden_score, input_feat)
            # multiply topic weight with embedding
            # topic_weight = topic_weight * input_feat["mask"].unsqueeze(1)  # (N, H, S)
            output = torch.matmul(topic_weight, embedding)  # (N, H, E)

            if self.entropy_method == "after_matmul":
                output = output * self.cal_entropy(topic_weight)
            if self.variant_name == "additive":
                output, document_weight = self.projection(output)  # (N, E)
            else:
                output = self.projection(output).squeeze(-1)  # (N, E)
            if self.entropy_method == "after_projection":
                output = output * self.cal_entropy(topic_weight).squeeze(-1)
        self.return_attention = return_attention
        output = self.classify_layer(output, topic_weight)
        if self.return_entropy:
            entropy_sum = torch.sum(-topic_weight * torch.log(1e-6 + topic_weight)) / self.head_num
            output = output + (entropy_sum,)
        return output


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
