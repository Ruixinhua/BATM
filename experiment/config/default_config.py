default_configs = {
    "PretrainedBaseline": {
        "n_layers": 1,
    },
    "TextCNNClassifyModel": {
        "num_filters": 100, "filter_sizes": (2, )
    },
    "NRMSNewsEncoderModel": {
        "variant_name": "base"
    },
    "BiAttentionClassifyModel": {
        "head_num": None, "head_dim": 20, "return_entropy": False, "alpha": 0.01, "n_layers": 1, "variant_name": "base",
    },
    "TopicExtractorClassifyModel": {
        "head_num": None, "head_dim": 20, "return_entropy": False, "alpha": 0.01, "n_layers": 1
    },
    "FastformerClassifyModel": {
        "embedding_dim": 300, "n_layers": 2, "hidden_act": "gelu", "head_num": 15, "type_vocab_size": 2,
        "vocab_size": 100000, "layer_norm_eps": 1e-12, "initializer_range": 0.02, "pooler_type": "weightpooler",
        "enable_fp16": "False"
    }
}


def arch_default_config(arch_type: str):
    default_config = {"type": arch_type}
    default_config.update(default_configs[arch_type])
    return default_config
