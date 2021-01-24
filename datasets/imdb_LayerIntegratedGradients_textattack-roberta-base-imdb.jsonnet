{
    "dataset": {
        "name": "imdb",
        "config": {
            "input_keys": [
                "input_ids",
                "attention_mask",
                "token_type_ids",
                "special_tokens_mask",
                "id"
            ]
        }
    },
    "explainer": {
        "name": "LayerIntegratedGradients",
        "config": {
            "internal_batch_size": 1,
            "n_samples": 25,
        }
    },
    "model": {
        "name": "textattack/roberta-base-imdb",
        "config": {
            "max_length": 512
        }
    }
}