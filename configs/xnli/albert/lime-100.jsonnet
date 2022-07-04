{
    "path": "$HOME/experiments/thermostat",
    "device": "cuda",
    "dataset": {
        "name": "xnli",
        "subset": "en",
        "split": "test",
        "text_field": ["premise", "hypothesis"],
        "columns": ['input_ids', 'attention_mask', 'token_type_ids', 'special_tokens_mask', 'labels'],
        "batch_size": 1,
        "root_dir": "$HOME/experiments/thermostat/datasets",
    },
    "explainer": {
        "name": "LimeBase",
        "internal_batch_size": 1,
        "n_samples": 100,
        "mask_prob": 0.3,
    },
    "model": {
        "name": "prajjwal1/albert-base-v2-mnli",
        "mode_load": "hf",
        "path_model": null,
        "tokenization": {
            "max_length": 512,
            "padding": "max_length",
            "return_tensors": "np",
            "truncation": true,
            "special_tokens_mask": true,
        }
    },
    "visualization": {
        "columns": ["attributions", "predictions", "input_ids", "label"],
        "gamma": 2.0,
        "normalize": true,
    }
}
