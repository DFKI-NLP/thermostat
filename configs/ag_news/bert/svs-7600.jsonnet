{
    "path": "$HOME/experiments/thermostat",
    "device": "cuda",
    "dataset": {
        "name": "ag_news",
        "split": "test",
        "start": 3800,
        "columns": ['input_ids', 'attention_mask', 'token_type_ids', 'special_tokens_mask', 'labels'],
        "batch_size": 1,
        "root_dir": "$HOME/experiments/thermostat/datasets",
    },
    "explainer": {
        "name": "ShapleyValueSampling",
        "internal_batch_size": 1,
        "n_samples": 25,
    },
    "model": {
        "name": "textattack/bert-base-uncased-ag-news",
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
        "columns": ["attributions", "predictions", "input_ids", "labels"],
        "gamma": 2.0,
        "normalize": true,
    }
}
