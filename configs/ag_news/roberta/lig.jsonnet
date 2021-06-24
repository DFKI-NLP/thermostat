{
    "path": "$HOME/experiments/thermostat",
    "device": "cuda",
    "dataset": {
        "name": "ag_news",
        "split": "test",
        "columns": ['input_ids', 'attention_mask', 'special_tokens_mask', 'labels'],
        "batch_size": 1,
        "root_dir": "$HOME/experiments/thermostat/datasets",
    },
    "explainer": {
        "name": "LayerIntegratedGradients",
        "internal_batch_size": 1,
        "n_samples": 25,
    },
    "model": {
        "name": "textattack/roberta-base-ag-news",
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
