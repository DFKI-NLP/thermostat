{
    "path": "$HOME/experiments/thermostat/imdb_LayerIntegratedGradients_textattack-roberta-base-imdb",
    "dataset": {
        "name": "imdb",
        "split": "test",
        "columns": ['input_ids', 'special_tokens_mask', 'attention_mask', 'labels'],
        "batch_size": 2,
        "root_dir": "$HOME/experiments/thermometer/datasets",
    },
    "explainer": {
        "name": "LayerIntegratedGradients",
        "internal_batch_size": 2,
        "n_samples": 5,
        "early_stopping": -1,
    },
    "model": {
        "name": "textattack/roberta-base-imdb",
        "mode_load": "hf",
        "path_model": null,
        "tokenizer": {
            "max_length": 512,
            "padding": "max_length",
            "return_tensors": "np",
            "truncation": true,
            "special_tokens_mask": true,
        }
    },
    "visualization": {
        "columns": ["attributions", "predictions", "input_ids", "labels"],
    }
}