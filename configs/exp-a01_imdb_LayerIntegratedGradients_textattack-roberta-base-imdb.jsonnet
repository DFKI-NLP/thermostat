{
    "path": "$HOME/experiments/thermometer/exp-a01",
    "dataset": {
        "config": "./configs/download/imdb.jsonnet",
        "split": "test",
    },
    "explainer": {
        "name": "LayerIntegratedGradients",
        "internal_batch_size": 1,
        "n_samples": 5,
        "early_stopping": -1,
    },
    "model": {
        "name": "textattack/roberta-base-imdb",
        "mode_load": "hf",
        "path_model": null,
        "tokenizer": "./configs/preprocess/roberta-base.jsonnet",
    }
}