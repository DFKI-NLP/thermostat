{
    "dataset": {
        "name": "imdb",
        "split": "test",
    },
    "explainer": {
        "name": "LayerIntegratedGradients",
        "internal_batch_size": 1,
        "n_samples": 25,
    },
    "model": {
        "name": "textattack/roberta-base-imdb",
        "tokenizer": "roberta-base",
    }
}