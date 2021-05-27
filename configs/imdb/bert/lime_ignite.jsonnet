{
    "path": "$HOME/experiments/thermostat",
    "device": "cuda",
    "dataset": {
        "name": "imdb",
        "split": "test",
        "columns": ['input_ids', 'attention_mask', 'token_type_ids', 'labels'],
        "batch_size": 1,
        "root_dir": "$HOME/experiments/thermostat/datasets",
    },
    "explainer": {
        "name": "LimeBase",
        "internal_batch_size": 1,
        "n_samples": 25,
        "mask_prob": 0.3,
    },
    "model": {
        "name": "bert-base-cased",
        "mode_load": "ignite",
        "path_model": "$HOME/models/thermostat/2021-03-08-experiment-imdb/models/2021-03-08-16-13-44.bert-base-cased.huggingface.imdb_model_f1=0.9324.pt",
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
        "gamma": 2.0,
        "normalize": true,
    }
}