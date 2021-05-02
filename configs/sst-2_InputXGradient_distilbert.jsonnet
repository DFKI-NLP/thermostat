{
    "path": "$HOME/experiments/thermostat",
    "device": "cuda",
    "dataset": {
        "name": "glue",
        "subset": "sst2",
        "split": "test",
        "text_field": "sentence",
        "columns": ['input_ids', 'special_tokens_mask', 'attention_mask', 'labels'],
        "batch_size": 1,
        "root_dir": "$HOME/experiments/thermostat/datasets",
    },
    "explainer": {
        "name": "InputXGradient",
        "internal_batch_size": 1,
    },
    "model": {
        "name": "distilbert-base-uncased-finetuned-sst-2-english",
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
        "gamma": 2.0,
        "normalize": true,
    }
}