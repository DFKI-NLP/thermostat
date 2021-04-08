# Thermostat
A collection of pre-computed heatmaps (also referred to as saliency maps, or visualization of attribution scores or feature importance scores) of natural language data sets to download and experiment with. It combines explainability methods from the `captum` library with Hugging Face's `datasets` and `transformers` (pre-trained language models).

In the area of explainable / interpretable natural language processing, applying explainability methods like Integrated Gradients or Shapley Value Sampling can be costly - in terms of both time and resources - to compute for whole data sets. This repository offers downloads of the heatmaps of these standard datasets for linguistic analysis.

## Datasets

Name | 🤗 | Task | #Cls
--- | --- | --- | ---
IMDb | [`imdb`](https://huggingface.co/datasets/viewer/?dataset=imdb) | Sentiment Analysis | 2
SST-2 | [`glue`](https://huggingface.co/datasets/viewer/?dataset=glue&config=sst2) | Sentiment Analysis | 2
SNLI | [`snli`](https://huggingface.co/datasets/viewer/?dataset=snli) | Textual Entailment | 3
QQP | [`glue`](https://huggingface.co/datasets/viewer/?dataset=glue&config=qqp) | Paraphrase Identification | 2
[PAWS](https://github.com/google-research-datasets/paws) | [`paws`](https://huggingface.co/datasets/viewer/?dataset=paws&config=labeled_final) | Paraphrase Identification | 2
[X-Stance](https://github.com/ZurichNLP/xstance) | [`x_stance`](https://huggingface.co/datasets/viewer/?dataset=x_stance) | Stance Detection | 2
[HANS](https://github.com/tommccoy1/hans) | [`hans`](https://huggingface.co/datasets/viewer/?dataset=hans) | Textual Entailment | 2
TREC | [`trec`](https://huggingface.co/datasets/viewer/?dataset=trec) | Question Classification | 6
SQuAD 1.1 | [`squad`](https://huggingface.co/datasets/viewer/?dataset=squad) | Question Answering | -

## Downstream Models
Name | 🤗
--- | ---
BERT base (uncased) | [`bert-base-uncased`](https://huggingface.co/bert-base-uncased)
RoBERTa base fine-tuned on IMDb | [`textattack/roberta-base-imdb`](https://huggingface.co/textattack/roberta-base-imdb)

## Explainers
Name | captum
--- | ---
Layer Integrated Gradients | [`.attr.LayerIntegratedGradients`](https://captum.ai/api/layer.html#layer-integrated-gradients)
Shapley Value Sampling | [`.attr.ShapleyValueSampling`](https://captum.ai/api/shapley_value_sampling.html)
LIME | [`.attr.LimeBase`](https://captum.ai/api/lime.html)



## Usage

**OUTDATED!**

```python
from datasets import load_dataset
from thermostat import load_map

imdb_train_data = load_dataset('imdb', split='train')
imdb_ig_bert_map = load_map(dataset=imdb_train_data,
                            explainer='LayerIntegratedGradients',
                            model='bert-base-uncased')
imdb_ig_bert_map.visualize()
```


## Config files

**OUTDATED!**

jsonnet config files have the following naming convention:
`<DATASET_ID>_<EXPLAINER_ID>_<MODEL_ID>.jsonnet` where
* `<DATASET_ID>` corresponds to a dataset (from `datasets` package by default, but can be any other locally stored dataset),
* `<EXPLAINER_ID>` corresponds to an explainability method (usually provided through the `captum` package) and
* `<MODEL_ID>` corresponds to a model (from `transformers` package by default)

The contents of a `imdb_LayerIntegratedGradients_bert-base-cased.jsonnet` config are:

```jsonnet
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
            "n_samples": 25,
            "internal_batch_size": 1
        }
    },
    "model": {
        "name": "bert-base-cased",
        "config": {
            "max_length": 512
        }
    }
}
```