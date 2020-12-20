# Thermometer
A collection of pre-computed heatmaps (also referred to as saliency maps, or visualization of attribution scores or feature importance scores) of natural language data sets to download and experiment with. It combines explainability methods from the `captum` library with Hugging Face's `datasets` and `transformers` (pre-trained language models).

In the area of explainable / interpretable natural language processing, applying explainability methods like Integrated Gradients or Shapley Value Sampling can be costly - in terms of both time and resources - to compute for whole data sets. This repository offers downloads of these heatmaps for linguistic analysis.

## Datasets

Name | Task | Type
--- | --- | ---
IMDb | Sentiment Analysis | Binary Classification
SST-2 | Sentiment Analysis | Binary Classification
QQP | Paraphrase Identification | Binary Classification
[PAWS](https://github.com/google-research-datasets/paws) | Paraphrase Identification | Binary Classification
[x-stance](https://github.com/ZurichNLP/xstance) | Stance Detection | Binary Classification
[HANS](https://github.com/tommccoy1/hans) | Textual Entailment | Multi-class Classification
SQuAD 1.1 | Question Answering |

## Downstream Models
* BERT base (uncased) : [`bert-base-uncased`](https://huggingface.co/bert-base-uncased)
* T5 (small) : [`t5-small`](https://huggingface.co/t5-small)

## Explainers
* Layer Integrated Gradients : [`captum.attr.LayerIntegratedGradients`](https://captum.ai/api/layer.html#layer-integrated-gradients)
* Shapley Value Sampling : [`captum.attr.ShapleyValueSampling`](https://captum.ai/api/shapley_value_sampling.html)




## Usage

```python
from datasets import load_dataset
from thermometer import load_map

imdb_train_data = load_dataset('imdb', split='train')
imdb_ig_bert_map = load_map(dataset=imdb_train_data,
                            explainer='LayerIntegratedGradients',
                            model='bert-base-uncased')
imdb_ig_bert_map.visualize()
```


## Example config

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