# Thermostat

> Explainability in NLP is becoming more important by the day and is getting explored on many different levels. In order to perform linguistic and statistical analysis of the output of explainability methods, expert knowledge is needed. It further induces both an implementational on the developer and a computational burden on the hardware. We propose **Thermostat**, a collection of heatmaps that can be downloaded by researchers and practitioners. This work is a dataset paper that combines popular text classification datasets and state-of-the-art language models with explainability methods commonly applied to NLP tasks. By offering pre-computed saliency maps of these datasets stemming from a variety of explainers, we help to prevent repetitive execution yielding similar results and therefore mitigate the environmental and financial cost.

**Thermostat** combines explainability methods from the `captum` library with Hugging Face's `datasets` and `transformers`.



## Explainers
Name | captum | Tested
--- | --- | ---
Layer Gradient x Activation (*GxA*) | [`.attr.LayerGradientXActivation`](https://captum.ai/api/layer.html#layer-gradient-x-activation) | ✅
Layer Integrated Gradients (*IG*) | [`.attr.LayerIntegratedGradients`](https://captum.ai/api/layer.html#layer-integrated-gradients) | ✅
Occlusion (*Occ*) | [`.attr.Occlusion`](https://captum.ai/api/occlusion.html) | ✅
Shapley Value Sampling (*SVS*) | [`.attr.ShapleyValueSampling`](https://captum.ai/api/shapley_value_sampling.html) | ✅
LIME | [`.attr.LimeBase`](https://captum.ai/api/lime.html) | ✅


## Datasets + Models


### IMDb

[`imdb`](https://huggingface.co/datasets/viewer/?dataset=imdb) is a sentiment analysis dataset with 2 classes (`pos` and `neg`).

Name | 🤗 | GxA | IG | Occ | SVS | LIME 
--- | --- | --- | --- | --- | --- | ---
ALBERT | [`textattack/albert-base-v2-imdb`](https://huggingface.co/textattack/albert-base-v2-imdb)
BERT | [`textattack/bert-base-uncased-imdb`](https://huggingface.co/textattack/bert-base-uncased-imdb) | ✅ | ✅ | ✅ | | ✅  
DistilBERT | [`textattack/distilbert-base-uncased-imdb`](https://huggingface.co/textattack/distilbert-base-uncased-imdb)
ELECTRA | [`monologg/electra-small-finetuned-imdb`](https://huggingface.co/monologg/electra-small-finetuned-imdb)
RoBERTa | [`textattack/roberta-base-imdb`](https://huggingface.co/textattack/roberta-base-imdb) | | | | |✅
XLNet | [`textattack/xlnet-base-cased-imdb`](https://huggingface.co/textattack/xlnet-base-cased-imdb)


### SST-2

SST-2 is a sentiment analysis dataset with 2 classes and part of the [`glue`](https://huggingface.co/datasets/viewer/?dataset=glue&config=sst2) benchmark.
There are no labels available for the test set.

Name | 🤗 | GxA | IG | Occ | SVS | LIME 
--- | --- | --- | --- | --- | --- | ---
ALBERT | [`textattack/albert-base-v2-SST-2`](https://huggingface.co/textattack/albert-base-v2-SST-2)
BERT | [`textattack/bert-base-uncased-SST-2`](https://huggingface.co/textattack/bert-base-uncased-SST-2) | ✅
DistilBERT | [`distilbert-base-uncased-finetuned-sst-2-english`](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english)
ELECTRA | [`howey/electra-base-sst2`](https://huggingface.co/howey/electra-base-sst2)
RoBERTa | [`textattack/roberta-base-SST-2`](https://huggingface.co/textattack/roberta-base-SST-2)
XLNet | [`textattack/xlnet-base-cased-SST-2`](https://huggingface.co/textattack/xlnet-base-cased-SST-2)


### MNLI

There are no labels available for the test set.

Name | 🤗 | GxA | IG | Occ | SVS | LIME 
--- | --- | --- | --- | --- | --- | ---
ALBERT | [`prajjwal1/albert-base-v2-mnli`](https://huggingface.co/prajjwal1/albert-base-v2-mnli)
BERT | [`textattack/bert-base-uncased-mnli`](https://huggingface.co/textattack/bert-base-uncased-MNLI)
DistilBERT | [`textattack/distilbert-base-uncased-mnli`](https://huggingface.co/textattack/distilbert-base-uncased-MNLI)
ELECTRA | [`howey/electra-base-mnli`](https://huggingface.co/howey/electra-base-mnli)
RoBERTa | [`textattack/roberta-base-mnli`](https://huggingface.co/textattack/roberta-base-MNLI)
XLNet | [`textattack/xlnet-base-cased-mnli`](https://huggingface.co/textattack/xlnet-base-cased-MNLI)



## Usage

Clone this repo and then from the top level directory (where the "thermostat" folder exists) you can call the following script:

```python
from datasets import load_dataset

# This will use the dataset script ("thermostat.py") in the "thermostat" directory
data = load_dataset("thermostat", "imdb-bert-lgxa", split="test")
# Print the first instance of 
# IMDb-BERT-LayerGradientXActivation attributions
print(data[0])

```

`data` then provides a subset with features/columns
* `attributions` (the attributions for each token for each data point; type: List of floats) : `[-0.18760254979133606, -0.0315956249833107, 0.04854373633861542, 0.00658783596009016, 0.017869707196950912,` ...
* `input_ids` (the token IDs of the original dataset; type: List of ints) :  `[101, 2092, 1010, 1045, 7166, 2000,` ...
* `label` (the label of the original dataset; type: int) : `1`
* `predictions` (the class logits of the classifier/downstream model; type: List of floats) : `[-3.4371631145477295, 4.042327404022217]`
* `dataset.description` provides the actual names of the dataset, the explainer and the model

### Visualizing attributions as a heatmap

Coming soon: `to_html` function.

### Tools for analysis

Coming soon:
* `to_words` function for casting `input_ids` back to tokens.

### Config files

jsonnet config files have the following naming convention:
`<DATASET_ID>/<EXPLAINER_ID>/<MODEL_ID>.jsonnet` where
* `<DATASET_ID>` corresponds to a dataset (from `datasets` package by default, but can be any other locally stored dataset),
* `<EXPLAINER_ID>` corresponds to an explainability method (usually provided through the `captum` package) and
* `<MODEL_ID>` corresponds to a model (from `transformers` package by default)


### Visualization adapters
Coming soon:
* Displacy
* ecco
* captum
