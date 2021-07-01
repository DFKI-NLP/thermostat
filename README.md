# Thermostat 🌡️

### **Thermostat** is a large collection of NLP model explanations and accompanying analysis tools.

* Combines explainability methods from the `captum` library with Hugging Face's `datasets` and `transformers`.
* Mitigates repetitive execution of common experiments in Explainable NLP and thus reduces the environmental impact and financial roadblocks.
* Increases comparability and replicability of research.
* Reduces the implementational burden.


## Usage

Downloading a dataset requires just two lines of code:

```python
import thermostat
data = thermostat.load("imdb-bert-lig")
```

Thermostat datasets always consist of three basic coordinates: Dataset, Model, and Explainer. In this example, the dataset is IMDb (sentiment analysis of movie reviews), the model is a BERT model fine-tuned on the IMDb data, the explanations are generated using a (Layer) Integrated Gradients explainer.

`data` then contains the following columns/features:
* `attributions` (the attributions for each token for each data point; type: List of floats)
* `idx` (the index of the instance in the dataset)
* `input_ids` (the token IDs of the original dataset; type: List of ints)
* `label` (the label of the original dataset; type: int)
* `predictions` (the class logits of the classifier/downstream model; type: List of floats)  

![instance-contents](figures/instance-contents.png)


Additionally,
```python
print(data)
```
... provides the actual names of the dataset, the explainer and the model:
```
IMDb dataset, BERT model, Layer Integrated Gradients explanations
Explainer: LayerIntegratedGradients
Model: textattack/bert-base-uncased-imdb
Dataset: imdb
```


### Visualizing attributions as a heatmap

```python
import thermostat

unit = thermostat.load("imdb-bert-lgxa")[20]
unit.render()
```

Refactoring done. More info to be added soon. Please refer to the demo.ipynb Notebook.

Meanwhile, here's the now deprecated HTML visualization:

![heatmap-html](figures/heatmap-html.png)


### Get simple tuple-based heatmap

```python
import thermostat

unit = thermostat.load("imdb-bert-lgxa")[20]
print(unit.explanation)
```

Refactoring done. More info to be added soon. Please refer to the demo.ipynb Notebook.


---

## Explainers
Name | captum implementation | Parameters
--- | --- | ---
Layer Gradient x Activation (`lgxa`) | [`.attr.LayerGradientXActivation`](https://captum.ai/api/layer.html#layer-gradient-x-activation) |
Layer Integrated Gradients (`lig`) | [`.attr.LayerIntegratedGradients`](https://captum.ai/api/layer.html#layer-integrated-gradients) | # samples = 25
LIME (`lime`) | [`.attr.LimeBase`](https://captum.ai/api/lime.html) | # samples = 25, <br>mask prob = 0.3
LIME <br>(Coming soon) | [`.attr.Lime`](https://captum.ai/api/lime.html) | # samples = 100
Occlusion (`occ`) | [`.attr.Occlusion`](https://captum.ai/api/occlusion.html) | sliding window = 3
Shapley Value Sampling (`svs`) | [`.attr.ShapleyValueSampling`](https://captum.ai/api/shapley_value_sampling.html) | # samples = 25



## Datasets + Models

![Overview](figures/overview_v1.png)

✅ = Dataset is downloadable  
⏏️ = Dataset is finished, but not uploaded yet  
🔄 = Currently running on cluster (x n = number of jobs/screens)  
⚠️ = Issue  

### IMDb

[`imdb`](https://huggingface.co/datasets/viewer/?dataset=imdb) is a sentiment analysis dataset with 2 classes (`pos` and `neg`). The available split is the `test` subset containing 25k examples.  
Example configuration: `imdb-xlnet-lig`

Name | 🤗 | `lgxa` | `lig` | `lime` | `occ` | `svs`  
--- | --- | --- | --- | --- | --- | ---
ALBERT (`albert`) | [`textattack/albert-base-v2-imdb`](https://huggingface.co/textattack/albert-base-v2-imdb) | ✅ | ✅ | ✅ | ✅ | ✅
BERT (`bert`) | [`textattack/bert-base-uncased-imdb`](https://huggingface.co/textattack/bert-base-uncased-imdb) | ✅ | ✅ | ✅ | ✅ | ✅
ELECTRA (`electra`) | [`monologg/electra-small-finetuned-imdb`](https://huggingface.co/monologg/electra-small-finetuned-imdb) | ✅ | ✅ | ✅ | ✅ | ✅
RoBERTa (`roberta`) | [`textattack/roberta-base-imdb`](https://huggingface.co/textattack/roberta-base-imdb) | ✅ | ✅ | ✅ | ✅ | ✅
XLNet (`xlnet`) | [`textattack/xlnet-base-cased-imdb`](https://huggingface.co/textattack/xlnet-base-cased-imdb) | ✅ | ✅ | ✅ | ✅ | ✅


### MultiNLI

[`multi_nli`](https://huggingface.co/datasets/viewer/?dataset=multi_nli) is a textual entailment dataset. The available split is the `validation_matched` subset containing 9815 examples.  
Example configuration: `multi_nli-roberta-lime`

Name | 🤗 | `lgxa` | `lig` | `lime` | `occ` | `svs`
--- | --- | --- | --- | --- | --- | ---
ALBERT (`albert`) | [`prajjwal1/albert-base-v2-mnli`](https://huggingface.co/prajjwal1/albert-base-v2-mnli) | ✅ | ✅ | ✅ | ✅ | 🔄x3/3
BERT (`bert`) | [`textattack/bert-base-uncased-MNLI`](https://huggingface.co/textattack/bert-base-uncased-MNLI) | ✅ | ✅ | ✅ | ✅ | ✅
ELECTRA (`electra`) | [`howey/electra-base-mnli`](https://huggingface.co/howey/electra-base-mnli) | ✅ | ✅ | ✅ | ✅ | ✅
RoBERTa (`roberta`) | [`textattack/roberta-base-MNLI`](https://huggingface.co/textattack/roberta-base-MNLI) | ✅ | ✅ | ✅ | ✅ | ✅
XLNet (`xlnet`) | [`textattack/xlnet-base-cased-MNLI`](https://huggingface.co/textattack/xlnet-base-cased-MNLI) | ✅ | ✅ | ✅ | ✅ | ✅


### XNLI

[`xnli`](https://huggingface.co/datasets/viewer/?dataset=xnli) is a textual entailment dataset. It provides the test set of MultiNLI through the "en" configuration. The fine-tuned models used here are the same as the MultiNLI ones. The available split is the `test` subset containing 5010 examples.  
Example configuration: `xnli-roberta-lime`

Name | 🤗 | `lgxa` | `lig` | `lime` | `occ` | `svs`
--- | --- | --- | --- | --- | --- | ---
ALBERT (`albert`) | [`prajjwal1/albert-base-v2-mnli`](https://huggingface.co/prajjwal1/albert-base-v2-mnli) | ✅ | ✅ | ✅ | ✅ | 🔄
BERT (`bert`) | [`textattack/bert-base-uncased-MNLI`](https://huggingface.co/textattack/bert-base-uncased-MNLI) | ✅ | ✅ | ✅ | ✅ | ✅
ELECTRA (`electra`) | [`howey/electra-base-mnli`](https://huggingface.co/howey/electra-base-mnli) | ✅ | ✅ | ✅ | ✅ | ✅
RoBERTa (`roberta`) | [`textattack/roberta-base-MNLI`](https://huggingface.co/textattack/roberta-base-MNLI) | ✅ | ✅ | ✅ | ✅ | 🔄
XLNet (`xlnet`) | [`textattack/xlnet-base-cased-MNLI`](https://huggingface.co/textattack/xlnet-base-cased-MNLI) | ✅ | ✅ | ✅ | ✅ | 🔄


### AG News

[`ag_news`](https://huggingface.co/datasets/viewer/?dataset=ag_news) is a news topic classification dataset. The available split is the `test` subset containing 7600 examples.  
Example configuration: `ag_news-albert-svs`

Name | 🤗 | `lgxa` | `lig` | `lime` | `occ` | `svs`
--- | --- | --- | --- | --- | --- | ---
ALBERT (`albert`) | [`textattack/albert-base-v2-ag-news`](https://huggingface.co/textattack/albert-base-v2-ag-news) | ✅ | ✅ | ✅ | ✅ | ✅
BERT (`bert`) | [`textattack/bert-base-uncased-ag-news`](https://huggingface.co/textattack/bert-base-uncased-ag-news) | ✅ | ✅ | ✅ | ✅ | ✅
RoBERTa (`roberta`) | [`textattack/roberta-base-ag-news`](https://huggingface.co/textattack/roberta-base-ag-news) | ✅ | ✅ | ✅ | ✅ | ✅


### Contribute a dataset

New explanation datasets must follow the JSONL format and include the five fields `attributions`, `idx`, `input_ids`, `label` and `predictions` as described above in "Usage".

The metadata will then be provided via a [`ThermostatConfig` entry in `builder_configs`](https://github.com/nfelnlp/thermostat/blob/c877ccf9155d42ec8a820ba9789cc84b8eb5f076/src/thermostat/data/thermostat_configs.py#L158).

Necessary fields include...
* `name` : The unique identifier string with the three coordinates `<DATASET>-<MODEL>-<EXPLAINER>`
* `dataset` : The full name of the dataset, usually follows the naming convention in `datasets`, e.g. `"imdb"`
* `explainer` : The full name of the explainer, usually follows the naming convention in `captum`, e.g. `"LayerIntegratedGradients"`
* `model` : The full name of the model, usually follows the naming convention in `transformers`, e.g. `"textattack/bert-base-uncased-imdb"`
* `label_column` : The name of the column in the JSONL file that contains the label, usually `"label"`
* `label_classes` : The list of label names or classes, e.g. `["entailment", "neutral", "contradiction"]` for NLI datasets
* `text_column` : Either a string (if there is only one text column) or a list of strings that identify the column in the JSONL file that contains the text(s), e.g. `"text"` (IMDb) or `["premise", "hypothesis"]` (NLI)
* `description` : Should at least state the full names of the three coordinates, can optionally include more info such as hyperparameter choices
* `data_url` : The URL to the data storage, e.g. a Google Drive link


### Parse a [Hugging Face dataset](https://huggingface.co/datasets) that contains explanations

If you want to access the convenience functions of Thermostat for a non-Thermostat dataset that  

a) you have downloaded from Hugging Face datasets and already contains explanation data (following the format above) or   
b) is a local explanation dataset and loadable using `datasets.load_dataset`,  

you can wrap your data with the `Thermopack` class:

```python
import thermostat
from datasets import load_dataset
data = load_dataset('your_dataset')
thermostat.Thermopack(data)
```
