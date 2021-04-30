# Thermostat

> Explainability in NLP is becoming more important by the day and is getting explored on many different levels. In order to perform linguistic and statistical analysis of the output of explainability methods, expert knowledge is needed. It further induces both an implementational on the developer and a computational burden on the hardware. We propose **Thermostat**, a collection of heatmaps that can be downloaded by researchers and practitioners. This work is a dataset paper that combines popular text classification datasets and state-of-the-art language models with explainability methods commonly applied to NLP tasks. By offering pre-computed saliency maps of these datasets stemming from a variety of explainers, we help to prevent repetitive execution yielding similar results and therefore mitigate the environmental and financial cost.

**Thermostat** combines explainability methods from the `captum` library with Hugging Face's `datasets` and `transformers`.


## Datasets

Name | 🤗 | Task | #Cls | Tested
--- | --- | --- | --- | ---
IMDb | [`imdb`](https://huggingface.co/datasets/viewer/?dataset=imdb) | Sentiment Analysis | 2 | ✅
SST-2 | [`glue`](https://huggingface.co/datasets/viewer/?dataset=glue&config=sst2) | Sentiment Analysis | 2
SNLI | [`snli`](https://huggingface.co/datasets/viewer/?dataset=snli) | Textual Entailment | 3 | ✅
QQP | [`glue`](https://huggingface.co/datasets/viewer/?dataset=glue&config=qqp) | Paraphrase Identification | 2
[PAWS](https://github.com/google-research-datasets/paws) | [`paws`](https://huggingface.co/datasets/viewer/?dataset=paws&config=labeled_final) | Paraphrase Identification | 2
[X-Stance](https://github.com/ZurichNLP/xstance) | [`x_stance`](https://huggingface.co/datasets/viewer/?dataset=x_stance) | Stance Detection | 2
[HANS](https://github.com/tommccoy1/hans) | [`hans`](https://huggingface.co/datasets/viewer/?dataset=hans) | Textual Entailment | 2
TREC | [`trec`](https://huggingface.co/datasets/viewer/?dataset=trec) | Question Classification | 6

## Downstream Models
Name | 🤗 | Tested
--- | --- | ---
BERT base (uncased) | [`bert-base-uncased`](https://huggingface.co/bert-base-uncased) | ✅
RoBERTa base fine-tuned on IMDb | [`textattack/roberta-base-imdb`](https://huggingface.co/textattack/roberta-base-imdb) | ✅
T5
XLNet
ConvBERT

## Explainers
Name | captum | Tested
--- | --- | ---
Layer Integrated Gradients | [`.attr.LayerIntegratedGradients`](https://captum.ai/api/layer.html#layer-integrated-gradients) | ✅
Shapley Value Sampling | [`.attr.ShapleyValueSampling`](https://captum.ai/api/shapley_value_sampling.html) | ✅
LIME | [`.attr.LimeBase`](https://captum.ai/api/lime.html) | ✅
KernelSHAP
Occlusion
Gradient x Input
Guided Backprop



## Usage

More info soon.

### Downloading a heatmap



### Tools for analysis


### Config files

jsonnet config files have the following naming convention:
`<DATASET_ID>_<EXPLAINER_ID>_<MODEL_ID>.jsonnet` where
* `<DATASET_ID>` corresponds to a dataset (from `datasets` package by default, but can be any other locally stored dataset),
* `<EXPLAINER_ID>` corresponds to an explainability method (usually provided through the `captum` package) and
* `<MODEL_ID>` corresponds to a model (from `transformers` package by default)


### Visualization adapters
* Displacy
* ecco
* captum


## Contribute
