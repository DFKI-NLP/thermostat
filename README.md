# Thermostat

> Explainability in NLP is becoming more important by the day and is getting explored on many different levels. In order to perform linguistic and statistical analysis of the output of explainability methods, expert knowledge is needed. It further induces both an implementational on the developer and a computational burden on the hardware. We propose **Thermostat**, a collection of heatmaps that can be downloaded by researchers and practitioners. This work is a dataset paper that combines popular text classification datasets and state-of-the-art language models with explainability methods commonly applied to NLP tasks. By offering pre-computed saliency maps of these datasets stemming from a variety of explainers, we help to prevent repetitive execution yielding similar results and therefore mitigate the environmental and financial cost.

**Thermostat** combines explainability methods from the `captum` library with Hugging Face's `datasets` and `transformers`.



## Explainers
Name | captum | Tested
--- | --- | ---
Layer Integrated Gradients | [`.attr.LayerIntegratedGradients`](https://captum.ai/api/layer.html#layer-integrated-gradients) | ✅
Shapley Value Sampling | [`.attr.ShapleyValueSampling`](https://captum.ai/api/shapley_value_sampling.html) | ✅
LIME | [`.attr.LimeBase`](https://captum.ai/api/lime.html) | ✅
KernelSHAP | [`.attr.KernelShap`](https://captum.ai/api/kernel_shap.html)
Occlusion | [`.attr.Occlusion`](https://captum.ai/api/occlusion.html)
Gradient x Input | [`.attr.InputXGradient`](https://captum.ai/api/input_x_gradient.html)
Guided Backprop | [`.attr.GuidedBackprop`](https://captum.ai/api/guided_backprop.html)


## Fine-tuned Models


### IMDb

[`imdb`](https://huggingface.co/datasets/viewer/?dataset=imdb) is a sentiment analysis dataset with 2 classes (`pos` and `neg`). ✅

Name | 🤗 | Tested
--- | --- | ---
ALBERT | [`textattack/albert-base-v2-imdb`](https://huggingface.co/textattack/albert-base-v2-imdb)
BERT | [`textattack/bert-based-uncased-imdb`](https://huggingface.co/textattack/bert-base-uncased-imdb)
BERT | [emp-exp bert-base-uncased](https://github.com/DFKI-NLP/emp-exp#download-our-data-models-and-logs) | ✅
DistilBERT | [`textattack/distilbert-base-uncased-imdb`](https://huggingface.co/textattack/distilbert-base-uncased-imdb)
RoBERTa | [`textattack/roberta-base-imdb`](https://huggingface.co/textattack/roberta-base-imdb) | ✅
T5 | [`t5-base-finetuned-imdb-sentiment`](https://huggingface.co/mrm8488/t5-base-finetuned-imdb-sentiment)
XLNet | [`textattack/xlnet-base-cased-imdb`](https://huggingface.co/textattack/xlnet-base-cased-imdb)


### SST-2

SST-2 is a sentiment analysis dataset with 2 classes and part of the [`glue`](https://huggingface.co/datasets/viewer/?dataset=glue&config=sst2) benchmark.
Name | 🤗 | Tested
--- | --- | ---
ALBERT | [`textattack/albert-base-v2-sst-2`](https://huggingface.co/textattack/albert-base-v2-SST-2)
BART | [`textattack/facebook-bart-large-sst-2`](https://huggingface.co/textattack/facebook-bart-large-SST-2)
BERT | [`textattack/bert-base-uncased-sst-2`](https://huggingface.co/textattack/bert-base-uncased-SST-2)
DistilBERT | [`distilbert-base-uncased-finetuned-sst-2-english`](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english)
RoBERTa | [`textattack/roberta-base-sst-2`](https://huggingface.co/textattack/roberta-base-SST-2)
XLNet | [`textattack/xlnet-base-cased-sst-2`](https://huggingface.co/textattack/xlnet-base-cased-SST-2)


### MNLI

Name | 🤗 | Tested
--- | --- | ---
BART | [`facebook/bart-large-mnli`](https://huggingface.co/facebook/bart-large-mnli)
BERT | [`textattack/bert-base-uncased-mnli`](https://huggingface.co/textattack/bert-base-uncased-MNLI)
DeBERTa | [`microsoft/deberta-xlarge-mnli`](https://huggingface.co/microsoft/deberta-xlarge-mnli)
DistilBERT | [`textattack/distilbert-base-uncased-mnli`](https://huggingface.co/textattack/distilbert-base-uncased-MNLI)
RoBERTa | [`textattack/roberta-base-mnli`](https://huggingface.co/textattack/roberta-base-MNLI)
XLNet | [`textattack/xlnet-base-cased-mnli`](https://huggingface.co/textattack/xlnet-base-cased-MNLI)


### SNLI

[`snli`](https://huggingface.co/datasets/viewer/?dataset=snli) is a textual entailment dataset with 3 classes (`entailment`, `neutral` and `contradiction`). ✅

Name | 🤗 | Tested
--- | --- | ---
ALBERT | [`textattack/albert-base-v2-snli`](https://huggingface.co/textattack/albert-base-v2-snli)
BERT | [`textattack/bert-base-uncased-snli`](https://huggingface.co/textattack/bert-base-uncased-snli)
DistilBERT | [`textattack/distilbert-base-cased-snli`](https://huggingface.co/textattack/distilbert-base-cased-snli)


### QQP

QQP is a paraphrase identification dataset of two classes and part of the [`glue`](https://huggingface.co/datasets/viewer/?dataset=glue&config=qqp) benchmark.

Name | 🤗 | Tested
--- | --- | ---
ALBERT | [`textattack/albert-base-v2-qqp`](https://huggingface.co/textattack/albert-base-v2-QQP)
BERT | [`textattack/bert-base-uncased-qqp`](https://huggingface.co/textattack/bert-base-uncased-QQP)
DistilBERT | [`textattack/distilbert-base-uncased-qqp`](https://huggingface.co/textattack/distilbert-base-uncased-QQP)
XLNet | [`textattack/xlnet-base-cased-qqp`](https://huggingface.co/textattack/xlnet-base-cased-QQP)


### TREC

[`trec`](https://huggingface.co/datasets/viewer/?dataset=trec) is a question classification dataset with 6 classes.

Name | 🤗 | Tested
--- | --- | ---
BERT | [`aychang/bert-base-cased-trec-coarse`](https://huggingface.co/aychang/bert-base-cased-trec-coarse)
DistilBERT | [`aychang/distilbert-base-cased-trec-coarse`](https://huggingface.co/aychang/distilbert-base-cased-trec-coarse)


### Other datasets

Currently, none of these datasets have fine-tuned models available in the transformers model hub.

Name | 🤗 | Task | #Cls | Tested
--- | --- | --- | --- | ---
[X-Stance](https://github.com/ZurichNLP/xstance) | [`x_stance`](https://huggingface.co/datasets/viewer/?dataset=x_stance) | Stance Detection | 2
[HANS](https://github.com/tommccoy1/hans) | [`hans`](https://huggingface.co/datasets/viewer/?dataset=hans) | Textual Entailment | 2



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
