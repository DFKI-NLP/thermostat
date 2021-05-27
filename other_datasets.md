### SST-2

SST-2 is a sentiment analysis dataset with 2 classes and part of the [`glue`](https://huggingface.co/datasets/viewer/?dataset=glue&config=sst2) benchmark.
There are no labels available for the test set.

Name | 🤗 | `lgxa` | `lig` | `lime` | `occ` | `svs` 
--- | --- | --- | --- | --- | --- | ---
ALBERT (`albert`) | [`textattack/albert-base-v2-SST-2`](https://huggingface.co/textattack/albert-base-v2-SST-2)
BERT (`bert`) | [`textattack/bert-base-uncased-SST-2`](https://huggingface.co/textattack/bert-base-uncased-SST-2) |
DistilBERT (`distilbert`) | [`distilbert-base-uncased-finetuned-sst-2-english`](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english) | - | - | - | - | -
ELECTRA (`electra`) | [`howey/electra-base-sst2`](https://huggingface.co/howey/electra-base-sst2)
RoBERTa (`roberta`) | [`textattack/roberta-base-SST-2`](https://huggingface.co/textattack/roberta-base-SST-2)
XLNet (`xlnet`) | [`textattack/xlnet-base-cased-SST-2`](https://huggingface.co/textattack/xlnet-base-cased-SST-2)


### QQP

QQP is a paraphrase identification dataset of two classes and part of the [`glue`](https://huggingface.co/datasets/viewer/?dataset=glue&config=qqp) benchmark.

Name | 🤗 | Tested
--- | --- | ---
ALBERT | [`textattack/albert-base-v2-qqp`](https://huggingface.co/textattack/albert-base-v2-QQP)
BERT | [`textattack/bert-base-uncased-qqp`](https://huggingface.co/textattack/bert-base-uncased-QQP)
DistilBERT | [`textattack/distilbert-base-uncased-qqp`](https://huggingface.co/textattack/distilbert-base-uncased-QQP)
ELECTRA | [`howey/electra-base-qqp`](https://huggingface.co/howey/electra-base-qqp)
XLNet | [`textattack/xlnet-base-cased-qqp`](https://huggingface.co/textattack/xlnet-base-cased-QQP)


### TREC

[`trec`](https://huggingface.co/datasets/viewer/?dataset=trec) is a question classification dataset with 6 classes.

Name | 🤗 | Tested
--- | --- | ---
BERT | [`aychang/bert-base-cased-trec-coarse`](https://huggingface.co/aychang/bert-base-cased-trec-coarse)
DistilBERT | [`aychang/distilbert-base-cased-trec-coarse`](https://huggingface.co/aychang/distilbert-base-cased-trec-coarse)

