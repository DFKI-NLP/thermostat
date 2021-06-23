CEN = ["contradiction", "entailment", "neutral"]
CNE = ["contradiction", "neutral", "entailment"]
ECN = ["entailment", "contradiction", "neutral"]
ENC = ["entailment", "neutral", "contradiction"]  # Legacy / Default from HF datasets
NCE = ["neutral", "contradiction", "entailment"]
NEC = ["neutral", "entailment", "contradiction"]

LC_MAP = {
    "multi_nli-albert": CEN,
    "multi_nli-bert": CEN,
    "multi_nli-roberta": CNE,
    "multi_nli-xlnet": CEN,
    "xnli-albert": CEN,
    "xnli-bert": CEN,
    "xnli-roberta": CNE,
    "xnli-xlnet": CEN,
}


def get_label_names(config_name):
    cn_dataset_model = '-'.join(config_name.split('-')[:2])
    if cn_dataset_model in LC_MAP.keys():
        return LC_MAP[cn_dataset_model]
