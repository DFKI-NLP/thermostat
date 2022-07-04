import datasets


_VERSION = datasets.Version('1.0.2', '')


# Base arguments for any dataset
_BASE_KWARGS = dict(
    features={
        "attributions": "attributions",
        "predictions": "predictions",
        "input_ids": "input_ids",
    },
    citation="Coming soon.",
    url="https://github.com/DFKI-NLP/",
)


# Base arguments for AG News dataset
_AGNEWS_KWARGS = dict(
    dataset="ag_news",
    label_classes=["World", "Sports", "Business", "Sci/Tech"],
    label_column="label",
    text_column="text",
    **_BASE_KWARGS,
)
_AGNEWS_ALBERT_KWARGS = dict(
    model="textattack/albert-base-v2-ag-news",
    **_AGNEWS_KWARGS,
)
_AGNEWS_BERT_KWARGS = dict(
    model="textattack/bert-base-uncased-ag-news",
    **_AGNEWS_KWARGS,
)
_AGNEWS_ROBERTA_KWARGS = dict(
    model="textattack/roberta-base-ag-news",
    **_AGNEWS_KWARGS,
)

# Base arguments for IMDb dataset
_IMDB_KWARGS = dict(
    dataset="imdb",
    label_classes=["neg", "pos"],
    label_column="label",
    text_column="text",
    **_BASE_KWARGS,
)
_IMDB_ALBERT_KWARGS = dict(
    model="textattack/albert-base-v2-imdb",
    **_IMDB_KWARGS,
)
_IMDB_BERT_KWARGS = dict(
    model="textattack/bert-base-uncased-imdb",
    **_IMDB_KWARGS,
)
_IMDB_ELECTRA_KWARGS = dict(
    model="monologg/electra-small-finetuned-imdb",
    **_IMDB_KWARGS,
)
_IMDB_ROBERTA_KWARGS = dict(
    model="textattack/roberta-base-imdb",
    **_IMDB_KWARGS,
)
_IMDB_XLNET_KWARGS = dict(
    model="textattack/xlnet-base-cased-imdb",
    **_IMDB_KWARGS,
)

# Base arguments for MNLI dataset
_MNLI_KWARGS = dict(
    dataset="multi_nli",
    label_column="label",
    label_classes=["entailment", "neutral", "contradiction"],
    text_column=["premise", "hypothesis"],
    **_BASE_KWARGS,
)
_MNLI_ALBERT_KWARGS = dict(
    model="prajjwal1/albert-base-v2-mnli",
    **_MNLI_KWARGS,
)
_MNLI_BERT_KWARGS = dict(
    model="textattack/bert-base-uncased-MNLI",
    **_MNLI_KWARGS,
)
_MNLI_ELECTRA_KWARGS = dict(
    model="howey/electra-base-mnli",
    **_MNLI_KWARGS,
)
_MNLI_ROBERTA_KWARGS = dict(
    model="textattack/roberta-base-MNLI",
    **_MNLI_KWARGS,
)
_MNLI_XLNET_KWARGS = dict(
    model="textattack/xlnet-base-cased-MNLI",
    **_MNLI_KWARGS,
)

# Base arguments for XNLI dataset
_XNLI_KWARGS = dict(
    dataset="xnli",
    label_column="label",
    label_classes=["entailment", "neutral", "contradiction"],
    text_column=["premise", "hypothesis"],
    **_BASE_KWARGS,
)
_XNLI_ALBERT_KWARGS = dict(
    model="prajjwal1/albert-base-v2-mnli",
    **_XNLI_KWARGS,
)
_XNLI_BERT_KWARGS = dict(
    model="textattack/bert-base-uncased-MNLI",
    **_XNLI_KWARGS,
)
_XNLI_ELECTRA_KWARGS = dict(
    model="howey/electra-base-mnli",
    **_XNLI_KWARGS,
)
_XNLI_ROBERTA_KWARGS = dict(
    model="textattack/roberta-base-MNLI",
    **_XNLI_KWARGS,
)
_XNLI_XLNET_KWARGS = dict(
    model="textattack/xlnet-base-cased-MNLI",
    **_XNLI_KWARGS,
)


class ThermostatConfig(datasets.BuilderConfig):
    """ BuilderConfig for Thermostat """

    def __init__(
        self,
        explainer,
        model,
        dataset,
        features,
        label_column,
        label_classes,
        text_column,
        data_url,
        citation,
        url,
        **kwargs,
    ):
        super(ThermostatConfig, self).__init__(version=_VERSION, **kwargs)
        self.explainer = explainer
        self.model = model
        self.dataset = dataset
        self.features = features
        self.label_column = label_column
        self.label_classes = label_classes
        self.text_column = text_column
        self.data_url = data_url
        self.citation = citation
        self.url = url


builder_configs = [
    ThermostatConfig(
        name="ag_news-albert-lgxa",
        description="AG News dataset, ALBERT model, Layer Gradient x Activation explanations",
        explainer="LayerGradientXActivation",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/8XPC557ePpWCBQY/download",
        **_AGNEWS_ALBERT_KWARGS,
    ),
    ThermostatConfig(
        name="ag_news-albert-lig",
        description="AG News dataset, ALBERT model, Layer Integrated Gradients explanations",
        explainer="LayerIntegratedGradients",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/zS7mcMsdAp5ZENX/download",
        **_AGNEWS_ALBERT_KWARGS,
    ),
    ThermostatConfig(
        name="ag_news-albert-lime",
        description="AG News dataset, ALBERT model, LIME explanations",
        explainer="LimeBase",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/8SLyHdDgRk2pXSL/download",
        **_AGNEWS_ALBERT_KWARGS,
    ),
    # new change
    ThermostatConfig(
        name="ag_news-albert-lime-100",
        description="AG News dataset, ALBERT model, LIME explanations, 100 samples",
        explainer="LimeBase",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/W3GT4ZDT2BzR5mj/download",
        **_AGNEWS_ALBERT_KWARGS,
    ),
    # new change
    ThermostatConfig(
        name="ag_news-albert-occlusion",
        description="AG News dataset, ALBERT model, Occlusion explanations",
        explainer="Occlusion",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/Li9HwfKfFqjCQTM/download",
        **_AGNEWS_ALBERT_KWARGS,
    ),
    ThermostatConfig(
        name="ag_news-albert-svs",
        description="AG News dataset, ALBERT model, Shapley Value Sampling explanations",
        explainer="ShapleyValueSampling",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/GLppwQjeBTsLtTC/download",
        **_AGNEWS_ALBERT_KWARGS,
    ),
    # shap value inclusion
    ThermostatConfig(
        name="ag_news-albert-lds",
        description="AG News dataset, ALBERT model, Layer DeepLift Shap explanations",
        explainer="LayerDeepLiftShap",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/nzriaZHxniNtJBN/download",
        **_AGNEWS_ALBERT_KWARGS,
    ),
    ThermostatConfig(
        name="ag_news-albert-lgs",
        description="AG News dataset, ALBERT model, Layer Gradient Shap explanations",
        explainer="LayerGradientShap",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/EjCXxWCETQH9onj/download",
        **_AGNEWS_ALBERT_KWARGS,
    ),
    # shap value inclusion
    ThermostatConfig(
        name="ag_news-bert-lgxa",
        description="AG News dataset, BERT model, Layer Gradient x Activation explanations",
        explainer="LayerGradientXActivation",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/zn5mKsryyrX3e58/download",
        **_AGNEWS_BERT_KWARGS,
    ),
    ThermostatConfig(
        name="ag_news-bert-lig",
        description="AG News dataset, BERT model, Layer Integrated Gradients explanations",
        explainer="LayerIntegratedGradients",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/Qq3dR7sHsfX9JXZ/download",
        **_AGNEWS_BERT_KWARGS,
    ),
    ThermostatConfig(
        name="ag_news-bert-lime",
        description="AG News dataset, BERT model, LIME explanations",
        explainer="LimeBase",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/rW8MJyAjBGQxsK9/download",
        **_AGNEWS_BERT_KWARGS,
    ),
    # new change
    ThermostatConfig(
        name="ag_news-bert-lime-100",
        description="AG News dataset, BERT model, LIME explanations, 100 samples",
        explainer="LimeBase",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/FkSdXZPpN78HSHR/download",
        **_AGNEWS_BERT_KWARGS,
    ),
    # new change
    ThermostatConfig(
        name="ag_news-bert-occlusion",
        description="AG News dataset, BERT model, Occlusion explanations",
        explainer="Occlusion",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/Grf97s6bJwoZGyx/download",
        **_AGNEWS_BERT_KWARGS,
    ),
    ThermostatConfig(
        name="ag_news-bert-svs",
        description="AG News dataset, BERT model, Shapley Value Sampling explanations",
        explainer="ShapleyValueSampling",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/dCbgsjdW6b9pzo3/download",
        **_AGNEWS_BERT_KWARGS,
    ),
    # shap value inclusion
    ThermostatConfig(
        name="ag_news-bert-lds",
        description="AG News dataset, BERT model, Layer DeepLift Shap explanations",
        explainer="LayerDeepLiftShap",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/XQzBnRniEyC8NEF/download",
        **_AGNEWS_BERT_KWARGS,
    ),
    ThermostatConfig(
        name="ag_news-bert-lgs",
        description="AG News dataset, BERT model, Layer Gradient Shap explanations",
        explainer="LayerGradientShap",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/nHt4Ld4AKfbG2ft/download",
        **_AGNEWS_BERT_KWARGS,
    ),
    # shap value inclusion
    ThermostatConfig(
        name="ag_news-roberta-lgxa",
        description="AG News dataset, RoBERTa model, Layer Gradient x Activation explanations",
        explainer="LayerGradientXActivation",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/Kz9GrYrMZB4gp7E/download",
        **_AGNEWS_ROBERTA_KWARGS,
    ),
    ThermostatConfig(
        name="ag_news-roberta-lig",
        description="AG News dataset, RoBERTa model, Layer Integrated Gradients explanations",
        explainer="LayerIntegratedGradients",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/mpH8sT6tXDoG5qi/download",
        **_AGNEWS_ROBERTA_KWARGS,
    ),
    ThermostatConfig(
        name="ag_news-roberta-lime",
        description="AG News dataset, RoBERTa model, LIME explanations",
        explainer="LimeBase",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/qRgBtwfjaXceJoL/download",
        **_AGNEWS_ROBERTA_KWARGS,
    ),
    # new change
    ThermostatConfig(
        name="ag_news-roberta-lime-100",
        description="AG News dataset, RoBERTa model, LIME explanations, 100 samples",
        explainer="LimeBase",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/kFyjX2LqBdcW9bp/download",
        **_AGNEWS_ROBERTA_KWARGS,
    ),
    # new change
    ThermostatConfig(
        name="ag_news-roberta-occlusion",
        description="AG News dataset, RoBERTa model, Occlusion explanations",
        explainer="Occlusion",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/78aZttqxKQNdW6J/download",
        **_AGNEWS_ROBERTA_KWARGS,
    ),
    ThermostatConfig(
        name="ag_news-roberta-svs",
        description="AG News dataset, RoBERTa model, Shapley Value Sampling explanations",
        explainer="ShapleyValueSampling",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/yabEAY5sLpjxKkW/download",
        **_AGNEWS_ROBERTA_KWARGS,
    ),
    # shap value inclusion
    ThermostatConfig(
        name="ag_news-roberta-lds",
        description="AG News dataset, RoBERTa model, Layer DeepLift Shap explanations",
        explainer="LayerDeepLiftShap",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/ZWoxq27s36For98/download",
        **_AGNEWS_ROBERTA_KWARGS,
    ),
    ThermostatConfig(
        name="ag_news-roberta-lgs",
        description="AG News dataset, RoBERTa model, Layer Gradient Shap explanations",
        explainer="LayerGradientShap",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/HgHjNFcMQbCC2Nb/download",
        **_AGNEWS_ROBERTA_KWARGS,
    ),
    # shap value inclusion
    ThermostatConfig(
        name="imdb-albert-lgxa",
        description="IMDb dataset, ALBERT model, Layer Gradient x Activation explanations",
        explainer="LayerGradientXActivation",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/srbYBbmKBsGMXWn/download",
        **_IMDB_ALBERT_KWARGS,
    ),
    ThermostatConfig(
        name="imdb-albert-lig",
        description="IMDb dataset, ALBERT model, Layer Integrated Gradients explanations",
        explainer="LayerIntegratedGradients",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/zjMddcqewEcwSPG/download",
        **_IMDB_ALBERT_KWARGS,
    ),
    ThermostatConfig(
        name="imdb-albert-lime",
        description="IMDb dataset, ALBERT model, LIME explanations",
        explainer="LimeBase",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/Tgktb4fq4EdXJNx/download",
        **_IMDB_ALBERT_KWARGS,
    ),
    # new change
    ThermostatConfig(
        name="imdb-albert-lime-100",
        description="IMDb dataset, ALBERT model, LIME explanations, 100 samples",
        explainer="LimeBase",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/FzErcT9TcFcG2Pr/download",
        **_IMDB_ALBERT_KWARGS,
    ),
    # new change
    ThermostatConfig(
        name="imdb-albert-occ",
        description="IMDb dataset, ALBERT model, Occlusion explanations",
        explainer="Occlusion",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/98XqEgbZt9KiSfm/download",
        **_IMDB_ALBERT_KWARGS,
    ),
    ThermostatConfig(
        name="imdb-albert-svs",
        description="IMDb dataset, ALBERT model, Shapley Value Sampling explanations",
        explainer="ShapleyValueSampling",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/sQMK2XsknbzK23a/download",
        **_IMDB_ALBERT_KWARGS,
    ),
    # shap value inclusion
    ThermostatConfig(
        name="imdb-albert-lds",
        description="IMDb dataset, ALBERT model, Layer DeepLift Shap explanations",
        explainer="LayerDeepLiftShap",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/SYg2GjRkewW8fn7/download",
        **_IMDB_ALBERT_KWARGS,
    ),
    ThermostatConfig(
        name="imdb-albert-lgs",
        description="IMDb dataset, ALBERT model, Layer Gradient Shap explanations",
        explainer="LayerGradientShap",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/rWZfSzPN7Gm3Cko/download",
        **_IMDB_ALBERT_KWARGS,
    ),
    # shap value inclusion
    ThermostatConfig(
        name="imdb-bert-lgxa",
        description="IMDb dataset, BERT model, Layer Gradient x Activation explanations",
        explainer="LayerGradientXActivation",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/NkPpnyMN8rdWE7L/download",
        **_IMDB_BERT_KWARGS,
    ),
    ThermostatConfig(
        name="imdb-bert-lig",
        description="IMDb dataset, BERT model, Layer Integrated Gradients explanations",
        explainer="LayerIntegratedGradients",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/SdrPeJQQSExFQ8e/download",
        **_IMDB_BERT_KWARGS,
    ),
    ThermostatConfig(
        name="imdb-bert-lime",
        description="IMDb dataset, BERT model, LIME explanations",
        explainer="LimeBase",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/ZQEdEmFtKeGkYWp/download",
        **_IMDB_BERT_KWARGS,
    ),
    # new change
    ThermostatConfig(
        name="imdb-bert-lime-100",
        description="IMDb dataset, BERT model, LIME explanations, 100 samples",
        explainer="LimeBase",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/Qx7z8SFcMTB5bFa/download",
        **_IMDB_BERT_KWARGS,
    ),
    # new change
    ThermostatConfig(
        name="imdb-bert-occ",
        description="IMDb dataset, BERT model, Occlusion explanations",
        explainer="Occlusion",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/PjMDBzaoHHqs2WF/download",
        **_IMDB_BERT_KWARGS,
    ),
    ThermostatConfig(
        name="imdb-bert-svs",
        description="IMDb dataset, BERT model, Shapley Value Sampling explanations",
        explainer="ShapleyValueSampling",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/DjmCKdBoWHt8jbX/download",
        **_IMDB_BERT_KWARGS,
    ),
    # shap value inclusion
    ThermostatConfig(
        name="imdb-bert-lds",
        description="IMDb dataset, BERT model, Layer DeepLift Shap explanations",
        explainer="LayerDeepLiftShap",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/dnqSfrgGPYcYsKt/download",
        **_IMDB_BERT_KWARGS,
    ),
    ThermostatConfig(
        name="imdb-bert-lgs",
        description="IMDb dataset, BERT model, Layer Gradient Shap explanations",
        explainer="LayerGradientShap",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/fnBqKxEjfsScqwg/download",
        **_IMDB_BERT_KWARGS,
    ),
    # shap value inclusion
    ThermostatConfig(
        name="imdb-electra-lgxa",
        description="IMDb dataset, ELECTRA model, Layer Gradient x Activation explanations",
        explainer="LayerGradientXActivation",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/WdLYpQerXC5KrHK/download",
        **_IMDB_ELECTRA_KWARGS,
    ),
    ThermostatConfig(
        name="imdb-electra-lig",
        description="IMDb dataset, ELECTRA model, Layer Integrated Gradients explanations",
        explainer="LayerIntegratedGradients",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/e3Mibf9dqRfobYw/download",
        **_IMDB_ELECTRA_KWARGS,
    ),
    ThermostatConfig(
        name="imdb-electra-lime",
        description="IMDb dataset, ELECTRA model, LIME explanations",
        explainer="LimeBase",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/7p2576kFqiQLL9x/download",
        **_IMDB_ELECTRA_KWARGS,
    ),
    # new change
    ThermostatConfig(
        name="imdb-electra-lime-100",
        description="IMDb dataset, ELECTRA model, LIME explanations, 100 samples",
        explainer="LimeBase",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/LBqzn6JiQNzwMAC/download",
        **_IMDB_ELECTRA_KWARGS,
    ),
    # new change
    ThermostatConfig(
        name="imdb-electra-occ",
        description="IMDb dataset, ELECTRA model, Occlusion explanations",
        explainer="Occlusion",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/TZYTgnySrEbm5Xx/download",
        **_IMDB_ELECTRA_KWARGS,
    ),
    ThermostatConfig(
        name="imdb-electra-svs",
        description="IMDb dataset, ELECTRA model, Shapley Value Sampling explanations",
        explainer="ShapleyValueSampling",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/MPHqZwJCP97sA4D/download",
        **_IMDB_ELECTRA_KWARGS,
    ),
    # shap value inclusion
    ThermostatConfig(
        name="imdb-electra-lds",
        description="IMDb dataset, ELECTRA model, Layer DeepLift Shap explanations",
        explainer="LayerDeepLiftShap",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/cHdECMFjHacAnFk/download",
        **_IMDB_ELECTRA_KWARGS,
    ),
    ThermostatConfig(
        name="imdb-electra-lgs",
        description="IMDb dataset, ELECTRA model, Layer Gradient Shap explanations",
        explainer="LayerGradientShap",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/dRM6RKSwD5fKteG/download",
        **_IMDB_ELECTRA_KWARGS,
    ),
    # shap value inclusion
    ThermostatConfig(
        name="imdb-roberta-lgxa",
        description="IMDb dataset, RoBERTa model, Layer Gradient x Activation explanations",
        explainer="LayerGradientXActivation",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/oWCkBFgsstPakKS/download",
        **_IMDB_ROBERTA_KWARGS,
    ),
    ThermostatConfig(
        name="imdb-roberta-lig",
        description="IMDb dataset, RoBERTa model, Layer Integrated Gradients explanations",
        explainer="LayerIntegratedGradients",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/qJBYkfwppGZ4NF5/download",
        **_IMDB_ROBERTA_KWARGS,
    ),
    ThermostatConfig(
        name="imdb-roberta-lime",
        description="IMDb dataset, RoBERTa model, LIME explanations",
        explainer="LimeBase",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/rpsMTw3S6JkQgcF/download",
        **_IMDB_ROBERTA_KWARGS,
    ),
    # new change
    ThermostatConfig(
        name="imdb-roberta-lime-100",
        description="IMDb dataset, RoBERTa model, LIME explanations, 100 samples",
        explainer="LimeBase",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/YZsAoJmR4EcwnG2/download",
        **_IMDB_ROBERTA_KWARGS,
    ),
    # new change
    ThermostatConfig(
        name="imdb-roberta-occ",
        description="IMDb dataset, RoBERTa model, Occlusion explanations",
        explainer="Occlusion",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/wDw9k7PRWwsfQPB/download",
        **_IMDB_ROBERTA_KWARGS,
    ),
    ThermostatConfig(
        name="imdb-roberta-svs",
        description="IMDb dataset, RoBERTa model, Shapley Value Sampling explanations",
        explainer="ShapleyValueSampling",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/339zLEttF6djtBR/download",
        **_IMDB_ROBERTA_KWARGS,
    ),
    # shap value inclusion
    ThermostatConfig(
        name="imdb-roberta-lds",
        description="IMDb dataset, RoBERTa model, Layer DeepLift Shap explanations",
        explainer="LayerDeepLiftShap",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/EzD7brEosFx4iW2/download",
        **_IMDB_ROBERTA_KWARGS,
    ),
    ThermostatConfig(
        name="imdb-roberta-lgs",
        description="IMDb dataset, RoBERTa model, Layer Gradient Shap explanations",
        explainer="LayerGradientShap",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/ptcfks7sTnpm85M/download",
        **_IMDB_ROBERTA_KWARGS,
    ),
    # shap value inclusion
    ThermostatConfig(
        name="imdb-xlnet-lgxa",
        description="IMDb dataset, XLNet model, Layer Gradient x Activation explanations",
        explainer="LayerGradientXActivation",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/53g8Gw28BX9eiPQ/download",
        **_IMDB_XLNET_KWARGS,
    ),
    ThermostatConfig(
        name="imdb-xlnet-lig",
        description="IMDb dataset, XLNet model, Layer Integrated Gradients explanations",
        explainer="LayerIntegratedGradients",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/sgn79wJgTWjoNjq/download",
        **_IMDB_XLNET_KWARGS,
    ),
    ThermostatConfig(
        name="imdb-xlnet-lime",
        description="IMDb dataset, XLNet model, LIME explanations",
        explainer="LimeBase",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/YCDW67f49wj5NXg/download",
        **_IMDB_XLNET_KWARGS,
    ),
    # new change
    ThermostatConfig(
        name="imdb-xlnet-lime-100",
        description="IMDb dataset, XLNet model, LIME explanations, 100 samples",
        explainer="LimeBase",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/T2KsA8ragxPz6eL/download",
        **_IMDB_XLNET_KWARGS,
    ),
    # new change
    ThermostatConfig(
        name="imdb-xlnet-occ",
        description="IMDb dataset, XLNet model, Occlusion explanations",
        explainer="Occlusion",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/H46msX6FQfFrCpg/download",
        **_IMDB_XLNET_KWARGS,
    ),
    ThermostatConfig(
        name="imdb-xlnet-svs",
        description="IMDb dataset, XLNet model, Shapley Value Sampling explanations",
        explainer="ShapleyValueSampling",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/y9grFyRQC2rDSaN/download",
        **_IMDB_XLNET_KWARGS,
    ),
    ThermostatConfig(
        name="multi_nli-albert-lgxa",
        description="MultiNLI dataset, ALBERT model, Layer Gradient x Activation explanations",
        explainer="LayerGradientXActivation",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/nkRmprdnbb5C4Tx/download",
        **_MNLI_ALBERT_KWARGS,
    ),
    ThermostatConfig(
        name="multi_nli-albert-lig",
        description="MultiNLI dataset, ALBERT model, Layer Integrated Gradients explanations",
        explainer="LayerIntegratedGradients",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/3WAqbXa2DG2RCgz/download",
        **_MNLI_ALBERT_KWARGS,
    ),
    ThermostatConfig(
        name="multi_nli-albert-lime",
        description="MultiNLI dataset, ALBERT model, LIME explanations",
        explainer="LimeBase",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/e6JRy9fidSAC5zK/download",
        **_MNLI_ALBERT_KWARGS,
    ),
    # new change
    ThermostatConfig(
        name="multi_nli-albert-lime-100",
        description="MultiNLI dataset, ALBERT model, LIME explanations, 100 samples",
        explainer="LimeBase",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/WB2N3nFkHTGkXY8/download",
        **_MNLI_ALBERT_KWARGS,
    ),
    # new change
    ThermostatConfig(
        name="multi_nli-albert-occ",
        description="MultiNLI dataset, ALBERT model, Occlusion explanations",
        explainer="Occlusion",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/F5xWYpyDpwaAPJs/download",
        **_MNLI_ALBERT_KWARGS,
    ),
    ThermostatConfig(
        name="multi_nli-albert-svs",
        description="MultiNLI dataset, ALBERT model, Shapley Value Sampling explanations",
        explainer="ShapleyValueSampling",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/fffM7w64CnTSzHA/download",
        **_MNLI_ALBERT_KWARGS,
    ),
    # shap value inclusion
    ThermostatConfig(
        name="multi_nli-albert-lds",
        description="MultiNLI dataset, ALBERT model, Layer DeepLift Shap explanations",
        explainer="LayerDeepLiftShap",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/LiMzjexGXc5PdAm/download",
        **_MNLI_ALBERT_KWARGS,
    ),
    ThermostatConfig(
        name="multi_nli-albert-lgs",
        description="MultiNLI dataset, ALBERT model, Layer Gradient Shap explanations",
        explainer="LayerGradientShap",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/ZL9fN8QKy8Di58B/download",
        **_MNLI_ALBERT_KWARGS,
    ),
    # shap value inclusion
    ThermostatConfig(
        name="multi_nli-bert-lgxa",
        description="MultiNLI dataset, BERT model, Layer Gradient x Activation explanations",
        explainer="LayerGradientXActivation",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/MdjebgkdexA2ZDt/download",
        **_MNLI_BERT_KWARGS,
    ),
    ThermostatConfig(
        name="multi_nli-bert-lig",
        description="MultiNLI dataset, BERT model, Layer Integrated Gradients explanations",
        explainer="LayerIntegratedGradients",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/g53nbtnXaFyPLM7/download",
        **_MNLI_BERT_KWARGS,
    ),
    ThermostatConfig(
        name="multi_nli-bert-lime",
        description="MultiNLI dataset, BERT model, LIME explanations",
        explainer="LimeBase",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/ptspBexoHaXtqXD/download",
        **_MNLI_BERT_KWARGS,
    ),
    # new change
    ThermostatConfig(
        name="multi_nli-bert-lime-100",
        description="MultiNLI dataset, BERT model, LIME explanations, 100 samples",
        explainer="LimeBase",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/LjFccwQ2mCAnsmH/download",
        **_MNLI_BERT_KWARGS,
    ),
    # new change
    ThermostatConfig(
        name="multi_nli-bert-occ",
        description="MultiNLI dataset, BERT model, Occlusion explanations",
        explainer="Occlusion",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/cHK6YCAo4ESZ3xx/download",
        **_MNLI_BERT_KWARGS,
    ),
    ThermostatConfig(
        name="multi_nli-bert-svs",
        description="MultiNLI dataset, BERT model, Shapley Value Sampling explanations",
        explainer="ShapleyValueSampling",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/d5TTHCkAb5TJmbg/download",
        **_MNLI_BERT_KWARGS,
    ),
    # shap value inclusion
    ThermostatConfig(
        name="multi_nli-bert-lds",
        description="MultiNLI dataset, BERT model, Layer DeepLift Shap explanations",
        explainer="LayerDeepLiftShap",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/yBtF9ybEm5kCzeg/download",
        **_MNLI_BERT_KWARGS,
    ),
    ThermostatConfig(
        name="multi_nli-bert-lgs",
        description="MultiNLI dataset, BERT model, Layer Gradient Shap explanations",
        explainer="LayerGradientShap",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/XMAgeYenwS7MSWP/download",
        **_MNLI_BERT_KWARGS,
    ),
    # shap value inclusion
    ThermostatConfig(
        name="multi_nli-electra-lgxa",
        description="MultiNLI dataset, ELECTRA model, Layer Gradient x Activation explanations",
        explainer="LayerGradientXActivation",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/2HrCmp9sxJNiKBc/download",
        **_MNLI_ELECTRA_KWARGS,
    ),
    ThermostatConfig(
        name="multi_nli-electra-lig",
        description="MultiNLI dataset, ELECTRA model, Layer Integrated Gradients explanations",
        explainer="LayerIntegratedGradients",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/2eZnJgCbWd2D4PB/download",
        **_MNLI_ELECTRA_KWARGS,
    ),
    ThermostatConfig(
        name="multi_nli-electra-lime",
        description="MultiNLI dataset, ELECTRA model, LIME explanations",
        explainer="LimeBase",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/WzBwpwC9FoQZCwB/download",
        **_MNLI_ELECTRA_KWARGS,
    ),
    # new change
    ThermostatConfig(
        name="multi_nli-electra-lime-100",
        description="MultiNLI dataset, ELECTRA model, LIME explanations, 100 samples",
        explainer="LimeBase",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/TX6jWs9wBdsJA9w/download",
        **_MNLI_ELECTRA_KWARGS,
    ),
    # new change
    ThermostatConfig(
        name="multi_nli-electra-occ",
        description="MultiNLI dataset, ELECTRA model, Occlusion explanations",
        explainer="Occlusion",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/MGjQmKK9kynZTQt/download",
        **_MNLI_ELECTRA_KWARGS,
    ),
    ThermostatConfig(
        name="multi_nli-electra-svs",
        description="MultiNLI dataset, ELECTRA model, Shapley Value Sampling explanations",
        explainer="ShapleyValueSampling",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/zx3rGTpMkRT68tk/download",
        **_MNLI_ELECTRA_KWARGS,
    ),
    # shap value inclusion
    ThermostatConfig(
        name="multi_nli-electra-lds",
        description="MultiNLI dataset, ELECTRA model, Layer DeepLift Shap explanations",
        explainer="LayerDeepLiftShap",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/PD578t8fzsR2Q6k/download",
        **_MNLI_ELECTRA_KWARGS,
    ),
    ThermostatConfig(
        name="multi_nli-electra-lgs",
        description="MultiNLI dataset, ELECTRA model, Layer Gradient Shap explanations",
        explainer="LayerGradientShap",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/iq2QNQRojpsoKwW/download",
        **_MNLI_ELECTRA_KWARGS,
    ),
    # shap value inclusion
    ThermostatConfig(
        name="multi_nli-roberta-lgxa",
        description="MultiNLI dataset, RoBERTa model, Layer Gradient x Activation explanations",
        explainer="LayerGradientXActivation",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/SxDwtGCpPzi3DDz/download",
        **_MNLI_ROBERTA_KWARGS,
    ),
    ThermostatConfig(
        name="multi_nli-roberta-lig",
        description="MultiNLI dataset, RoBERTa model, Layer Integrated Gradients explanations",
        explainer="LayerIntegratedGradients",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/8zaTxTCijG6g7Y5/download",
        **_MNLI_ROBERTA_KWARGS,
    ),
    ThermostatConfig(
        name="multi_nli-roberta-lime",
        description="MultiNLI dataset, RoBERTa model, LIME explanations",
        explainer="LimeBase",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/dY4z4ptcMtiYzZs/download",
        **_MNLI_ROBERTA_KWARGS,
    ),
    # new change
    ThermostatConfig(
        name="multi_nli-roberta-lime-100",
        description="MultiNLI dataset, RoBERTa model, LIME explanations, 100 samples",
        explainer="LimeBase",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/KTQWmCDX2EjHtQE/download",
        **_MNLI_ROBERTA_KWARGS,
    ),
    # new change
    ThermostatConfig(
        name="multi_nli-roberta-occ",
        description="MultiNLI dataset, RoBERTa model, Occlusion explanations",
        explainer="Occlusion",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/w5YSxNc6L8QZisG/download",
        **_MNLI_ROBERTA_KWARGS,
    ),
    ThermostatConfig(
        name="multi_nli-roberta-svs",
        description="MultiNLI dataset, RoBERTa model, Shapley Value Sampling explanations",
        explainer="ShapleyValueSampling",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/3aPeTawM8cbAsEg/download",
        **_MNLI_ROBERTA_KWARGS,
    ),
    # shap value inclusion
    ThermostatConfig(
        name="multi_nli-roberta-lds",
        description="MultiNLI dataset, RoBERTa model, Layer DeepLift Shap explanations",
        explainer="LayerDeepLiftShap",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/Kg4eLx2SZWrf8MF/download",
        **_MNLI_ROBERTA_KWARGS,
    ),
    ThermostatConfig(
        name="multi_nli-roberta-lgs",
        description="MultiNLI dataset, RoBERTa model, Layer Gradient Shap explanations",
        explainer="LayerGradientShap",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/7Fxtawd3t8WXTWC/download",
        **_MNLI_ROBERTA_KWARGS,
    ),
    # shap value inclusion
    ThermostatConfig(
        name="multi_nli-xlnet-lgxa",
        description="MultiNLI dataset, XLNet model, Layer Gradient x Activation explanations",
        explainer="LayerGradientXActivation",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/n79G9kf9jbNx8o7/download",
        **_MNLI_XLNET_KWARGS,
    ),
    ThermostatConfig(
        name="multi_nli-xlnet-lig",
        description="MultiNLI dataset, XLNet model, Layer Integrated Gradients explanations",
        explainer="LayerIntegratedGradients",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/MZr8jTnaCBdMPGe/download",
        **_MNLI_XLNET_KWARGS,
    ),
    ThermostatConfig(
        name="multi_nli-xlnet-lime",
        description="MultiNLI dataset, XLNet model, LIME explanations",
        explainer="LimeBase",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/B7tfLSRKBGYxJ3s/download",
        **_MNLI_XLNET_KWARGS,
    ),
    # new change
    ThermostatConfig(
        name="multi_nli-xlnet-lime-100",
        description="MultiNLI dataset, XLNet model, LIME explanations, 100 samples",
        explainer="LimeBase",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/SesZACA2AwyefFp/download",
        **_MNLI_XLNET_KWARGS,
    ),
    # new change
    ThermostatConfig(
        name="multi_nli-xlnet-occ",
        description="MultiNLI dataset, XLNet model, Occlusion explanations",
        explainer="Occlusion",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/YWjJ6T7n6oeKbJJ/download",
        **_MNLI_XLNET_KWARGS,
    ),
    ThermostatConfig(
        name="multi_nli-xlnet-svs",
        description="MultiNLI dataset, XLNet model, Shapley Value Sampling explanations",
        explainer="ShapleyValueSampling",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/CXYRFGsR2NFeAZa/download",
        **_MNLI_XLNET_KWARGS,
    ),
    ThermostatConfig(
        name="xnli-albert-lgxa",
        description="XNLI dataset, ALBERT model, Layer Gradient x Activation explanations",
        explainer="LayerGradientXActivation",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/zCSq69Z853fs3ez/download",
        **_XNLI_ALBERT_KWARGS,
    ),
    ThermostatConfig(
        name="xnli-albert-lig",
        description="XNLI dataset, ALBERT model, Layer Integrated Gradients explanations",
        explainer="LayerIntegratedGradients",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/ZcP34Eg6eb3TrWF/download",
        **_XNLI_ALBERT_KWARGS,
    ),
    ThermostatConfig(
        name="xnli-albert-lime",
        description="XNLI dataset, ALBERT model, LIME explanations",
        explainer="LimeBase",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/sijLW3ceigxDsKY/download",
        **_XNLI_ALBERT_KWARGS,
    ),
    # new change
    ThermostatConfig(
        name="xnli-albert-lime-100",
        description="XNLI dataset, ALBERT model, LIME explanations, 100 samples",
        explainer="LimeBase",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/oQW5cRc6GbqHtB6/download",
        **_XNLI_ALBERT_KWARGS,
    ),
    # new change
    ThermostatConfig(
        name="xnli-albert-occ",
        description="XNLI dataset, ALBERT model, Occlusion explanations",
        explainer="Occlusion",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/bEg95CGBtzaFQij/download",
        **_XNLI_ALBERT_KWARGS,
    ),
    ThermostatConfig(
        name="xnli-albert-svs",
        description="XNLI dataset, ALBERT model, Shapley Value Sampling explanations",
        explainer="ShapleyValueSampling",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/wekiPq7ijzsCQK4/download",
        **_XNLI_ALBERT_KWARGS,
    ),
    # shap value inclusion
    ThermostatConfig(
        name="xnli-albert-lds",
        description="XNLI dataset, ALBERT model, Layer DeepLift Shap explanations",
        explainer="LayerDeepLiftShap",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/AzKmCQfEP6CmwTH/download",
        **_XNLI_ALBERT_KWARGS,
    ),
    ThermostatConfig(
        name="xnli-albert-lgs",
        description="XNLI dataset, ALBERT model, Layer Gradient Shap explanations",
        explainer="LayerGradientShap",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/qWdK5YkSBxmMPKp/download",
        **_XNLI_ALBERT_KWARGS,
    ),
    # shap value inclusion
    ThermostatConfig(
        name="xnli-bert-lgxa",
        description="XNLI dataset, BERT model, Layer Gradient x Activation explanations",
        explainer="LayerGradientXActivation",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/pb3Q6GQodyMqkgJ/download",
        **_XNLI_BERT_KWARGS,
    ),
    ThermostatConfig(
        name="xnli-bert-lig",
        description="XNLI dataset, BERT model, Layer Integrated Gradients explanations",
        explainer="LayerIntegratedGradients",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/MX8YkzjFtpd43PM/download",
        **_XNLI_BERT_KWARGS,
    ),
    ThermostatConfig(
        name="xnli-bert-lime",
        description="XNLI dataset, BERT model, LIME explanations",
        explainer="LimeBase",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/KfjqkRTd7FSWSkx/download",
        **_XNLI_BERT_KWARGS,
    ),
    # new change
    ThermostatConfig(
        name="xnli-bert-lime-100",
        description="XNLI dataset, BERT model, LIME explanations, 100 samples",
        explainer="LimeBase",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/FXHt989a2En8aZZ/download",
        **_XNLI_BERT_KWARGS,
    ),
    # new change
    ThermostatConfig(
        name="xnli-bert-occ",
        description="XNLI dataset, BERT model, Occlusion explanations",
        explainer="Occlusion",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/AHoTKbtSCQ73QxN/download",
        **_XNLI_BERT_KWARGS,
    ),
    ThermostatConfig(
        name="xnli-bert-svs",
        description="XNLI dataset, BERT model, Shapley Value Sampling explanations",
        explainer="ShapleyValueSampling",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/D4ctEijzerMoNT8/download",
        **_XNLI_BERT_KWARGS,
    ),
    # shap value inclusion
    ThermostatConfig(
        name="xnli-bert-lds",
        description="XNLI dataset, BERT model, Layer DeepLift Shap explanations",
        explainer="LayerDeepLiftShap",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/LtARP8wA4mLCcL2/download",
        **_XNLI_BERT_KWARGS,
    ),
    ThermostatConfig(
        name="xnli-bert-lgs",
        description="XNLI dataset, BERT model, Layer Gradient Shap explanations",
        explainer="LayerGradientShap",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/AHw6WdfNbYXP9zD/download",
        **_XNLI_BERT_KWARGS,
    ),
    # shap value inclusion
    ThermostatConfig(
        name="xnli-electra-lgxa",
        description="XNLI dataset, ELECTRA model, Layer Gradient x Activation explanations",
        explainer="LayerGradientXActivation",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/t4ge7pA57gy4dKr/download",
        **_XNLI_ELECTRA_KWARGS,
    ),
    ThermostatConfig(
        name="xnli-electra-lig",
        description="XNLI dataset, ELECTRA model, Layer Integrated Gradients explanations",
        explainer="LayerIntegratedGradients",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/fHBSRQoAXzo3EKj/download",
        **_XNLI_ELECTRA_KWARGS,
    ),
    ThermostatConfig(
        name="xnli-electra-lime",
        description="XNLI dataset, ELECTRA model, LIME explanations",
        explainer="LimeBase",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/XnkiHXgxNsptxTJ/download",
        **_XNLI_ELECTRA_KWARGS,
    ),
    # new change
    ThermostatConfig(
        name="xnli-electra-lime-100",
        description="XNLI dataset, ELECTRA model, LIME explanations, 100 samples",
        explainer="LimeBase",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/7zNtxCHxEZk2tzC/download",
        **_XNLI_ELECTRA_KWARGS,
    ),
    # new change
    ThermostatConfig(
        name="xnli-electra-occ",
        description="XNLI dataset, ELECTRA model, Occlusion explanations",
        explainer="Occlusion",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/2RtRaN8q5fDHWyF/download",
        **_XNLI_ELECTRA_KWARGS,
    ),
    ThermostatConfig(
        name="xnli-electra-svs",
        description="XNLI dataset, ELECTRA model, Shapley Value Sampling explanations",
        explainer="ShapleyValueSampling",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/T3KKsM5TtsHyCAL/download",
        **_XNLI_ELECTRA_KWARGS,
    ),
    # shap value inclusion
    ThermostatConfig(
        name="xnli-electra-lds",
        description="XNLI dataset, ELECTRA model, Layer DeepLift Shap explanations",
        explainer="LayerDeepLiftShap",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/MmHwGXa6EpzT3ns/download",
        **_XNLI_ELECTRA_KWARGS,
    ),
    ThermostatConfig(
        name="xnli-electra-lgs",
        description="XNLI dataset, ELECTRA model, Layer Gradient Shap explanations",
        explainer="LayerGradientShap",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/gLwPwAaJDCB956T/download",
        **_XNLI_ELECTRA_KWARGS,
    ),
    # shap value inclusion
    ThermostatConfig(
        name="xnli-roberta-lgxa",
        description="XNLI dataset, RoBERTa model, Layer Gradient x Activation explanations",
        explainer="LayerGradientXActivation",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/aPoCzcXDCfya3Ww/download",
        **_XNLI_ROBERTA_KWARGS,
    ),
    ThermostatConfig(
        name="xnli-roberta-lig",
        description="XNLI dataset, RoBERTa model, Layer Integrated Gradients explanations",
        explainer="LayerIntegratedGradients",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/iPA6rCfjc49ofpN/download",
        **_XNLI_ROBERTA_KWARGS,
    ),
    ThermostatConfig(
        name="xnli-roberta-lime",
        description="XNLI dataset, RoBERTa model, LIME explanations",
        explainer="LimeBase",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/pZKo7m4g9WJXfoe/download",
        **_XNLI_ROBERTA_KWARGS,
    ),
    # new change
    ThermostatConfig(
        name="xnli-roberta-lime-100",
        description="XNLI dataset, RoBERTa model, LIME explanations, 100 samples",
        explainer="LimeBase",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/CHSR7Arw8M56bxN/download",
        **_XNLI_ROBERTA_KWARGS,
    ),
    # new change
    ThermostatConfig(
        name="xnli-roberta-occ",
        description="XNLI dataset, RoBERTa model, Occlusion explanations",
        explainer="Occlusion",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/XB2tnATQW3tbxPW/download",
        **_XNLI_ROBERTA_KWARGS,
    ),
    ThermostatConfig(
        name="xnli-roberta-svs",
        description="XNLI dataset, RoBERTa model, Shapley Value Sampling explanations",
        explainer="ShapleyValueSampling",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/opYTzjSeWWL7eYg/download",
        **_XNLI_ROBERTA_KWARGS,
    ),
    # shap value inclusion
    ThermostatConfig(
        name="xnli-roberta-lds",
        description="XNLI dataset, RoBERTa model, Layer DeepLift Shap explanations",
        explainer="LayerDeepLiftShap",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/zsrrBirPHY4gp2p/download",
        **_XNLI_ROBERTA_KWARGS,
    ),
    ThermostatConfig(
        name="xnli-roberta-lgs",
        description="XNLI dataset, RoBERTa model, Layer Gradient Shap explanations",
        explainer="LayerGradientShap",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/fxAMzqGzgBZb9b4/download",
        **_XNLI_ROBERTA_KWARGS,
    ),
    # shap value inclusion
    ThermostatConfig(
        name="xnli-xlnet-lgxa",
        description="XNLI dataset, XLNet model, Layer Gradient x Activation explanations",
        explainer="LayerGradientXActivation",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/jBXEkQS7WTz3a7J/download",
        **_XNLI_XLNET_KWARGS,
    ),
    ThermostatConfig(
        name="xnli-xlnet-lig",
        description="XNLI dataset, XLNet model, Layer Integrated Gradients explanations",
        explainer="LayerIntegratedGradients",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/kx7cJYFbyCjy58z/download",
        **_XNLI_XLNET_KWARGS,
    ),
    ThermostatConfig(
        name="xnli-xlnet-lime",
        description="XNLI dataset, XLNet model, LIME explanations",
        explainer="LimeBase",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/6s4DFPNYpzi8722/download",
        **_XNLI_XLNET_KWARGS,
    ),
    # new change
    ThermostatConfig(
        name="xnli-xlnet-lime-100",
        description="XNLI dataset, XLNet model, LIME explanations, 100 samples",
        explainer="LimeBase",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/ZzN9PSkiRrJNza2/download",
        **_XNLI_XLNET_KWARGS,
    ),
    # new change
    ThermostatConfig(
        name="xnli-xlnet-occ",
        description="XNLI dataset, XLNet model, Occlusion explanations",
        explainer="Occlusion",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/yEFEyrq4pbGKP4s/download",
        **_XNLI_XLNET_KWARGS,
    ),
    ThermostatConfig(
        name="xnli-xlnet-svs",
        description="XNLI dataset, XLNet model, Shapley Value Sampling explanations",
        explainer="ShapleyValueSampling",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/fT34Q7CD2GQkdxJ/download",
        **_XNLI_XLNET_KWARGS,
    ),
]
