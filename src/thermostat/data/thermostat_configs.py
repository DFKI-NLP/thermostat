import datasets


_VERSION = datasets.Version('1.0.0', '')


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
    text_column=["premise", "hypothesis"],
    **_BASE_KWARGS,
)
_MNLI_ALBERT_KWARGS = dict(
    model="prajjwal1/albert-base-v2-mnli",
    label_classes=["contradiction", "entailment", "neutral"],
    **_MNLI_KWARGS,
)
_MNLI_BERT_KWARGS = dict(
    model="textattack/bert-base-uncased-MNLI",
    label_classes=["contradiction", "entailment", "neutral"],
    **_MNLI_KWARGS,
)
_MNLI_ELECTRA_KWARGS = dict(
    model="howey/electra-base-mnli",
    label_classes=["entailment", "neutral", "contradiction"],
    **_MNLI_KWARGS,
)
_MNLI_ROBERTA_KWARGS = dict(
    model="textattack/roberta-base-MNLI",
    label_classes=["contradiction", "entailment", "neutral"],
    **_MNLI_KWARGS,
)
_MNLI_XLNET_KWARGS = dict(
    model="textattack/xlnet-base-cased-MNLI",
    label_classes=["contradiction", "entailment", "neutral"],
    **_MNLI_KWARGS,
)

# Base arguments for XNLI dataset
_XNLI_KWARGS = dict(
    dataset="xnli",
    label_column="label",
    text_column=["premise", "hypothesis"],
    **_BASE_KWARGS,
)
_XNLI_ALBERT_KWARGS = dict(
    model="prajjwal1/albert-base-v2-mnli",
    label_classes=["contradiction", "entailment", "neutral"],
    **_XNLI_KWARGS,
)
_XNLI_BERT_KWARGS = dict(
    model="textattack/bert-base-uncased-MNLI",
    label_classes=["contradiction", "entailment", "neutral"],
    **_XNLI_KWARGS,
)
_XNLI_ELECTRA_KWARGS = dict(
    model="howey/electra-base-mnli",
    label_classes=["entailment", "neutral", "contradiction"],
    **_XNLI_KWARGS,
)
_XNLI_ROBERTA_KWARGS = dict(
    model="textattack/roberta-base-MNLI",
    label_classes=["contradiction", "entailment", "neutral"],
    **_XNLI_KWARGS,
)
_XNLI_XLNET_KWARGS = dict(
    model="textattack/xlnet-base-cased-MNLI",
    label_classes=["contradiction", "entailment", "neutral"],
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
    ThermostatConfig(
        name="ag_news-bert-occlusion",
        description="AG News dataset, BERT model, Occlusion explanations",
        explainer="Occlusion",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/Grf97s6bJwoZGyx/download",
        **_AGNEWS_BERT_KWARGS,
    ),
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
    ThermostatConfig(
        name="ag_news-roberta-occlusion",
        description="AG News dataset, RoBERTa model, Occlusion explanations",
        explainer="Occlusion",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/78aZttqxKQNdW6J/download",
        **_AGNEWS_ROBERTA_KWARGS,
    ),
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
    ThermostatConfig(
        name="imdb-albert-occ",
        description="IMDb dataset, ALBERT model, Occlusion explanations",
        explainer="Occlusion",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/98XqEgbZt9KiSfm/download",
        **_IMDB_ALBERT_KWARGS,
    ),
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
    ThermostatConfig(
        name="imdb-bert-occ",
        description="IMDb dataset, BERT model, Occlusion explanations",
        explainer="Occlusion",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/PjMDBzaoHHqs2WF/download",
        **_IMDB_BERT_KWARGS,
    ),
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
    ThermostatConfig(
        name="imdb-electra-occ",
        description="IMDb dataset, ELECTRA model, Occlusion explanations",
        explainer="Occlusion",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/TZYTgnySrEbm5Xx/download",
        **_IMDB_ELECTRA_KWARGS,
    ),
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
    ThermostatConfig(
        name="imdb-xlnet-occ",
        description="IMDb dataset, XLNet model, Occlusion explanations",
        explainer="Occlusion",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/H46msX6FQfFrCpg/download",
        **_IMDB_XLNET_KWARGS,
    ),
    ThermostatConfig(
        name="multi_nli-albert-lgxa",
        description="MNLI dataset, ALBERT model, Layer Gradient x Activation explanations",
        explainer="LayerGradientXActivation",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/nkRmprdnbb5C4Tx/download",
        **_MNLI_ALBERT_KWARGS,
    ),
    ThermostatConfig(
        name="multi_nli-albert-lig",
        description="MNLI dataset, ALBERT model, Layer Integrated Gradients explanations",
        explainer="LayerIntegratedGradients",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/3WAqbXa2DG2RCgz/download",
        **_MNLI_ALBERT_KWARGS,
    ),
    ThermostatConfig(
        name="multi_nli-albert-lime",
        description="MNLI dataset, ALBERT model, LIME explanations",
        explainer="LimeBase",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/e6JRy9fidSAC5zK/download",
        **_MNLI_ALBERT_KWARGS,
    ),
    ThermostatConfig(
        name="multi_nli-albert-occ",
        description="MNLI dataset, ALBERT model, Occlusion explanations",
        explainer="Occlusion",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/F5xWYpyDpwaAPJs/download",
        **_MNLI_ALBERT_KWARGS,
    ),
    ThermostatConfig(
        name="multi_nli-bert-lgxa",
        description="MNLI dataset, BERT model, Layer Gradient x Activation explanations",
        explainer="LayerGradientXActivation",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/MdjebgkdexA2ZDt/download",
        **_MNLI_BERT_KWARGS,
    ),
    ThermostatConfig(
        name="multi_nli-bert-lig",
        description="MNLI dataset, BERT model, Layer Integrated Gradients explanations",
        explainer="LayerIntegratedGradients",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/g53nbtnXaFyPLM7/download",
        **_MNLI_BERT_KWARGS,
    ),
    ThermostatConfig(
        name="multi_nli-bert-lime",
        description="MNLI dataset, BERT model, LIME explanations",
        explainer="LimeBase",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/ptspBexoHaXtqXD/download",
        **_MNLI_BERT_KWARGS,
    ),
    ThermostatConfig(
        name="multi_nli-bert-occ",
        description="MNLI dataset, BERT model, Occlusion explanations",
        explainer="Occlusion",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/cHK6YCAo4ESZ3xx/download",
        **_MNLI_BERT_KWARGS,
    ),
    ThermostatConfig(
        name="multi_nli-bert-svs",
        description="MNLI dataset, BERT model, Shapley Value Sampling explanations",
        explainer="ShapleyValueSampling",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/d5TTHCkAb5TJmbg/download",
        **_MNLI_BERT_KWARGS,
    ),
    ThermostatConfig(
        name="multi_nli-electra-lgxa",
        description="MNLI dataset, ELECTRA model, Layer Gradient x Activation explanations",
        explainer="LayerGradientXActivation",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/2HrCmp9sxJNiKBc/download",
        **_MNLI_ELECTRA_KWARGS,
    ),
    ThermostatConfig(
        name="multi_nli-electra-lig",
        description="MNLI dataset, ELECTRA model, Layer Integrated Gradients explanations",
        explainer="LayerIntegratedGradients",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/2eZnJgCbWd2D4PB/download",
        **_MNLI_ELECTRA_KWARGS,
    ),
    ThermostatConfig(
        name="multi_nli-electra-lime",
        description="MNLI dataset, ELECTRA model, LIME explanations",
        explainer="LimeBase",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/WzBwpwC9FoQZCwB/download",
        **_MNLI_ELECTRA_KWARGS,
    ),
    ThermostatConfig(
        name="multi_nli-electra-occ",
        description="MNLI dataset, ELECTRA model, Occlusion explanations",
        explainer="Occlusion",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/MGjQmKK9kynZTQt/download",
        **_MNLI_ELECTRA_KWARGS,
    ),
    ThermostatConfig(
        name="multi_nli-roberta-lgxa",
        description="MNLI dataset, RoBERTa model, Layer Gradient x Activation explanations",
        explainer="LayerGradientXActivation",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/SxDwtGCpPzi3DDz/download",
        **_MNLI_ROBERTA_KWARGS,
    ),
    ThermostatConfig(
        name="multi_nli-roberta-lig",
        description="MNLI dataset, RoBERTa model, Layer Integrated Gradients explanations",
        explainer="LayerIntegratedGradients",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/8zaTxTCijG6g7Y5/download",
        **_MNLI_ROBERTA_KWARGS,
    ),
    ThermostatConfig(
        name="multi_nli-roberta-lime",
        description="MNLI dataset, RoBERTa model, LIME explanations",
        explainer="LimeBase",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/dY4z4ptcMtiYzZs/download",
        **_MNLI_ROBERTA_KWARGS,
    ),
    ThermostatConfig(
        name="multi_nli-roberta-occ",
        description="MNLI dataset, RoBERTa model, Occlusion explanations",
        explainer="Occlusion",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/w5YSxNc6L8QZisG/download",
        **_MNLI_ROBERTA_KWARGS,
    ),
    ThermostatConfig(
        name="multi_nli-xlnet-lgxa",
        description="MNLI dataset, XLNet model, Layer Gradient x Activation explanations",
        explainer="LayerGradientXActivation",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/n79G9kf9jbNx8o7/download",
        **_MNLI_XLNET_KWARGS,
    ),
    ThermostatConfig(
        name="multi_nli-xlnet-lig",
        description="MNLI dataset, XLNet model, Layer Integrated Gradients explanations",
        explainer="LayerIntegratedGradients",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/MZr8jTnaCBdMPGe/download",
        **_MNLI_XLNET_KWARGS,
    ),
    ThermostatConfig(
        name="multi_nli-xlnet-lime",
        description="MNLI dataset, XLNet model, LIME explanations",
        explainer="LimeBase",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/B7tfLSRKBGYxJ3s/download",
        **_MNLI_XLNET_KWARGS,
    ),
    ThermostatConfig(
        name="multi_nli-xlnet-occ",
        description="MNLI dataset, XLNet model, Occlusion explanations",
        explainer="Occlusion",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/YWjJ6T7n6oeKbJJ/download",
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
    ThermostatConfig(
        name="xnli-albert-occ",
        description="XNLI dataset, ALBERT model, Occlusion explanations",
        explainer="Occlusion",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/bEg95CGBtzaFQij/download",
        **_XNLI_ALBERT_KWARGS,
    ),
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
    ThermostatConfig(
        name="xnli-bert-occ",
        description="XNLI dataset, BERT model, Occlusion explanations",
        explainer="Occlusion",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/AHoTKbtSCQ73QxN/download",
        **_XNLI_BERT_KWARGS,
    ),
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
    ThermostatConfig(
        name="xnli-roberta-occ",
        description="XNLI dataset, RoBERTa model, Occlusion explanations",
        explainer="Occlusion",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/XB2tnATQW3tbxPW/download",
        **_XNLI_ROBERTA_KWARGS,
    ),
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
    ThermostatConfig(
        name="xnli-xlnet-occ",
        description="XNLI dataset, XLNet model, Occlusion explanations",
        explainer="Occlusion",
        data_url="https://cloud.dfki.de/owncloud/index.php/s/yEFEyrq4pbGKP4s/download",
        **_XNLI_XLNET_KWARGS,
    ),
]
