# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TODO: Add a description here."""


import datasets
import json


# TODO: Add BibTeX citation
# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = "Coming soon."

_DESCRIPTION = "Coming soon."

# TODO: Add a link to an official homepage for the dataset here
_HOMEPAGE = 'https://github.com/nfelnlp/thermostat'

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = ''

_VERSION = datasets.Version('1.0.0', '')


# Base arguments for any dataset
_BASE_KWARGS = dict(
    features={
        "attributions": "attributions",
        "predictions": "predictions",
        "input_ids": "input_ids",
    },
    citation="Coming soon.",
    url="https://github.com/nfelnlp/thermostat",
)


# Base arguments for AG News dataset
_AGNEWS_KWARGS = dict(
    dataset="ag_news",
    label_classes=["World", "Sports", "Business", "Sci/Tech"],
    label_column="label",
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
    label_classes=["entailment", "neutral", "contradiction"],
    label_column="label",
    **_BASE_KWARGS,
)
_MNLI_ALBERT_KWARGS = dict(
    model="prajjwal1/albert-base-v2-mnli",
    **_IMDB_KWARGS,
)
_MNLI_BERT_KWARGS = dict(
    model="textattack/bert-base-uncased-MNLI",
    **_IMDB_KWARGS,
)
_MNLI_ELECTRA_KWARGS = dict(
    model="howey/electra-base-mnli",
    **_IMDB_KWARGS,
)
_MNLI_ROBERTA_KWARGS = dict(
    model="textattack/roberta-base-MNLI",
    **_IMDB_KWARGS,
)
_MNLI_XLNET_KWARGS = dict(
    model="textattack/xlnet-base-cased-MNLI",
    **_IMDB_KWARGS,
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
        self.data_url = data_url
        self.citation = citation
        self.url = url


class Thermostat(datasets.GeneratorBasedBuilder):
    """One config (e.g. 'imdb-bert-lgxa') contains the attribution scores of a Layer Gradient x Activation explainer
    applied on the IMDb dataset that has been classified by a BERT downstream model. """

    BUILDER_CONFIGS = [
        ThermostatConfig(
            name="ag_news-albert-lgxa",
            description="AG News dataset, ALBERT model, Layer Gradient x Activation explanations",
            explainer="LayerGradientXActivation",
            data_url="https://drive.google.com/file/d/14JcfOkfwyEWkgpj8pJPiqqDR8QLPTHZG/",
            **_AGNEWS_ALBERT_KWARGS,
        ),
        ThermostatConfig(
            name="ag_news-bert-lgxa",
            description="AG News dataset, BERT model, Layer Gradient x Activation explanations",
            explainer="LayerGradientXActivation",
            data_url="https://drive.google.com/file/d/1p2Cp1m8pOAaBiQ_9mcWQpg3KF-cEefEk/",
            **_AGNEWS_BERT_KWARGS,
        ),
        ThermostatConfig(
            name="ag_news-roberta-lgxa",
            description="AG News dataset, RoBERTa model, Layer Gradient x Activation explanations",
            explainer="LayerGradientXActivation",
            data_url="https://drive.google.com/file/d/1w4pNDGEVdKYwEjbOkKPqMuV2pneK__64/",
            **_AGNEWS_ROBERTA_KWARGS,
        ),
        ThermostatConfig(
            name="imdb-albert-lgxa",
            description="IMDb dataset, ALBERT model, Layer Gradient x Activation explanations",
            explainer="LayerGradientXActivation",
            data_url="https://cloud.dfki.de/owncloud/index.php/s/b9HFFJPFxaEy5ZX/download",
            **_IMDB_ALBERT_KWARGS,
        ),
        ThermostatConfig(
            name="imdb-albert-lime",
            description="IMDb dataset, ALBERT model, LIME explanations",
            explainer="LimeBase",
            data_url="https://drive.google.com/file/d/1UF6xVcKkQ3BNFLzv96uR6NRxKLl_jjOR/",
            **_IMDB_ALBERT_KWARGS,
        ),
        ThermostatConfig(
            name="imdb-bert-lgxa",
            description="IMDb dataset, BERT model, Layer Gradient x Activation explanations",
            explainer="LayerGradientXActivation",
            data_url="https://cloud.dfki.de/owncloud/index.php/s/nztfgJWrYgyzG5M/download",
            **_IMDB_BERT_KWARGS,
        ),
        ThermostatConfig(
            name="imdb-bert-lig",
            description="IMDb dataset, BERT model, Layer Integrated Gradients explanations",
            explainer="LayerIntegratedGradients",
            data_url="https://cloud.dfki.de/owncloud/index.php/s/Zp2HZrbxAFm8GS4/download",
            **_IMDB_BERT_KWARGS,
        ),
        ThermostatConfig(
            name="imdb-bert-occ",
            description="IMDb dataset, BERT model, Occlusion explanations",
            explainer="Occlusion",
            data_url="https://drive.google.com/file/d/1UiraVu4T_sT1WEk4QiVOyqyyxm_CzPoY/",
            **_IMDB_BERT_KWARGS,
        ),
        ThermostatConfig(
            name="imdb-electra-lgxa",
            description="IMDb dataset, ELECTRA model, Layer Gradient x Activation explanations",
            explainer="LayerGradientXActivation",
            data_url="https://drive.google.com/file/d/1PousVJBGrCSQRK72aK82BlmyV7JZ2LiA/",
            **_IMDB_ELECTRA_KWARGS,
        ),
        ThermostatConfig(
            name="imdb-electra-occ",
            description="IMDb dataset, ELECTRA model, Occlusion explanations",
            explainer="Occlusion",
            data_url="https://drive.google.com/file/d/1rnQ0cj6csGW_UvJiD7gCrtBxK_YPdUtR/",
            **_IMDB_ELECTRA_KWARGS,
        ),
        ThermostatConfig(
            name="imdb-roberta-lgxa",
            description="IMDb dataset, RoBERTa model, Layer Gradient x Activation explanations",
            explainer="LayerGradientXActivation",
            data_url="https://drive.google.com/file/d/1tBtkNfywAM6yDbxuE_vd6Zh4C0pgCGJi/",
            **_IMDB_ROBERTA_KWARGS,
        ),
        ThermostatConfig(
            name="imdb-roberta-occ",
            description="IMDb dataset, RoBERTa model, Occlusion explanations",
            explainer="Occlusion",
            data_url="https://drive.google.com/file/d/1ZOaoR2VTM73LV7dSve6-l0mLRyM0Y9_V/",
            **_IMDB_ROBERTA_KWARGS,
        ),
        ThermostatConfig(
            name="imdb-xlnet-lgxa",
            description="IMDb dataset, XLNet model, Layer Gradient x Activation explanations",
            explainer="LayerGradientXActivation",
            data_url="https://drive.google.com/file/d/1G4if9U7Em9lelYS4y8dZpnOlr3bTcfgY/",
            **_IMDB_XLNET_KWARGS,
        ),
        ThermostatConfig(
            name="imdb-xlnet-lig",
            description="IMDb dataset, XLNet model, Layer Integrated Gradients explanations",
            explainer="LayerIntegratedGradients",
            data_url="https://drive.google.com/file/d/1WuQJP5A3KtrFNkBdE3-z5ELSfZ-Q2MZY/",
            **_IMDB_XLNET_KWARGS,
        ),
        ThermostatConfig(
            name="imdb-xlnet-lime",
            description="IMDb dataset, XLNet model, LIME explanations",
            explainer="LimeBase",
            data_url="https://cloud.dfki.de/owncloud/index.php/s/KQWoPRktycR93Kf/download",
            **_IMDB_XLNET_KWARGS,
        ),
        ThermostatConfig(
            name="mnli-albert-lgxa",
            description="MNLI dataset, ALBERT model, Layer Gradient x Activation explanations",
            explainer="LayerGradientXActivation",
            data_url="https://drive.google.com/file/d/1umzL_4rjMh-gf-5SQQJh-NRslOa6PEA2/",
            **_MNLI_ALBERT_KWARGS,
        ),
        ThermostatConfig(
            name="mnli-albert-occ",
            description="MNLI dataset, ALBERT model, Occlusion explanations",
            explainer="Occlusion",
            data_url="https://drive.google.com/file/d/1qyjHwGwbTu_otiopsTQyDV5NmMqaNIeD/",
            **_MNLI_ALBERT_KWARGS,
        ),
        ThermostatConfig(
            name="mnli-bert-lgxa",
            description="MNLI dataset, BERT model, Layer Gradient x Activation explanations",
            explainer="LayerGradientXActivation",
            data_url="https://drive.google.com/file/d/1cacYK4UBtu3JefTg3Nk8uvonYGdsCefm/",
            **_MNLI_BERT_KWARGS,
        ),
        ThermostatConfig(
            name="mnli-bert-lime",
            description="MNLI dataset, BERT model, LIME explanations",
            explainer="LimeBase",
            data_url="https://drive.google.com/file/d/1BCEYMCP91E4JsZaIGc39g6w8gFbyl9HF/",
            **_MNLI_BERT_KWARGS,
        ),
        ThermostatConfig(
            name="mnli-electra-lgxa",
            description="MNLI dataset, ELECTRA model, Layer Gradient x Activation explanations",
            explainer="LayerGradientXActivation",
            data_url="https://drive.google.com/file/d/1yrhbxU9DDyeJRkb4g_pPuAtKV1U3k-YX/",
            **_MNLI_ELECTRA_KWARGS,
        ),
        ThermostatConfig(
            name="mnli-electra-lig",
            description="MNLI dataset, ELECTRA model, Layer Integrated Gradients explanations",
            explainer="LayerIntegratedGradients",
            data_url="https://drive.google.com/file/d/1DLw-4KYkDmKpHFseUjeDEbd64qv3odB6/",
            **_MNLI_ELECTRA_KWARGS,
        ),
        ThermostatConfig(
            name="mnli-roberta-lgxa",
            description="MNLI dataset, RoBERTa model, Layer Gradient x Activation explanations",
            explainer="LayerGradientXActivation",
            data_url="https://drive.google.com/file/d/17sgI9S0skCXLTI3II0TWK4u_jFVJVSJ1/",
            **_MNLI_ROBERTA_KWARGS,
        ),
        ThermostatConfig(
            name="mnli-roberta-lig",
            description="MNLI dataset, RoBERTa model, Layer Integrated Gradients explanations",
            explainer="LayerIntegratedGradients",
            data_url="https://drive.google.com/file/d/1tS5-WyFM6LUz0O7fXdFDqZpT6gA4PeEj/",
            **_MNLI_ROBERTA_KWARGS,
        ),
        ThermostatConfig(
            name="mnli-roberta-lime",
            description="MNLI dataset, RoBERTa model, LIME explanations",
            explainer="LimeBase",
            data_url="https://drive.google.com/file/d/1nbd4cdEhPVpHF1e9RU6sj3sgvkgfmK1I/",
            **_MNLI_ROBERTA_KWARGS,
        ),
        ThermostatConfig(
            name="mnli-xlnet-lgxa",
            description="MNLI dataset, XLNet model, Layer Gradient x Activation explanations",
            explainer="LayerGradientXActivation",
            data_url="https://drive.google.com/file/d/1T6nF_0II6jtXVjuOAYKVnQ4WTn6550sP/",
            **_MNLI_XLNET_KWARGS,
        ),
        ThermostatConfig(
            name="mnli-xlnet-lime",
            description="MNLI dataset, XLNet model, LIME explanations",
            explainer="LimeBase",
            data_url="https://drive.google.com/file/d/1c3XoRAcl90IjSZQYrONoOicGMv8NdA1s/",
            **_MNLI_XLNET_KWARGS,
        ),
        ThermostatConfig(
            name="mnli-xlnet-occ",
            description="MNLI dataset, XLNet model, Occlusion explanations",
            explainer="Occlusion",
            data_url="https://drive.google.com/file/d/1qgzjE9UgBpNUaPFBqe8fiXKQqD49CRuW/",
            **_MNLI_XLNET_KWARGS,
        ),
    ]

    def _info(self):
        features = {}
        if self.config.label_classes:
            features["label"] = datasets.features.ClassLabel(names=self.config.label_classes)
        else:
            features["label"] = datasets.Value("float32")
        features["idx"] = datasets.Value("int32")

        # Thermostat-specific fields: Explainer outputs
        features["attributions"] = datasets.Sequence(datasets.Value("float32"))
        features["predictions"] = datasets.Sequence(datasets.Value("float32"))
        features["input_ids"] = datasets.Sequence(datasets.Value("int32"))

        return datasets.DatasetInfo(
            description=self.config.description + f'\nExplainer: {self.config.explainer}\nModel: {self.config.model}'
                                                  f'\nDataset: {self.config.dataset}\n',
            features=datasets.Features(features),
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    @staticmethod
    def _get_drive_url(url):
        base_url = 'https://drive.google.com/uc?id='
        split_url = url.split('/')
        return base_url + split_url[5]

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # This method is tasked with downloading/extracting the data and defining the splits depending on the
        #  configuration
        if self.config.data_url.startswith("https://drive.google.com"):
            dl_path = dl_manager.download_and_extract(self._get_drive_url(self.config.data_url))
        else:
            dl_path = dl_manager.download(self.config.data_url)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "data_file": dl_path,
                    "split": "test",
                }
            )
        ]

    def _generate_examples(
        self, data_file, split  # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    ):
        """ Yields examples as (key, example) tuples. """
        # This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.

        with open(data_file, encoding="utf-8") as f:
            for id_, row in enumerate(f):
                data = json.loads(row)
                example = {feat: data[col] for feat, col in self.config.features.items()}
                example["idx"] = id_
                example["label"] = data["label"]
                yield example["idx"], example
