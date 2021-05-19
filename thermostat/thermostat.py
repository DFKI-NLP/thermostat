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

# Base arguments for IMDb dataset
_IMDB_KWARGS = dict(
    dataset="imdb",
    label_classes=["neg", "pos"],
    label_column="label",
    **_BASE_KWARGS,
)

_IMDB_BERT_KWARGS = dict(
    model="textattack/bert-base-uncased-imdb",
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
                                                  f'\nDataset: {self.config.dataset}',
            features=datasets.Features(features),
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # This method is tasked with downloading/extracting the data and defining the splits depending on the
        #  configuration
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
