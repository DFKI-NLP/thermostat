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
import textwrap


# TODO: Add BibTeX citation
# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
Coming soon.
"""

# TODO: Add a link to an official homepage for the dataset here
_HOMEPAGE = ''

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = ''

_EXPLAINERS = [  # TODO: Uncomment later
    "LayerGradientXActivation",  # "LayerIntegratedGradients", "LIME", "Occlusion", "ShapleyValueSampling"
]


# Base arguments for IMDb dataset
_IMDB_BASE_KWARGS = dict(
    features={
        "attributions": "attributions",
        "predictions": "predictions",
        "input_ids": "input_ids",
    },
    label_classes=["neg", "pos"],
    label_column="label",
    data_url="https://cloud.dfki.de/owncloud/index.php/s/nztfgJWrYgyzG5M/download",  # TODO
    citation="",
    url="",
)


class ThermostatConfig(datasets.BuilderConfig):
    """ BuilderConfig for Thermostat """

    def __init__(
        self,
        classifier,
        features,
        label_column,
        label_classes,
        data_url,
        citation,
        url,
        **kwargs,
    ):
        """
        Args:
            classifier: Downstream model (e.g. "bert"),
            features,
            label_column,
            label_classes,
            data_url,
            citation,
            url,
        """
        super(ThermostatConfig, self).__init__(version=datasets.Version('1.0.0', ''), **kwargs)
        self.classifier = classifier
        self.features = features
        self.label_column = label_column
        self.label_classes = label_classes
        self.data_url = data_url
        self.citation = citation
        self.url = url


class Thermostat(datasets.GeneratorBasedBuilder):
    """One config (e.g. 'imdb_bert') contains the attribution scores of all available explainers applied on the IMDb
    dataset classified by a BERT downstream model. """

    BUILDER_CONFIGS = [
        ThermostatConfig(
            name='imdb_bert',
            description=textwrap.dedent(
                """\
                IMDb dataset explanations by a BERT model
                """
            ),
            classifier='textattack/bert-base-uncased-imdb',
            **_IMDB_BASE_KWARGS),
    ]

    def _info(self):
        self.description = ''

        self.features = {
            "attributions": None,
            "predictions": None,
            "input_ids": None,
        }
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
            description=self.description,
            features=datasets.Features(features),
            homepage=self.config.url,
            citation=self.config.citation + "\n" + _CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # This method is tasked with downloading/extracting the data and defining the splits depending on the
        #  configuration
        # TODO: Decide if the downloaded file is a .zip folder containing multiple files or not
        # For debugging, dl_path variable is a single .json file
        dl_path = dl_manager.download(self.config.data_url)
        return [
            datasets.SplitGenerator(
                name=explainer,
                gen_kwargs={
                    "data_file": dl_path,
                    "split": explainer,
                }
            )
            for explainer in _EXPLAINERS
        ]

    def _generate_examples(
        self, data_file, split  # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    ):
        """ Yields examples as (key, example) tuples. """
        # This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.
        # The `key` is here for legacy reason (tfds) and is not important in itself.

        with open(data_file, encoding="utf-8") as f:
            for id_, row in enumerate(f):
                data = json.loads(row)  # TODO: Unpack data
                if id_ == 0:  # TODO: Only once per .jsonl file in the future
                    self.dataset_config = data['dataset']
                    self.model_config = data['model']
                    self.explainer_config = data['explainer']
                    # Construct description from various fields
                    self.description = f'{self.dataset_config}\n{self.model_config}\n{self.explainer_config}'
                example = {feat: data[col] for feat, col in self.config.features.items()}
                example["idx"] = id_
                example["label"] = data["label"]
                # TODO: Use 'split' for selecting explainer subset
                yield example["idx"], example
