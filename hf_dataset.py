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

from thermostat.data.thermostat_configs import builder_configs


# TODO: Add BibTeX citation
# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = "Coming soon."

_DESCRIPTION = "Thermostat is a large collection of NLP model explanations and accompanying analysis tools."

# TODO: Add a link to an official homepage for the dataset here
_HOMEPAGE = 'https://github.com/DFKI-NLP/thermostat'

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = ''


class Thermostat(datasets.GeneratorBasedBuilder):
    """One config (e.g. 'imdb-bert-lgxa') contains the attribution scores of a Layer Gradient x Activation explainer
    applied on the IMDb dataset that has been classified by a BERT downstream model. """

    BUILDER_CONFIGS = builder_configs

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
            description=self.config.description + f'\nDataset: {self.config.dataset}\nModel: {self.config.model}'
                                                  f'\nExplainer: {self.config.explainer}\n',
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
