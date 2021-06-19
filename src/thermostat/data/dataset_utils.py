from collections import defaultdict

import datasets
from datasets import Dataset
from itertools import groupby
from overrides import overrides
from spacy import displacy
from tqdm import tqdm
from transformers import AutoTokenizer
from typing import Dict, List

from thermostat.data import thermostat_configs
from thermostat.data.tokenization import fuse_subwords
from thermostat.utils import lazy_property
from thermostat.visualize import Sequence, normalize_attributions


def list_configs():
    """ Returns the list of names of all available configs in the Thermostat HF dataset"""
    return [config.name for config in thermostat_configs.builder_configs]


def get_config(config_name):
    """ based on : https://stackoverflow.com/a/7125547 """
    return next((x for x in thermostat_configs.builder_configs if x.name == config_name), None)


def get_text_fields(config_name):
    text_fields = get_config(config_name).text_column
    if type(text_fields) != list:
        text_fields = [text_fields]
    return text_fields


def load(config_str: str = None):
    assert config_str, f'Please enter a config. Available options: {list_configs()}.'

    def load_from_single_config(config):
        print(f'Loading Thermostat configuration: {config}')
        return datasets.load_dataset("hf_dataset.py", config, split="test")

    if config_str in list_configs():
        data = load_from_single_config(config_str)

    elif config_str in ['-'.join(c.split('-')[:2]) for c in list_configs()]:
        # Resolve "dataset+model" to all explainer subsets
        raise NotImplementedError()

    elif config_str in [f'{c.split("-")[0]}-{c.split("-")[-1]}' for c in list_configs()]:
        # Resolve "dataset+explainer" to all model subsets
        raise NotImplementedError()

    else:
        raise ValueError(f'Invalid config. Available options: {list_configs()}')

    return Thermopack(data)


def get_coordinate(thermostat_dataset: Dataset, coordinate: str) -> str:
    """ Determine a coordinate (dataset, model, or explainer) of a Thermostat dataset from its description """
    assert coordinate in ['Model', 'Dataset', 'Explainer']
    coord_prefix = f'{coordinate}: '
    assert coord_prefix in thermostat_dataset.description
    str_post_coord_prefix = thermostat_dataset.description.split(coord_prefix)[1]
    if '\n' in str_post_coord_prefix:
        coord_value = str_post_coord_prefix.split('\n')[0]
    else:
        coord_value = str_post_coord_prefix
    return coord_value


class ThermopackMeta(type):
    """ Inspired by: https://stackoverflow.com/a/65917858 """
    def __new__(mcs, name, bases, dct):
        child = super().__new__(mcs, name, bases, dct)
        for base in bases:
            for field_name, field in base.__dict__.items():
                if callable(field) and not field_name.startswith('__'):
                    setattr(child, field_name, mcs.force_child(field, field_name, base, child))
        return child

    @staticmethod
    def force_child(fun, fun_name, base, child):
        """Turn from Base- to Child-instance-returning function."""
        def wrapper(*args, **kwargs):
            result = fun(*args, **kwargs)
            if not result:
                # Ignore if returns None
                return None
            if type(result) == base:
                print(fun_name)
                # Return Child instance if the Base method tries to return Base instance.
                return child(result)
            return result
        return wrapper


class Thermopack(Dataset, metaclass=ThermopackMeta):
    def __init__(self, hf_dataset):
        super().__init__(hf_dataset.data, info=hf_dataset.info, split=hf_dataset.split,
                         indices_table=hf_dataset._indices)
        self.dataset = hf_dataset

        # Model
        self.model_name = get_coordinate(hf_dataset, 'Model')

        # Dataset
        self.dataset_name = get_coordinate(hf_dataset, 'Dataset')
        self.label_names = hf_dataset.info.features['label'].names

        # Align label indices (some MNLI and XNLI models have a different order in the label names)
        label_classes = get_config(self.config_name).label_classes
        if label_classes != self.label_names:
            self.dataset = self.dataset.map(lambda instance: {
                'label': label_classes.index(self.label_names[instance['label']])})
            print(f'Reordered labels from {self.label_names} to {label_classes}')
            self.label_names = label_classes

        # Explainer
        self.explainer_name = get_coordinate(hf_dataset, 'Explainer')

    @lazy_property
    def tokenizer(self):
        print(f'Loading the tokenizer for model: {self.model_name}')
        return AutoTokenizer.from_pretrained(self.model_name)

    @lazy_property
    def units(self):
        units = []
        for instance in tqdm(self.dataset, desc=f'Preparing instances (Thermounits) for {self.config_name}'):
            # Decode labels and predictions
            true_label_index = instance['label']
            true_label = {'index': true_label_index,
                          'name': self.label_names[true_label_index]}

            predicted_label_index = instance['predictions'].index(max(instance['predictions']))
            predicted_label = {'index': predicted_label_index,
                               'name': self.label_names[predicted_label_index]}

            units.append(Thermounit(
                instance, true_label, predicted_label,
                self.model_name, self.dataset_name, self.explainer_name, self.tokenizer, self.config_name))
        return units

    @overrides
    def __getitem__(self, idx):
        """ Indexing a Thermopack returns a Thermounit """
        return self.units[idx]

    @overrides
    def __iter__(self):
        for unit in self.units:
            yield unit

    @overrides
    def __str__(self):
        return self.info.description


class Thermounit:
    """ Processed single instance of a Thermopack (Thermostat dataset/configuration) """
    def __init__(self, instance, true_label, predicted_label, model_name, dataset_name, explainer_name, tokenizer,
                 config_name):
        self.instance = instance
        self.index = self.instance['idx']
        self.attributions = self.instance['attributions']
        self.true_label = true_label
        self.predicted_label = predicted_label
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.explainer_name = explainer_name
        self.tokenizer = tokenizer
        self.config_name = config_name
        self.text_fields: List = []

    @property
    def tokens(self) -> Dict:
        # "tokens" includes all special tokens, later used for the heatmap when aligning with attributions
        tokens = self.tokenizer.convert_ids_to_tokens(self.instance['input_ids'])
        # Make token index
        tokens_enum = dict(enumerate(tokens))
        return tokens_enum

    def fill_text_fields(self):
        if self.text_fields:
            return

        # Determine groups of tokens split by [SEP] tokens
        text_groups = []
        for group in [list(g) for k, g in groupby(self.tokens.items(),
                                                  lambda kt: kt[1] != self.tokenizer.sep_token) if k]:
            # Remove groups that only contain special tokens
            if len([t for t in group if t[1] in self.tokenizer.all_special_tokens]) < len(group):
                text_groups.append(group)

        setattr(self, 'text_fields', get_text_fields(self.config_name))
        # Assign text field values based on groups
        for text_field, field_tokens in zip(self.text_fields, text_groups):
            setattr(self, text_field, [t for t in field_tokens if t[1] not in self.tokenizer.all_special_tokens])

    @lazy_property
    def texts(self):
        self.fill_text_fields()

        return TextFieldsDict({text_field: getattr(self, text_field) for text_field in self.text_fields})

    @lazy_property
    def heatmap(self, gamma=1.0, normalize=True, flip_attributions_idx=0, fuse_subwords_strategy='salient'):
        """ Generate a list of tuples in the form of <token,color> for a single data point of a Thermostat dataset """
        self.fill_text_fields()

        atts = self.attributions

        if normalize:
            atts = normalize_attributions(atts)

        if flip_attributions_idx == self.predicted_label['index']:
            atts = [att * -1 for att in atts]

        heatmaps = {}
        for text_field, tokens_enum in self.texts.items():
            # Select attributions according to token indices (tokens_enum keys)
            selected_atts = [atts[idx] for idx in [t[0] for t in tokens_enum]]
            tokens = [t[1] for t in tokens_enum]

            if fuse_subwords_strategy:
                tokens, selected_atts = fuse_subwords(tokens, selected_atts, self.tokenizer,
                                                      strategy=fuse_subwords_strategy)
                setattr(self, text_field, tokens)

            sequence = Sequence(words=tokens, scores=selected_atts)
            hm = sequence.words_rgb(token_pad=self.tokenizer.pad_token,
                                    position_pad=self.tokenizer.padding_side,
                                    gamma=gamma)
            heatmaps[text_field] = hm
        return heatmaps

    def render(self, attribution_labels=False, jupyter=False):
        """ Uses the displaCy visualization tool to render a HTML from the heatmap """

        full_html = ''
        for field_name, text_field_heatmap, in self.heatmap.items():
            ents = []
            colors = {}
            ii = 0
            for token_rgb in text_field_heatmap:
                token, rgb = token_rgb.values()
                att_rounded = str(rgb.score)

                ff = ii + len(token)

                # One entity in displaCy contains start and end markers (character index) and optionally a label
                # The label can be added by setting "attribution_labels" to True
                ent = {
                    'start': ii,
                    'end': ff,
                    'label': att_rounded,
                }

                ents.append(ent)
                # A "colors" dict takes care of the mapping between attribution labels and hex colors
                colors[att_rounded] = rgb.hex
                ii = ff

            to_render = {
                'text': ''.join([t['token'] for t in text_field_heatmap]),
                'ents': ents,
            }

            if attribution_labels:
                template = """
                <mark class="entity" style="background: {bg}; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 2; 
                border-radius: 0.35em; box-decoration-break: clone; -webkit-box-decoration-break: clone">
                    {text}
                    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: 
                    uppercase; vertical-align: middle; margin-left: 0.5rem">{label}</span>
                </mark>
                """
            else:
                template = """
                <mark class="entity" style="background: {bg}; padding: 0.15em 0.3em; margin: 0 0.2em; line-height: 2.2;
                border-radius: 0.25em; box-decoration-break: clone; -webkit-box-decoration-break: clone">
                    {text}
                </mark>
                """

            html = displacy.render(
                to_render,
                style='ent',
                manual=True,
                jupyter=jupyter,
                options={'template': template,
                         'colors': colors,
                         }
            )
            if jupyter:
                html = displacy.render(
                    to_render,
                    style='ent',
                    manual=True,
                    jupyter=False,
                    options={'template': template,
                             'colors': colors,
                             }
                )
            full_html += html
        return full_html if not jupyter else None


class TextFieldsDict(object):
    def __init__(self, contents):
        self.contents = contents
        self.fields = list(self.contents.keys())

    def __len__(self):
        return sum([len(getattr(self, text_field)) for text_field in self.fields])

    def __str__(self):
        return '\n'.join([f'{kv[0]}: {" ".join([it[1] for it in kv[1]])}' for kv in self.contents.items()])

    def items(self):
        return self.contents.items()


def avg_attribution_stat(thermostat_dataset: Dataset) -> List:
    """ Given a Thermostat dataset, calculate the average attribution for each token across the whole dataset """
    model_id = get_coordinate(thermostat_dataset, coordinate='Model')
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    token_atts = defaultdict(list)
    for row in thermostat_dataset:
        for input_id, attribution_score in zip(row['input_ids'], row['attributions']):
            # Distinguish between the labels
            if row['label'] == 0:
                # Add the negative attribution score for label 0
                # to the list of attribution scores of a single token
                token_atts[tokenizer.decode(input_id)].append(-attribution_score)
            else:
                token_atts[tokenizer.decode(input_id)].append(attribution_score)

    avgs = defaultdict(float)
    # Calculate the average attribution score from the list of attribution scores of each token
    for token, scores in token_atts.items():
        avgs[token] = sum(scores)/len(scores)
    return sorted(avgs.items(), key=lambda x: x[1], reverse=True)


def explainer_agreement_stat(thermostat_datasets: List) -> List:
    """ Calculate agreement on token attribution scores between multiple Thermostat datasets/explainers """
    assert len(thermostat_datasets) > 1
    all_explainers_atts = {}
    for td in thermostat_datasets:
        assert type(td) == Dataset
        explainer_id = get_coordinate(td, coordinate='Explainer')
        # Add all attribution scores to a dictionary with the key being the name of the explainer
        all_explainers_atts[explainer_id] = td['attributions']

    model_id = get_coordinate(thermostat_datasets[0], coordinate='Model')
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Dissimilarity dict for tokens and their contexts
    tokens_dissim = {}
    for row in zip(thermostat_datasets[0]['input_ids'],
                   *list(all_explainers_atts.values())):
        # Decode all tokens of one data point
        tokens = tokenizer.decode(list(row)[0], skip_special_tokens=True)
        for idx, input_id in enumerate(zip(*list(row))):
            if list(input_id)[0] in tokenizer.all_special_ids:
                continue

            att_explainers = list(input_id)[1:]
            max_att = max(att_explainers)
            min_att = min(att_explainers)

            # Key: All tokens (context), single token in question, index of token in context
            tokens_dissim[(tokenizer.decode(list(input_id)[0]), tokens, idx)]\
                = {'dissim': max_att - min_att,  # Maximum difference in attribution
                   'atts': dict(zip(all_explainers_atts.keys(), att_explainers))}
    return sorted(tokens_dissim.items(), key=lambda x: x[1]['dissim'], reverse=True)
