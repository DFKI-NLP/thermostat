import numpy as np
import os
# new change 
from sys import platform
# new change
from datasets import Dataset, load_dataset
from itertools import groupby
from overrides import overrides
from sklearn.metrics import classification_report
from tqdm import tqdm
from transformers import AutoTokenizer
from typing import Dict, List

from thermostat.data import additional_configs, thermostat_configs
from thermostat.data.tokenization import fuse_subwords
from thermostat.utils import lazy_property
from thermostat.visualize import ColorToken, Heatmap, normalize_attributions


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
            if result is None:
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
        # Init Dataset super class
        super().__init__(hf_dataset.data, info=hf_dataset.info, split=hf_dataset.split,
                         indices_table=hf_dataset._indices)

        # Model
        self.model_name = get_coordinate(hf_dataset, 'Model')
        # Dataset
        self.dataset_name = get_coordinate(hf_dataset, 'Dataset')
        # Explainer
        self.explainer_name = get_coordinate(hf_dataset, 'Explainer')

        self.dataset: Dataset = hf_dataset
        self.units: List = [PlaceholderThermounit(unit['attributions'],
                                                  unit['idx'],
                                                  unit['input_ids'],
                                                  unit['label'],
                                                  unit['predictions']) for unit in self.dataset]
        self.legacy_label_names = get_config(self.config_name).label_classes

    def __getattr__(self, name):
        """ Only gets called if an attribute that is not part of the Thermopack is accessed.
         In this case, if the name of the requested attribute is one of the five attributes of a Thermounit,
         return the respective entries of all units as a list """
        if name in list(self.units[0].__dict__.keys()):
            return ThermounitAttributeArray([getattr(u, name) for u in self.units])
        else:
            raise AttributeError

    @overrides
    def __getitem__(self, idx):
        """ Indexing a Thermopack by an integer instantiates a Thermounit and returns it.
            Indexing by string returns the associated column if its exists (similar to HF datasets). """

        if isinstance(idx, str):
            """ String indexing """
            if idx in list(self.units[0].__dict__.keys()):
                """ Same behaviour as __getattr__: Return the requested attribute of all units as a list. """
                return ThermounitAttributeArray([getattr(u, idx) for u in self.units])
            elif not any([isinstance(unit, Thermounit) for unit in self.units]):
                print(f'The instances (Thermounits) of this Thermopack are placeholders and have to be fully processed '
                      f'before all attributes and metadata are available.')
                self.decode()
                return self[idx]  # Recursion
            raise KeyError(f'Not a valid slice or column name: {idx}')

        elif isinstance(idx, slice):
            """ Slicing """
            data_indices = range(len(self))
            slice_indices = data_indices[slice(idx.start, idx.stop, idx.step)]
            return Thermopack(self.dataset.select(slice_indices))

        elif not isinstance(self.units[idx], Thermounit):
            """ Decode labels and predictions """
            instance = self.units[idx]

            # Overwrite true (HF dataset) label indices with custom label indices from downstream model
            #  (some MNLI and XNLI models have a different order in the label names)
            true_label_index = self.label_names.index(self.legacy_label_names[instance.label])
            true_label = {'index': true_label_index,
                          'name': self.label_names[true_label_index]}

            predicted_label_index = instance.predictions.index(max(instance.predictions))
            predicted_label = {'index': predicted_label_index,
                               'name': self.label_names[predicted_label_index]}

            tunit = Thermounit(instance, true_label, predicted_label, self.model_name, self.dataset_name,
                               self.explainer_name, self.tokenizer, self.config_name)
            self.units[idx] = tunit
        return self.units[idx]

    @overrides
    def __iter__(self):
        """ Yields Thermounit instances """
        for unit_index in range(len(self)):  # length of dataset
            yield self[unit_index]  # uses __getitem__

    @overrides
    def __str__(self):
        return self.info.description

    @property
    def label_names(self):
        gold_label_names = additional_configs.get_label_names(self.config_name)
        if gold_label_names:
            return gold_label_names
        else:
            return self.legacy_label_names

    def decode(self):
        """ Replace all PlaceholderThermounit instances with fully processed Thermounit instances """
        for unit in tqdm(self, desc='Decoding Thermounits', total=len(self)):
            u = unit

    @lazy_property
    def tokenizer(self):
        """ Initializes the tokenizer from the model name """
        return AutoTokenizer.from_pretrained(self.model_name)

    def accuracy(self):
        return sum([u_i.true_label == u_i.predicted_label for u_i in self])/len(self)

    def classification_report(self):
        """ Uses sklearn to print the confusion matrix """
        y_true = self['true_label_index']
        y_pred = self['predicted_label_index']
        print(classification_report(y_true, y_pred, target_names=self.label_names))

    def true_pred_counter(self):
        from collections import Counter
        if 'true_label_index' not in self.units[0].__dict__:
            self.decode()
        return Counter([(m_i.true_label_index, m_i.predicted_label_index) for m_i in self.units])


class PlaceholderThermounit:
    """ Raw single instance of a Thermopack. Accessing units of a Thermopack will automatically cast these to
     Thermounits. This class exists for efficiency purposes. Properly processing an entire dataset while loading it
     takes too long. """
    def __init__(self, attributions, idx, input_ids, label, predictions):
        self.attributions = attributions
        self.idx = idx
        self.input_ids = input_ids
        self.label = label
        self.predictions = predictions


class ThermounitAttributeArray(np.ndarray):
    """ NumPy Array of a list of attribute values of Thermopack units
     Follows: https://numpy.org/devdocs/user/basics.subclassing.html"""
    def __new__(cls, array, info=None):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(array).view(cls)
        # add the new attribute to the created instance
        obj.info = info
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.info = getattr(obj, 'info', None)


class Thermounit(PlaceholderThermounit):
    """ Processed single instance of a Thermopack (Thermostat dataset/configuration) """
    def __init__(self, instance, true_label, predicted_label, model_name, dataset_name, explainer_name, tokenizer,
                 config_name):
        if not isinstance(instance, Thermounit):
            super().__init__(*instance.__dict__.values())
        for key in instance.__dict__:
            setattr(self, key, instance.__dict__[key])
        self.true_label = true_label['name']
        self.true_label_index = true_label['index']
        self.predicted_label = predicted_label['name']
        self.predicted_label_index = predicted_label['index']
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.explainer_name = explainer_name
        self.tokenizer = tokenizer
        self.config_name = config_name

    def __len__(self):
        return len(self.explanation)

    @property
    def tokens(self) -> Dict:
        # "tokens" includes all special tokens, later used for the heatmap when aligning with attributions
        tokens = self.tokenizer.convert_ids_to_tokens(self.input_ids)
        # Make token index
        tokens_enum = dict(enumerate(tokens))
        return tokens_enum

    @property
    def text_fields(self):
        # Set text_fields attribute, e.g. containing "premise" and "hypothesis"
        return get_text_fields(self.config_name)

    def fill_text_fields(self, fuse_subwords_strategy='salient'):
        """ Use detokenizer to fill text fields """

        # Determine groups of tokens split by [SEP] tokens
        text_groups = []
        for group in [list(g) for k, g in groupby(self.tokens.items(),
                                                  lambda kt: kt[1] != self.tokenizer.sep_token) if k]:
            # Remove groups that only contain special tokens
            if len([t for t in group if t[1] not in self.tokenizer.all_special_tokens]) < len(group):
                text_groups.append(group)

        # Assign text field values based on groups
        for text_field, field_tokens in zip(self.text_fields, text_groups):
            # Create new list containing all non-special tokens
            non_special_tokens_enum = [t for t in field_tokens if t[1] not in self.tokenizer.all_special_tokens]
            # Select attributions according to token indices (tokens_enum keys)

            selected_atts = [self.attributions[idx] for idx in [t[0] for t in non_special_tokens_enum]]
            if fuse_subwords_strategy:
                tokens_enum, atts = fuse_subwords(non_special_tokens_enum, selected_atts, self.tokenizer,
                                                  strategy=fuse_subwords_strategy)
            else:
                tokens_enum, atts = non_special_tokens_enum, selected_atts

            assert (len(tokens_enum) == len(atts))
            # Cast each token into ColorToken objects with default color white which can later be overwritten
            # by a Heatmap object
            color_tokens = [ColorToken(token=token_enum[1],
                                       attribution=att,
                                       text_field=text_field,
                                       token_index=token_enum[0],
                                       thermounit_vars=vars(self))
                            for token_enum, att in zip(tokens_enum, atts)]

            # Set class attribute with the name of the text field
            setattr(self, text_field, Heatmap(color_tokens))

        # Introduce a texts attribute that also stores all assigned text fields into a dict with the key being the
        # name of each text field
        setattr(self, 'texts', {text_field: getattr(self, text_field) for text_field in self.text_fields})

    @property
    def explanation(self, keep_padding_tokens=False):
        """ Token-attribution tuples of a Thermounit """
        if keep_padding_tokens:
            tokens = self.tokens
        else:
            tokens = [(idx, token) for idx, token in self.tokens.items() if token != self.tokenizer.pad_token]
        attributions = [att for i, att in enumerate(self.attributions) if i in [t[0] for t in tokens]]
        token_att_tuples = list(zip([t[1] for t in tokens], attributions, [t[0] for t in tokens]))

        return token_att_tuples

    @property
    def heatmap(self, gamma=1.0, normalize=True, flip_attributions_idx=None, fuse_subwords_strategy='salient'):
        """ Generate a heatmap from explanation (!) data (without instantiating text fields)
         for a single data point of a Thermostat dataset """

        # Handle attributions, apply normalization and sign flipping if needed
        atts = [x[1] for x in self.explanation]
        if normalize:
            atts = normalize_attributions(atts)
        if flip_attributions_idx == self.predicted_label:
            atts = [att * -1 for att in atts]

        non_pad_tokens_enum = [tuple(x[i] for i in [2, 0]) for x in self.explanation]
        if fuse_subwords_strategy:
            tokens_enum, atts = fuse_subwords(non_pad_tokens_enum, atts, self.tokenizer,
                                              strategy=fuse_subwords_strategy)
        else:
            tokens_enum, atts = non_pad_tokens_enum, atts

        assert (len(tokens_enum) == len(atts))
        # Cast each token into ColorToken objects with default color white which can later be overwritten
        # by a Heatmap object
        color_tokens = [ColorToken(token=token_enum[1],
                                   attribution=att,
                                   text_field='text',
                                   token_index=token_enum[0],
                                   thermounit_vars=vars(self))
                        for token_enum, att in zip(tokens_enum, atts)]
        return Heatmap(color_tokens=color_tokens, attributions=atts, gamma=gamma)

    def render(self, labels=False):
        self.heatmap.render(labels=labels)


def list_configs():
    """ Returns the list of names of all available configs in the Thermostat HF dataset"""
    return [config.name for config in thermostat_configs.builder_configs]


def get_config(config_name):
    """ Returns a ThermostatConfig if a config exists by the name of `config_name`, else returns None
     based on : https://stackoverflow.com/a/7125547 """
    return next((x for x in thermostat_configs.builder_configs if x.name == config_name), None)


def get_text_fields(config_name):
    """ Returns a list of the text fields in a Thermostat config """
    text_fields = get_config(config_name).text_column
    if type(text_fields) != list:
        text_fields = [text_fields]
    return text_fields


def load(config_str: str = None, **kwargs) -> Thermopack:
    """
    Wrapper around the load_dataset method from the HF datasets library:
    https://huggingface.co/docs/datasets/package_reference/loading_methods.html#datasets.load_dataset
    :param config_str: equivalent to the second argument (`name`) of `datasets.load_dataset`. The value has to be one of
    the available configs in `thermostat.data.thermostat_configs.builder_configs` (accessible via `list_configs()`).
    :param kwargs: Additional keywords will all be passed to `datasets.load_dataset`. `path`, `name` and `split` are
    already reserved.
    """
    assert config_str, f'Please enter a config. Available options: {list_configs()}'
    assert config_str in list_configs(), f'Invalid config. Available options: {list_configs()}'

    """ Following https://stackoverflow.com/a/23430335/6788442 """
    ld_kwargs = {key: value for key, value in kwargs.items() if
                 key in load_dataset.__code__.co_varnames and key not in ['path', 'name', 'split']}

    print(f'Loading Thermostat configuration: {config_str}')
    if ld_kwargs:
        print(f'Additional parameters for loading: {ld_kwargs}')
    # new change
    if platform == "win32":
       dataset_script_path = os.path.dirname(os.path.realpath(__file__)).replace('\\thermostat\\data',
                                                                              '\\thermostat\\dataset.py')
    else:
        dataset_script_path = os.path.dirname(os.path.realpath(__file__)).replace('/thermostat/data',
                                                                              '/thermostat/dataset.py')
    # new change
    
    data = load_dataset(path=dataset_script_path,
                        name=config_str, split="test", **ld_kwargs)

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
