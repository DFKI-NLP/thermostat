from datasets import load_dataset, load_from_disk, Features, Value
from transformers import AutoTokenizer
from typing import Dict

from thermostat.utils import read_path


def download_dataset(config: Dict, logger):
    """
    :param config: "dataset" sub-config of a jsonnet config
    """
    assert all(x in config for x in ['name', 'split', 'root_dir'])
    dataset = load_dataset(config['name'], split=config['split'])
    root_dir = read_path(config['root_dir'])
    dataset.save_to_disk(dataset_path=f'{root_dir}/{config["name"]}')
    logger.info(f'(Progress) Terminated normally')


def get_dataset(config: Dict):
    """ Returns a pytorch dataset from a config. """
    assert 'tokenizer' in config
    tokenizer_config = config['tokenizer']

    def encode(instances):
        return tokenizer(instances[text_field],
                         truncation=tokenizer_config['truncation'],
                         padding=tokenizer_config['padding'],
                         max_length=tokenizer_config['max_length'],
                         return_special_tokens_mask=tokenizer_config['special_tokens_mask'])

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_config['name'])
    dataset = load_from_disk(f'{read_path(config["root_dir"])}/{config["name"]}')

    dataset = dataset.select(indices=get_dataset_index_range(dataset, config))

    text_field = 'text' if 'text_field' not in config else config['text_field']
    dataset = dataset.map(encode, batched=True, batch_size=config['batch_size'])
    dataset = dataset.map(lambda examples: {'labels': examples['label']},
                          batched=True, batch_size=config['batch_size'])
    dataset.set_format(type='torch', columns=config['columns'])
    return dataset


def get_dataset_index_range(dataset, dataset_config):
    start = 0 if 'start' not in dataset_config else dataset_config['start']
    if start < 0:
        start = 0

    end = len(dataset) if 'end' not in dataset_config else dataset_config['end']
    if end < 0:
        end = len(dataset)

    return range(start, end)


def get_local_explanations(config: Dict):
    """
    :param config: visualization dict of config
    :return:
    """

    dataset = load_dataset('json', data_files=read_path(config['path_in']))
    dataset = dataset['train']  # todo: why is this necessary, why is a Dict returned?

    def encode_local(instances):
        res = {k: instances[k] for k in config['columns']}
        return res

    dataset = dataset.map(encode_local, batched=True)
    dataset.set_format(type='torch', columns=config['columns'])
    return dataset  # TODO: why is the train-indexing this necessary?
