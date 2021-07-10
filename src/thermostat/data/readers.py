from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
from typing import Dict, List


def download_dataset(config: Dict, logger):
    """
    :param config: "dataset" sub-config of a jsonnet config
    """
    assert all(x in config for x in ['name', 'split', 'root_dir'])
    if 'subset' in config:
        dataset = load_dataset(config['name'], config['subset'], split=config['split'])
        dataset.save_to_disk(dataset_path=f'{config["root_dir"]}/{config["name"]}_{config["subset"]}')
    else:
        dataset = load_dataset(config['name'], split=config['split'])
        dataset.save_to_disk(dataset_path=f'{config["root_dir"]}/{config["name"]}')
    logger.info(f'(Progress) Terminated normally')


def get_dataset(config: Dict):
    """ Returns a pytorch dataset from a config. """
    assert 'tokenizer' in config
    tokenizer = config['tokenizer']
    dataset_config = config['dataset']
    tokenization_config = config['model']['tokenization']

    def encode(instances):
        return tokenizer(instances[text_field],
                         truncation=tokenization_config['truncation'],
                         padding=tokenization_config['padding'],
                         max_length=tokenization_config['max_length'],
                         return_special_tokens_mask=tokenization_config['special_tokens_mask'])

    def encode_pair(instances):
        return tokenizer(instances[text_field[0]],
                         instances[text_field[1]],
                         truncation=tokenization_config['truncation'],
                         padding=tokenization_config['padding'],
                         max_length=tokenization_config['max_length'],
                         return_special_tokens_mask=tokenization_config['special_tokens_mask'])

    # Handle datasets with subsets correctly, e.g. "glue" has "sst2" and "qqp" as their subsets
    dataset_dir = f'{dataset_config["name"]}_' \
                  f'{dataset_config["subset"]}' if 'subset' in dataset_config else dataset_config['name']
    try:
        dataset = load_from_disk(f'{dataset_config["root_dir"]}/{dataset_dir}')
    except FileNotFoundError:
        raise FileNotFoundError(f'Execute download_data.py to first store the missing dataset ({dataset_dir}) '
                                f'locally.')

    dataset = dataset.select(indices=get_dataset_index_range(dataset, dataset_config))

    if 'text_field' in dataset_config:
        text_field = dataset_config['text_field']
    else:
        text_field = 'text'
    encode_fn = encode_pair if type(text_field) == list and len(text_field) else encode
    dataset = dataset.map(encode_fn, batched=True, batch_size=dataset_config['batch_size'])

    if 'label_field' in dataset_config:
        label_field = dataset_config['label_field']
        expression = label_field['expression']
    else:
        label_field = 'label'

    def get_label(example: Dict):
        if type(label_field) == str:
            return example[label_field]
        else:
            try:
                return eval('example' + expression)
            except IndexError:
                return None

    dataset = dataset.map(lambda examples: {'labels': get_label(examples)},
                          batch_size=dataset_config['batch_size'])  # batched=True,
    dataset.set_format(type='torch', columns=dataset_config['columns'])
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

    dataset = load_dataset('json', data_files=config['path_explanations'])
    dataset = dataset['train']

    def encode_local(instances):
        res = {k: instances[k] for k in config['columns']}
        return res

    dataset = dataset.map(encode_local, batched=True)
    dataset.set_format(type='torch', columns=config['columns'])
    return dataset


def get_tokenizer(config: Dict):
    """
    :param config: model sub-config
    :return:
    """
    tokenizer = AutoTokenizer.from_pretrained(config['name'])
    return tokenizer
