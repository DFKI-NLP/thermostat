import json
import logging
import numpy as np
import os
import torch

from torch.utils.data import DataLoader
from tqdm import tqdm

import thermostat.explainers as thermex
from thermostat.data.readers import get_dataset
from thermostat.utils import detach_to_list, get_logger, get_time, read_config, read_path


# TODO: argparser or other input handler instead of hard-coded config
config = read_config('configs/imdb_LimeBase_bert-base-cased-ignite-imdb.jsonnet')

logger = get_logger(name='explain', file_out='./pipeline.log', level=logging.INFO)

torch.manual_seed(123)
np.random.seed(123)

_now = get_time()

# File I/O
experiment_path = read_path(config['path'])
if not os.path.exists(experiment_path):
    raise NotADirectoryError(f'{experiment_path}\nThis experiment path does not exist yet.')

explainer_name = config['explainer']['name']
path_out = f'{read_path(experiment_path)}/{_now}.{explainer_name}'
logger.info(f'(File I/O) Output file: {path_out}')
assert not os.path.isfile(path_out), f'File {path_out} already exists!'

# Log config
logger.info(f'(Config) Config: \n{json.dumps(config, indent=2)}')

# Device
torch.cuda.empty_cache()
if 'device' in config.keys() and torch.cuda.is_available():
    device = config['device']
else:
    device = 'cpu'
logger.info(f'(Config) Explaining on device: {device}')

# Dataset
dataset_config = config['dataset']
dataset_config['tokenizer'] = config['model']['tokenizer']
dataset = get_dataset(config=dataset_config)
#dataset = dataset.filter(lambda instance: detach_to_list(instance['input_ids'][0]).index(1) < 250)
config['dataset']['num_labels'] = len(dataset.features['label'].names)

# Explainer
explainer = getattr(thermex, f'Explainer{explainer_name}').from_config(config=config)
explainer.to(device)
batch_size = config['explainer']['internal_batch_size']
logger.info(f'(Progress) Loaded explainer')

# DataLoader
dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
logger.info(f'(Progress) Initialized data loader')

file_out = open(path_out, 'w+')

for idx_batch, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
    logger.info(f'(Progress) Processing batch {idx_batch} / instance {idx_batch * batch_size}')
    attribution, predictions = explainer.explain(batch)

    for idx_instance in range(len(batch['input_ids'])):
        idx_instance_running = (idx_batch * batch_size)

        ids = detach_to_list(batch['input_ids'][idx_instance])
        labels = detach_to_list(batch['labels'][idx_instance])
        attrbs = detach_to_list(attribution[idx_instance])
        preds = detach_to_list(predictions[idx_instance])
        result = {'dataset': dataset_config,
                  'batch': idx_batch,
                  'instance': idx_instance,
                  'index_running': idx_instance_running,
                  'explainer': config['explainer'],
                  'input_ids': ids,
                  'labels': labels,
                  'label_names': dataset.features['label'].names,
                  'attributions': attrbs,
                  'predictions': preds,
                  'path_model': config['model']['path_model']}

        file_out.write(json.dumps(result) + os.linesep)

logger.info('(Progress) Terminated normally.')
logger.info(f'(Progress) Done training {str(explainer)} explainer; wrote model weights to {path_out}')
