import json
import logging
import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import thermostat.explainers as thermex
from thermostat.data.readers import get_dataset, get_tokenizer
from thermostat.utils import detach_to_list, get_logger, get_time, read_config, read_path


logger = get_logger(name='explain', file_out='./pipeline.log', level=logging.INFO)

# Config handling
config_file = 'configs/imdb/bert/LayerIntegratedGradients.jsonnet'
config = read_config(config_file)
logger.info(f'(Config) Config: \n{json.dumps(config, indent=2)}')  # Log config

# Output file naming
explainer_name = config['explainer']['name']
path_out = f'{read_path(config["experiment_path"])}/{get_time()}.{explainer_name}.jsonl'  # TODO: Decide on CSV vs JSON
logger.info(f'(File I/O) Output file: {path_out}')
assert not os.path.isfile(path_out), f'File {path_out} already exists!'

# Random seeds
torch.manual_seed(123)
np.random.seed(123)

# Device
torch.cuda.empty_cache()
if 'device' in config.keys() and torch.cuda.is_available():
    device = config['device']
else:
    device = 'cpu'
logger.info(f'(Config) Explaining on device: {device}')

# Tokenizer
tokenizer = get_tokenizer(config['model'])
config['tokenizer'] = tokenizer

# Dataset
dataset = get_dataset(config=config)
config['dataset']['label_names'] = dataset.features['label'].names
config['dataset']['version'] = str(dataset.version)

# Explainer
explainer = getattr(thermex, f'Explainer{config["explainer"]["name"]}').from_config(config=config)
explainer.to(device)
batch_size = config['explainer']['internal_batch_size'] if 'internal_batch_size' in config['explainer'] else 1
logger.info(f'(Progress) Loaded explainer')

# DataLoader
dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
logger.info(f'(Progress) Initialized data loader')

config['model']['tokenizer'] = str(tokenizer)  # Overwrite the actual tokenizer with a string of the config

file_out = open(path_out, 'w+')

for idx_batch, batch in tqdm(enumerate(dataloader), total=len(dataloader), position=0, leave=True):
    if idx_batch % 1000 == 0:
        logger.info(f'(Progress) Processing batch {idx_batch} / instance {idx_batch * batch_size}')
    attribution, predictions = explainer.explain(batch)

    for idx_instance in range(len(batch['input_ids'])):
        idx_instance_running = (idx_batch * batch_size)

        ids = detach_to_list(batch['input_ids'][idx_instance])
        label = detach_to_list(batch['labels'][idx_instance])
        attrbs = detach_to_list(attribution[idx_instance])
        preds = detach_to_list(predictions[idx_instance])
        result = {'dataset': config['dataset'],
                  'model': config['model'],
                  'explainer': config['explainer'],
                  'batch': idx_batch,
                  'instance': idx_instance,
                  'index_running': idx_instance_running,
                  'input_ids': ids,
                  'label': label,
                  'attributions': attrbs,
                  'predictions': preds}
        # TODO: Add GPU runtime

        file_out.write(json.dumps(result) + os.linesep)

logger.info('(Progress) Terminated normally.')
logger.info(f'(Progress) Done training {str(explainer)} explainer; wrote model weights to {path_out}')
