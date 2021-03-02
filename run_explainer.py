import json
import logging
import numpy as np
import os
import torch

from torch.utils.data import DataLoader
from tqdm import tqdm

import thermometer.explainers as thermex
from thermometer.data.readers import get_dataset
from thermometer.utils import detach_to_list, get_logger, get_time, read_config, read_path


config = read_config('configs/exp-a01_imdb_LayerIntegratedGradients_textattack-roberta-base-imdb.jsonnet')

logger = get_logger(name='explain', file_out='./pipeline.log', level=logging.INFO)

torch.manual_seed(123)
np.random.seed(123)

_now = get_time()

# File I/O
experiment_path = read_path(config['path'])
explainer_name = config['explainer']['name']
experiment_in = [f for f in os.listdir(experiment_path)
                 if "preprocess" in f and explainer_name not in f and f.endswith('.jsonl')][0]
path_in = os.path.join(experiment_path, experiment_in)
logger.info(f'(File I/O) Input file: {path_in}')
path_out = f'{read_path(experiment_path)}/{_now}.{explainer_name}.{experiment_in}'
logger.info(f'(File I/O) Output file: {path_out}')
assert not os.path.isfile(path_out), f'File {path_out} already exists!'

# Log config
logger.info(f'(Config) Config: \n{json.dumps(config, indent=2)}')

# Device
torch.cuda.empty_cache()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f'(Config) Explaining on device: {device}')

# Dataset
dataset_config = config['dataset']
dataset_config['tokenizer'] = config['model']['tokenizer']
dataset = get_dataset(config=dataset_config)
config['dataset']['num_labels'] = len(dataset.features['label'].names)

# Explainer
explainer = getattr(thermex, f'Explainer{explainer_name}').from_config(config=config)
explainer.to(device)
batch_size = config['explainer']['internal_batch_size']
logger.info(f'(Progress) Loaded explainer')

# DataLoader
dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
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
                  'attributions': attrbs,
                  'predictions': preds,
                  'path_model': config['model']['path_model']}

        file_out.write(json.dumps(result) + os.linesep)

logger.info('(Progress) Terminated normally.')
logger.info(f'(Progress) Done training {str(explainer)} explainer; wrote model weights to {path_out}')
