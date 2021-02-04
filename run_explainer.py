import json
import logging
import numpy as np
import os
import torch

from torch.utils.data import DataLoader

import thermometer.explainers as thermex
from thermometer.data.dtypes import DatapointColored, DatapointProcessed
from thermometer.data.readers import DatasetProcessedDatapoints, JsonIterator, ShapleyJsonToTensor
from thermometer.utils import get_logger, get_time, read_config, read_path


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

logger.info(f'(Config) Config: \n{json.dumps(config, indent=2)}')

torch.cuda.empty_cache()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f'(Config) Explaining on device: {device}')

path_out = f'{read_path(experiment_path)}/{_now}.{explainer_name}.{experiment_in}'
logger.info(f'(File I/O) Output file: {path_out}')

explainer = getattr(thermex, f'Explainer{explainer_name}').from_config(config=config)
explainer.to(device)

logger.info(f'(Progress) Loaded trainer')

dataset_config = read_config(config['dataset']['config'])
dataset_config['path_in'] = path_in
preprocess_config = read_config(config['model']['tokenizer'])
dataset_config['name_input'] = preprocess_config['name_tokenizer']
dataset = DatasetProcessedDatapoints.from_config(config=dataset_config)

batch_size = config['explainer']['internal_batch_size']
dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

json_iterator = JsonIterator()

logger.info(f'(Progress) Initialized data loader')

file_out = open(path_out, 'w+')
for idx, (batch_tensors, batch_lines) in enumerate(zip(dataloader,
                                                       json_iterator(path_in=path_in,
                                                                     batch_size=batch_size))):
    if 0 < config['explainer']['early_stopping'] < idx:
        logger.info(f'Stopping early after {idx} batches.')
        break
    attribution, predictions = explainer.explain(batch_tensors)
    attribution = attribution.detach().cpu().numpy().tolist()
    if predictions is not None:  # generative explaines do not return predictions of the downstream model
        predictions = predictions.detach().cpu().numpy().tolist()
    for i, line in enumerate(batch_lines):
        json_line = json.loads(line)
        if isinstance(dataset, ShapleyJsonToTensor):  # legacy code starts indexing at zero
            assert (json_id := json_line['id']) == (batch_id := (batch_tensors['id'][i].item() + 1)), \
                f'Sanity check failed: Ids should match, but {json_id} != {batch_id}'
        else:
            assert json_line['id'] == batch_tensors['id'][i].item(), 'Sanity check failed: Ids should match'
        explanation = {'attribution': attribution[i],
                       'prediction': predictions[i] if predictions is not None else None,
                       'config': config}

        p_datapoint = DatapointProcessed.from_dict(json_line)
        c_datapoint = DatapointColored.from_parent_class(datapoint_processed=p_datapoint,
                                                         name_explanation=explainer_name,
                                                         explanation=explanation)

        file_out.write(str(c_datapoint) + os.linesep)
        logger.info(f'(Progress) Gave {(idx + 1) * batch_size} explanations')

logger.info('(Progress) Terminated normally.')
logger.info(f'(Progress) Done training {str(explainer)} explainer; wrote model weights to {path_out}')
