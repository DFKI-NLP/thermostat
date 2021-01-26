"""
explain.py from GXAI project. Copied over on 2021-01-13: Commit:
https://github.com/rbtsbg/gxai/commit/5238d45a293abc44be627d4e608252521709a834
Author: rbtsbg
"""

import json
import logging
import numpy as np
import os
import torch

from torch.utils.data import DataLoader
from typing import Dict

from thermometer.data.dtypes import DatapointColored, DatapointProcessed
from thermometer.data.readers import DatasetProcessedDatapoints, JsonIterator, ShapleyJsonToTensor
from thermometer.explainers import ExplainerLayerIntegratedGradients, ExplainerShapleyValueSampling
from thermometer.utils import get_logger, get_time, read_path


def get_explainer(config: Dict):
    name = config["name"]

    if name == 'LayerIntegratedGradients':
        res = ExplainerLayerIntegratedGradients.from_config(config=config)
        return res
    if name == 'ShapleyValueSampling':
        res = ExplainerShapleyValueSampling.from_config(config=config)
        return res
    else:
        raise NotImplementedError


def run_explain(config: Dict, logger=None):
    torch.manual_seed(123)
    np.random.seed(123)

    _now = get_time()

    path_out = read_path(config['path_out'])
    config_dataset = config['dataset']['config']
    path_in = read_path(config_dataset['path_in'])

    early_stopping = config['early_stopping']
    batch_size = config['batch_size']

    if logger is None:
        path_log_out = path_out + '.log'
        logger = get_logger(name='explain', level=logging.INFO, file_out=path_log_out)
        logger.info(f'(Config) Log file: {path_log_out}')
    logger.info(f'(Config) Output file: {path_out}')

    logger.info(f'(Config) Config: \n{json.dumps(config, indent=2)}')

    torch.cuda.empty_cache()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f'(Config) Explaining on device: {device}')

    explainer = get_explainer(config=config["explainer"])
    explainer.to(device)

    logger.info(f'(Progress) Loaded trainer')

    dataset = DatasetProcessedDatapoints.from_config(config=config_dataset)

    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

    json_iterator = JsonIterator()

    logger.info(f'(Progress) Initialized data loader')

    file_out = open(path_out, 'w+')
    for idx, (batch_tensors, batch_lines) in enumerate(zip(dataloader,
                                                           json_iterator(path_in=path_in, batch_size=batch_size))):
        if 0 < early_stopping < idx:
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
                                                             name_explanation=name_explanation,
                                                             explanation=explanation)

            file_out.write(str(c_datapoint) + os.linesep)
            logger.info(f'(Progress) Gave {(idx + 1) * batch_size} explanations')

    logger.info('(Progress) Terminated normally.')
    logger.info(f'(Progress) Done training {str(explainer)} explainer; wrote model weights to {path_out}')
