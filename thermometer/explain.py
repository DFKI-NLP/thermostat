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
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import Dict, Callable

from data.readers import JsonIterator, ShapleyJsonToTensor
from data.types import DatapointColored, DatapointProcessed
from explainers import *
from train import get_reader, load_checkpoint
from utils import get_logger, get_time, read_path, Configurable


class Explainer(Configurable):

    def validate_config(self, config: Dict) -> bool:
        raise NotImplementedError

    def from_config(cls, config: Dict):
        raise NotImplementedError

    def explain(self, batch):
        raise NotImplementedError

    def to(self, device):
        raise NotImplementedError


class ExplainerCaptum(Explainer):
    available_models = ['bert-base-cased', 'xlnet-base-cased']

    def __init__(self):
        super().__init__()

    @staticmethod
    def get_inputs_and_additional_args(name_model: str, batch):
        assert name_model in ExplainerCaptum.available_models, f'Unkown model:  {name_model}'
        if name_model == 'bert-base-cased' or name_model == 'xlnet-base-cased':
            assert 'input_ids' in batch, f'Input ids expected for {name_model} but not found.'
            assert 'attention_mask' in batch, f'Attention mask expected for {name_model} but not found.'
            assert 'token_type_ids' in batch, f'Token type ids expected for model {name_model} but not found.'
            input_ids = batch['input_ids']
            additional_forward_args = (batch['attention_mask'], batch['token_type_ids'])
            return input_ids, additional_forward_args
        else:
            raise NotImplementedError

    @staticmethod
    def get_forward_func(name_model: str, model):
        assert name_model in ExplainerCaptum.available_models, f'Unkown model:  {name_model}'

        def bert_forward(input_ids, attention_mask, token_type_ids):
            input_model = {
                'input_ids': input_ids.long(),
                'attention_mask': attention_mask.long(),
                'token_type_ids': token_type_ids.long(),
            }
            output_model = model(**input_model)[0]
            return output_model

        if name_model == 'bert-base-cased' or name_model == 'xlnet-base-cased':
            return bert_forward
        else:  # when adding a model, also update ExplainerCaptum.available_models
            raise NotImplementedError(f'Unknown model {name_model}')

    def validate_config(self, config: Dict) -> bool:
        raise NotImplementedError

    def from_config(cls, config: Dict):
        raise NotImplementedError

    def explain(self, input):
        raise NotImplementedError


class ExplainerAutoModelInitializer(ExplainerCaptum):  # todo check if this is a mixin rather

    def __init__(self):
        super().__init__()
        self.name_model: str = None
        self.model: AutoModelForSequenceClassification = None
        self.path_model: str = None
        self.forward_func: Callable = None
        self.pad_token_id = None #
        self.explainer = None
        self.device = None

    def validate_config(self, config: Dict) -> bool:
        assert 'name_model' in config, f'Provide the name of the model to explain. Available models: ' \
                                       f'{ExplainerCaptum.available_models}'
        assert 'path_model' in config, f'Provide a path to the model which should be explained.'
        # needed to deal w/ legacy code:
        assert 'mode_load' in config, f'Should the model be loaded using the ignite framework or huggingface?'
        assert 'num_labels' in config, f'Provide the number of labels.'
        return True

    @classmethod
    def from_config(cls, config):
        res = cls()
        res.validate_config(config)

        # model
        res.name_model = config['name_model']
        res.path_model = read_path(config['path_model'])
        res.mode_load = config['mode_load']
        if res.mode_load == 'huggingface':
            res.model = AutoModelForSequenceClassification.from_pretrained(res.path_model,
                                                                           num_labels=config['num_labels'])
        elif res.mode_load == 'ignite':
            # todo: num_labels hard coded for xlnet
            res.model = AutoModelForSequenceClassification.from_pretrained(res.name_model,
                                                                           num_labels=config['num_labels'])
            load_checkpoint(res.path_model, res.model)
        else:
            raise NotImplementedError
        res.forward_func = res.get_forward_func(name_model=res.name_model, model=res.model)
        res.pad_token_id = AutoTokenizer.from_pretrained(res.name_model).pad_token_id
        return res

    def to(self, device):
        self.device = device
        self.model.to(self.device)

    def get_baseline(self, batch):
        if self.pad_token_id == 0:
            # all non-special token ids are replaced by 0, the pad id
            baseline = batch['input_ids'] * batch['special_tokens_mask']
            return baseline
        else:
            baseline = batch['input_ids'] * batch['special_tokens_mask']  # all input ids now 0
            # add pad_id everywhere,
            # substract again where special tokens are, leaves non special tokens with pad id
            # and conserves original pad ids
            baseline = (baseline + self.pad_token_id) - (batch['special_tokens_mask'] * self.pad_token_id)
            return baseline

    def explain(self, input):
        raise NotImplementedError


def get_explainer(name: str, config: Dict):
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

    name_explainer = config['explainer']['name']  # what explainer to use
    config_explainer = config['explainer']['config']  # the explainer config
    name_explanation = config['name_explanation']  # under what name to save the explanation

    path_out = read_path(config['path_out'])
    name_dataset = config['dataset']['name']
    config_dataset = config['dataset']['config']
    path_in = read_path(config_dataset['path_in'])

    early_stopping = config['early_stopping']
    batch_size = config['batch_size']
    append = config['append']

    if logger is None:
        path_log_out = path_out + '.log'
        logger = get_logger(name='explain', level=logging.INFO, file_out=path_log_out)
        logger.info(f'(Config) Log file: {path_log_out}')
    logger.info(f'(Config) Output file: {path_out}')

    logger.info(f'(Config) Config: \n{json.dumps(config, indent=2)}')

    torch.cuda.empty_cache()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f'(Config) Explaining on device: {device}')

    explainer = get_explainer(name=name_explainer, config=config_explainer)
    explainer.to(device)

    logger.info(f'(Progress) Loaded trainer')

    dataset = get_reader(name=name_dataset, config=config_dataset)

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
        if predictions is not None: # generative explaines do not return predictions of the downstream model
            predictions = predictions.detach().cpu().numpy().tolist()
        for i, line in enumerate(batch_lines):
            json_line = json.loads(line)
            if isinstance(dataset, ShapleyJsonToTensor): # legacy code starts indexing at zero
                assert (json_id := json_line['id']) == (batch_id := (batch_tensors['id'][i].item() + 1)), \
                    f'Sanity check failed: Ids should match, but {json_id} != {batch_id}'
            else:
                assert json_line['id'] == batch_tensors['id'][i].item(), 'Sanity check failed: Ids should match'
            explanation = {'attribution': attribution[i],
                           'prediction': predictions[i] if predictions is not None else None,
                           'config': config}
            if not append:
                p_datapoint = DatapointProcessed.from_dict(json_line)
                c_datapoint = DatapointColored.from_parent_class(datapoint_processed=p_datapoint,
                                                                 name_explanation=name_explanation,
                                                                 explanation=explanation)
            else:
                c_datapoint = DatapointColored.from_dict(json_line)
                c_datapoint.append_explanation(name_explanation=name_explanation,
                                               explanation=explanation)
            file_out.write(str(c_datapoint) + os.linesep)
            logger.info(f'(Progress) Gave {(idx + 1) * batch_size} explanations')

    logger.info('(Progress) Terminated normally.')
    logger.info(f'(Progress) Done training {str(explainer)} explainer; wrote model weights to {path_out}')
