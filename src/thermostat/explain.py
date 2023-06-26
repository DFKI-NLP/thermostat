import json
import logging
import numpy as np
import os
import torch
import transformers
import transformers.models as tlm
from ignite.handlers import ModelCheckpoint
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification
from typing import Dict, Callable

from thermostat.data.readers import get_dataset, get_tokenizer
from thermostat.utils import Configurable, detach_to_list, get_logger, get_time, read_config


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

    def __init__(self):
        super().__init__()

    @staticmethod
    def get_inputs_and_additional_args(base_model, batch):
        if base_model in [tlm.albert.AlbertModel,
                          tlm.bert.BertModel,
                          tlm.electra.ElectraModel,
                          tlm.xlnet.XLNetModel]:
            assert 'input_ids' in batch, f'Input ids expected for {base_model} but not found.'
            assert 'attention_mask' in batch, f'Attention mask expected for {base_model} but not found.'
            assert 'token_type_ids' in batch, f'Token type ids expected for model {base_model} but not found.'
            input_ids = batch['input_ids']
            additional_forward_args = (batch['attention_mask'], batch['token_type_ids'])
        elif base_model in [tlm.roberta.RobertaModel,
                            tlm.distilbert.DistilBertModel]:
            assert 'input_ids' in batch, f'Input ids expected for {base_model} but not found.'
            assert 'attention_mask' in batch, f'Attention mask expected for {base_model} but not found.'
            input_ids = batch['input_ids']
            additional_forward_args = (batch['attention_mask'],)
        else:
            raise NotImplementedError(f'Unknown model: {base_model}')
        return input_ids, additional_forward_args

    @staticmethod
    def get_forward_func(name_model: str, model):

        def bert_forward(input_ids, attention_mask, token_type_ids):
            input_model = {
                'input_ids': input_ids.long(),  # TODO: Is the cast to long necessary?
                'attention_mask': attention_mask.long(),
                'token_type_ids': token_type_ids.long(),
            }
            output_model = model(**input_model)[0]
            return output_model

        def roberta_forward(input_ids, attention_mask):
            input_model = {
                'input_ids': input_ids.long(),
                'attention_mask': attention_mask.long(),
            }
            output_model = model(**input_model)[0]
            return output_model

        if type(model.base_model) in [tlm.albert.AlbertModel,
                                      tlm.bert.BertModel,
                                      tlm.electra.ElectraModel,
                                      tlm.xlnet.XLNetModel]:
            return bert_forward
        elif type(model.base_model) in [tlm.roberta.RobertaModel,
                                        tlm.distilbert.DistilBertModel]:
            return roberta_forward
        else:
            raise NotImplementedError(f'Unknown model {name_model}')

    @staticmethod
    def get_embedding_layer(model):
        """ Used for LIG and LGXA in explainers/grad.py """
        if type(model.base_model) == tlm.xlnet.XLNetModel:
            return model.base_model.word_embedding
        else:
            return model.base_model.embeddings

    def validate_config(self, config: Dict) -> bool:
        raise NotImplementedError

    def from_config(cls, config: Dict):
        raise NotImplementedError

    def explain(self, input):
        raise NotImplementedError

    def to(self, device):
        raise NotImplementedError


class ExplainerAutoModelInitializer(ExplainerCaptum):  # todo check if this is a mixin rather

    def __init__(self):
        super().__init__()
        self.name_model: str = None
        self.model = None  # AutoModelForSequenceClassification per default, but changes into TLM-specific class
        self.path_model: str = None
        self.forward_func: Callable = None
        self.pad_token_id = None
        self.explainer = None
        self.device = None

    def validate_config(self, config: Dict) -> bool:
        assert 'name' in config['model'], f'Provide the name of the model to explain.'
        return True

    @classmethod
    def from_config(cls, config):
        res = cls()
        res.validate_config(config)

        # device
        res.device = config['device']

        # model
        res.name_model = config['model']['name']
        if config['model']['path_model']:  # can be empty when loading a HF model!
            res.path_model = config['model']['path_model']
        
        # new change
        # if not config['model']['class']:
        if 'class' not in config['model'].keys():
        # new change
            res.num_labels = len(config['dataset']['label_names'])
            # TODO: Assert that num_labels in dataset corresponds to classification head in model
            res.model = AutoModelForSequenceClassification.from_pretrained(res.name_model, num_labels=res.num_labels)
        else:
            res.model = getattr(transformers, config['model']['class']).from_pretrained(res.name_model)

        if 'mode_load' in config['model']:
            res.mode_load = config['model']['mode_load']
            if res.mode_load == 'ignite':
                checkpoint = torch.load(res.path_model)
                to_load = {'model': res.model}
                ModelCheckpoint.load_objects(to_load=to_load, # overwrite pretrained weights w/ fine-tuned weights
                                             checkpoint=checkpoint)

        res.forward_func = res.get_forward_func(name_model=res.name_model, model=res.model)
        res.pad_token_id = config['tokenizer'].pad_token_id
        res.special_token_ids = config['tokenizer'].all_special_ids
        return res

    def to(self, device):
        self.device = device
        self.model.to(self.device)

    def get_baseline(self, batch):
        assert 'special_tokens_mask' in batch
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

    def __str__(self):
        return str(self.explainer)


def explain_custom_data(
        config_file,
        home_dir=None
):
    log_file = config_file.replace('/', '_').strip('.jsonnet')
    logger = get_logger(name='explain', file_out=f'./{log_file}.log', level=logging.INFO)

    # Config handling
    config = read_config(config_file, home_dir=home_dir)
    logger.info(f'(Config) Config: \n{json.dumps(config, indent=2)}')  # Log config

    # Output file naming
    explainer_name = config['explainer']['name']
    path_out = f'{config["experiment_path"]}/{get_time()}.{explainer_name}.jsonl'
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
    logger.info('(Progress) Preparing data')
    dataset = get_dataset(config=config)
    config['dataset']['label_names'] = dataset.features['label'].names
    config['dataset']['version'] = str(dataset.version)

    # Explainer
    import thermostat.explainers as thermex
    logger.info('(Progress) Preparing model and explainer')
    explainer = getattr(thermex, f'Explainer{config["explainer"]["name"]}').from_config(config=config)
    explainer.to(device)
    batch_size = config['explainer']['internal_batch_size'] if 'internal_batch_size' in config['explainer'] else 1
    logger.info(f'(Progress) Loaded explainer')

    # DataLoader
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size)
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
            result = {
                'dataset': config['dataset'],
                'model': config['model'],
                'explainer': config['explainer'],
                'batch': idx_batch,
                'instance': idx_instance,
                'index_running': idx_instance_running,
                'input_ids': ids,
                'label': label,
                'attributions': attrbs,
                'predictions': preds,
            }

            file_out.write(json.dumps(result) + os.linesep)

    logger.info('(Progress) Terminated normally.')
    logger.info(f'(Progress) Output file: {path_out}')
