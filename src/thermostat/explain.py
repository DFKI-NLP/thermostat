import torch
import transformers.models as tlm
from ignite.handlers import ModelCheckpoint
from transformers import AutoModelForSequenceClassification
from typing import Dict, Callable

from thermostat.utils import read_path, Configurable


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
        assert 'path_model' in config['model'], f'Provide a path to the model which should be explained.'
        assert 'mode_load' in config['model'], f'Should the model be loaded using the ignite framework or huggingface?'
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

        res.mode_load = config['model']['mode_load']
        assert res.mode_load in ['hf', 'ignite']

        res.num_labels = len(config['dataset']['label_names'])
        # TODO: Assert that num_labels in dataset corresponds to classification head in model

        if res.mode_load == 'hf':
            res.model = AutoModelForSequenceClassification.from_pretrained(res.name_model,
                                                                           num_labels=res.num_labels)
        elif res.mode_load == 'ignite':
            res.model = AutoModelForSequenceClassification.from_pretrained(res.name_model,
                                                                           num_labels=res.num_labels)
            checkpoint = torch.load(res.path_model)
            to_load = {'model': res.model}
            ModelCheckpoint.load_objects(to_load=to_load,
                                         checkpoint=checkpoint)  # overwrite pretrained weights w/ fine-tuned weights
        else:
            raise NotImplementedError('"hf" and "ignite" values are supported for config field "mode_load".')

        res.forward_func = res.get_forward_func(name_model=res.name_model, model=res.model)
        res.pad_token_id = config['tokenizer'].pad_token_id
        res.special_token_ids = config['tokenizer'].all_special_ids
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

    def __str__(self):
        return str(self.explainer)
