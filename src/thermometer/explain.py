from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import Dict, Callable


from thermometer.train import load_checkpoint
from thermometer.utils import read_config, read_path, Configurable


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
    available_models = ['bert-base-cased', 'xlnet-base-cased', 'textattack/roberta-base-imdb']

    def __init__(self):
        super().__init__()

    @staticmethod
    def get_inputs_and_additional_args(name_model: str, batch):
        assert name_model in ExplainerCaptum.available_models, f'Unknown model: {name_model}'
        if name_model in ['bert-base-cased', 'xlnet-base-cased']:
            assert 'input_ids' in batch, f'Input ids expected for {name_model} but not found.'
            assert 'attention_mask' in batch, f'Attention mask expected for {name_model} but not found.'
            assert 'token_type_ids' in batch, f'Token type ids expected for model {name_model} but not found.'
            input_ids = batch['input_ids']
            additional_forward_args = (batch['attention_mask'], batch['token_type_ids'])
            return input_ids, additional_forward_args
        elif name_model == 'textattack/roberta-base-imdb':  # TODO: Separate classes?
            assert 'input_ids' in batch, f'Input ids expected for {name_model} but not found.'
            assert 'attention_mask' in batch, f'Attention mask expected for {name_model} but not found.'
            input_ids = batch['input_ids']
            additional_forward_args = (batch['attention_mask'],)
            return input_ids, additional_forward_args
        else:
            raise NotImplementedError

    @staticmethod
    def get_forward_func(name_model: str, model):
        assert name_model in ExplainerCaptum.available_models, f'Unknown model: {name_model}'

        def bert_forward(input_ids, attention_mask, token_type_ids):
            input_model = {
                'input_ids': input_ids.long(),
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

        if name_model in ['bert-base-cased', 'xlnet-base-cased']:
            return bert_forward
        elif name_model == 'textattack/roberta-base-imdb':  # TODO: Separate classes?
            return roberta_forward
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
        self.pad_token_id = None
        self.explainer = None
        self.device = None

    def validate_config(self, config: Dict) -> bool:
        assert 'name' in config['model'], f'Provide the name of the model to explain. Available models: ' \
                                          f'{ExplainerCaptum.available_models}'
        assert 'path_model' in config['model'], f'Provide a path to the model which should be explained.'
        assert 'mode_load' in config['model'], f'Should the model be loaded using the ignite framework or huggingface?'
        return True

    @classmethod
    def from_config(cls, config):
        res = cls()
        res.validate_config(config)

        # model
        res.name_model = config['model']['name']
        if config['model']['path_model']:  # can be empty when loading a HF model!
            res.path_model = read_path(config['model']['path_model'])

        res.mode_load = config['model']['mode_load']
        assert res.mode_load in ['hf', 'ignite']

        res.num_labels = config['dataset']['num_labels']
        res.model = AutoModelForSequenceClassification.from_pretrained(res.name_model,
                                                                       num_labels=res.num_labels)
        if res.mode_load == 'ignite':
            load_checkpoint(res.path_model, res.model)

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