import torch
from captum.attr import LayerIntegratedGradients
from transformers import BertForSequenceClassification, RobertaForSequenceClassification, XLNetForSequenceClassification
from typing import Dict

from thermometer.explain import ExplainerAutoModelInitializer


class ExplainerLayerIntegratedGradients(ExplainerAutoModelInitializer):
    def __init__(self):
        super().__init__()
        self.name_layer: str = None
        self.layer = None
        self.n_samples: int = None
        self.internal_batch_size = None

    def validate_config(self, config: Dict) -> bool:
        super().validate_config(config)
        assert 'n_samples' in config['explainer'], \
            'Define how many samples to take along the straight line path from the baseline.'
        assert 'internal_batch_size' in config['explainer'], 'Define an internal batch size for the attribute method.'

    @staticmethod
    def get_embedding_layer_name(model):
        if isinstance(model, BertForSequenceClassification):
            return 'bert.embeddings'
        elif isinstance(model, XLNetForSequenceClassification):
            return 'transformer.word_embedding'
        elif isinstance(model, RobertaForSequenceClassification):
            return 'roberta.embeddings'
        else:
            raise NotImplementedError

    @classmethod
    def from_config(cls, config):
        res = super().from_config(config)
        res.validate_config(config)
        res.n_samples = config['explainer']['n_samples']
        res.internal_batch_size = config['explainer']['internal_batch_size']
        res.name_layer = res.get_embedding_layer_name(res.model)
        for name, layer in res.model.named_modules():
            if name == res.name_layer:
                res.layer = layer
                break
        assert res.layer is not None, f'Layer {res.name_layer} not found.'
        res.explainer = LayerIntegratedGradients(forward_func=res.forward_func, layer=res.layer)
        return res

    def explain(self, batch):
        self.model.eval()
        self.model.zero_grad()
        batch = {k: v.to(self.device) for k, v in batch.items()}
        inputs, additional_forward_args = self.get_inputs_and_additional_args(name_model=self.name_model, batch=batch)
        predictions = self.forward_func(inputs, *additional_forward_args)
        target = torch.argmax(predictions, dim=1)
        base_line = self.get_baseline(batch=batch)
        attributions = self.explainer.attribute(inputs=inputs,
                                                n_steps=self.n_samples,
                                                additional_forward_args=additional_forward_args,
                                                target=target,
                                                baselines=base_line,
                                                internal_batch_size=self.internal_batch_size)
        attributions = torch.sum(attributions, dim=2)
        if isinstance(self.model, XLNetForSequenceClassification):
            # for xlnet, attributions.shape = [seq_len, batch_dim]
            # but [batch_dim, seq_len] is assumed
            attributions = attributions.T
        return attributions, predictions  # xlnet: [130, 1]
