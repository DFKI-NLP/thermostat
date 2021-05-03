import torch
from captum.attr import (
    GuidedBackprop, InputXGradient, LayerIntegratedGradients,
    configure_interpretable_embedding_layer, remove_interpretable_embedding_layer)
from transformers import XLNetForSequenceClassification
from typing import Dict

from thermostat.explain import ExplainerAutoModelInitializer


class ExplainerInputXGradient(ExplainerAutoModelInitializer):
    def __init__(self):
        super().__init__()

    def validate_config(self, config: Dict) -> bool:
        super().validate_config(config)

    @classmethod
    def from_config(cls, config):
        res = super().from_config(config)
        res.validate_config(config)
        res.explainer = InputXGradient(forward_func=res.forward_func)
        return res

    def explain(self, batch):
        self.model.eval()
        self.model.zero_grad()
        batch = {k: v.to(self.device) for k, v in batch.items()}
        inputs, additional_forward_args = self.get_inputs_and_additional_args(base_model=type(self.model.base_model),
                                                                              batch=batch)

        predictions = self.forward_func(inputs, *additional_forward_args)
        target = torch.argmax(predictions, dim=1)
        attributions = self.explainer.attribute(inputs=inputs,
                                                additional_forward_args=additional_forward_args,
                                                target=target)
        return attributions, predictions


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
        inputs, additional_forward_args = self.get_inputs_and_additional_args(base_model=type(self.model.base_model),
                                                                              batch=batch)
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


class ExplainerGuidedBackprop(ExplainerAutoModelInitializer):
    def __init__(self):
        super().__init__()

    def validate_config(self, config: Dict) -> bool:
        super().validate_config(config)

    @classmethod
    def from_config(cls, config):
        res = super().from_config(config)
        res.validate_config(config)
        res.explainer = GuidedBackprop(res.model)  # Input to GBP is not the forward function but the model
        res.explainer.forward_func = res.forward_func  # Forward func needs be overridden afterwards
        return res

    def explain(self, batch):
        batch = {k: v.to(self.device) for k, v in batch.items()}
        inputs, additional_forward_args = self.get_inputs_and_additional_args(base_model=type(self.model.base_model),
                                                                              batch=batch)

        predictions = self.forward_func(inputs, *additional_forward_args)
        target = torch.argmax(predictions, dim=1)

        class EmbedsWrapper(torch.nn.Module):
            def __init__(self, transformer):
                super().__init__()
                self.transformer = transformer

            def forward(self, input_ids, attention_mask, token_type_ids):
                # Following
                # https://github.com/copenlu/ALPS_2021/blob/2d1a500be8affaf874da688ebcbb544f66ecb5e4/tutorial_src/
                # model_builders.py#L162
                output_model = self.transformer(inputs_embeds=input_ids,
                                                attention_mask=attention_mask)['logits']
                return output_model

        # TODO: Something's not quite right here
        from transformers import AutoModelForSequenceClassification
        m = AutoModelForSequenceClassification.from_pretrained(self.name_model,
                                                               num_labels=self.num_labels).to(self.device)
        wrapped_model = EmbedsWrapper(m)
        new_explainer = GuidedBackprop(wrapped_model)
        inputs_embeds = m.base_model.embeddings(inputs)

        attributions = new_explainer.attribute(inputs=inputs_embeds,
                                               additional_forward_args=additional_forward_args,
                                               target=target)
        # TODO: Fix Attributions shape (1 x 512 x 768)
        return attributions, predictions
