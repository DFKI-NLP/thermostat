import torch
from captum.attr import Occlusion
from typing import Dict

from thermostat.explain import ExplainerAutoModelInitializer


class ExplainerOcclusion(ExplainerAutoModelInitializer):
    def __init__(self):
        super().__init__()
        self.internal_batch_size = None
        self.sliding_window_shapes = None

    def validate_config(self, config: Dict) -> bool:
        super().validate_config(config)
        assert 'internal_batch_size' in config['explainer'], 'Define an internal batch size for the attribute method.'
        assert 'sliding_window_shapes' in config['explainer'], 'Set shape of patch to occlude each input.'

    @classmethod
    def from_config(cls, config):
        res = super().from_config(config)
        res.validate_config(config)
        res.internal_batch_size = config['explainer']['internal_batch_size']
        res.sliding_window_shapes = tuple(config['explainer']['sliding_window_shapes'])
        res.explainer = Occlusion(forward_func=res.forward_func)
        return res

    def explain(self, batch):
        batch = {k: v.to(self.device) for k, v in batch.items()}
        inputs, additional_forward_args = self.get_inputs_and_additional_args(base_model=type(self.model.base_model),
                                                                              batch=batch)

        predictions = self.forward_func(inputs, *additional_forward_args)
        target = torch.argmax(predictions, dim=1)
        attributions = self.explainer.attribute(inputs=inputs,
                                                sliding_window_shapes=self.sliding_window_shapes,
                                                additional_forward_args=additional_forward_args,
                                                target=target)
        return attributions, predictions
