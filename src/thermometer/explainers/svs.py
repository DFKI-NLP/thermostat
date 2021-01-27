import torch
from captum.attr import ShapleyValueSampling
from typing import Dict

from thermometer.explain import ExplainerAutoModelInitializer


class ExplainerShapleyValueSampling(ExplainerAutoModelInitializer):

    def __init__(self):
        super().__init__()
        self.n_samples = None

    def validate_config(self, config: Dict) -> bool:
        super().validate_config(config)
        assert 'n_samples' in config, 'Define how many samples to take along the straight line path from the baseline.'

    @classmethod
    def from_config(cls, config):
        res = super().from_config(config)
        res.n_samples = config['n_samples']
        res.explainer = ShapleyValueSampling(res.forward_func)  # KernelShap(forward_func=res.forward_func)
        return res

    def explain(self, batch):
        # todo: set model.eval() ? -> in a test self.model.training was False
        batch = {k: v.to(self.device) for k, v in batch.items()}
        inputs, additional_forward_args = self.get_inputs_and_additional_args(name_model=self.name_model, batch=batch)
        predictions = self.forward_func(inputs, *additional_forward_args)
        target = torch.argmax(predictions, dim=1)
        base_line = self.get_baseline(batch=batch)
        attributions = self.explainer.attribute(inputs=inputs,
                                                n_samples=self.n_samples,
                                                additional_forward_args=additional_forward_args,
                                                target=target,
                                                baselines=base_line)
        return attributions, predictions
