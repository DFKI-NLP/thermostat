import torch
from captum.attr import LimeBase
from captum._utils.models.linear_model import SkLearnLinearModel
from typing import Any, Dict

from thermostat.explain import ExplainerAutoModelInitializer


class ExplainerLimeBase(ExplainerAutoModelInitializer):
    def __init__(self):
        super().__init__()
        self.internal_batch_size = None
        self.n_samples = None

    def validate_config(self, config: Dict) -> bool:
        super().validate_config(config)
        assert 'internal_batch_size' in config['explainer'], 'Define an internal batch size for the attribute method.'
        assert 'n_samples' in config['explainer'], 'Set number of samples for attribution function'

    @classmethod
    def from_config(cls, config):
        res = super().from_config(config)
        res.validate_config(config)
        res.internal_batch_size = config['explainer']['internal_batch_size']

        def similarity_kernel(
                original_input: torch.Tensor,
                perturbed_input: torch.Tensor,
                perturbed_interpretable_input: torch.Tensor,
                **kwargs) -> torch.Tensor:
            # kernel_width will be provided to attribute as a kwarg
            kernel_width = kwargs["kernel_width"]
            l2_dist = torch.norm(original_input - perturbed_input)
            return torch.exp(- (l2_dist**2) / (kernel_width**2))
            # Alternatively:
            # return torch.sum(original_input == perturbed_input)

        # Sampling function
        def perturb_func(
                original_input: torch.Tensor,
                **kwargs: Any) -> torch.Tensor:
            return original_input + torch.randn_like(original_input)

        def to_interp_rep_transform_custom(curr_sample, original_input, **kwargs: Any):
            return curr_sample

        res.explainer = LimeBase(forward_func=res.forward_func,
                                 interpretable_model=SkLearnLinearModel("linear_model.Ridge"),
                                 similarity_func=similarity_kernel,
                                 perturb_func=perturb_func,
                                 perturb_interpretable_space=False,
                                 from_interp_rep_transform=None,
                                 to_interp_rep_transform=to_interp_rep_transform_custom)
        return res

    def explain(self, batch):
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
