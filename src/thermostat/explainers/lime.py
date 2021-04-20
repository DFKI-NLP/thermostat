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
        res.n_samples = config['explainer']['n_samples']
        res.kernel_width = config['explainer']['kernel_width']

        def default_similarity_kernel(
                original_input: torch.Tensor,
                perturbed_input: torch.Tensor,
                perturbed_interpretable_input: torch.Tensor,
                **kwargs) -> torch.Tensor:
            """
            Following # https://github.com/PAIR-code/lit/blob/main/lit_nlp/components/citrus/lime.py#L74
            :param original_input:
            :param perturbed_input:
            :param perturbed_interpretable_input:
            :param kwargs:
            :return:
            """
            l2_dist = torch.norm(original_input - perturbed_input)
            return torch.sqrt(torch.exp(- (l2_dist**2) / (res.kernel_width**2)))

        def token_similarity_kernel(
                original_input: torch.Tensor,
                perturbed_input: torch.Tensor,
                perturbed_interpretable_input: torch.Tensor,
                **kwargs) -> torch.Tensor:
            """
            Following https://github.com/copenlu/ALPS_2021
            :param original_input:
            :param perturbed_input:
            :param perturbed_interpretable_input:
            :param kwargs:
            :return:
            """
            return torch.sum(original_input == perturbed_input)

        # Sampling function
        def perturb_func(
                original_input: torch.Tensor,
                **kwargs: Any) -> torch.Tensor:
            """
            Following https://github.com/copenlu/ALPS_2021
            :param original_input:
            :param kwargs:
            :return:
            """
            # Build mask for replacing random tokens with [PAD] token
            mask = torch.randint(low=0, high=2, size=original_input.size()).to(res.device)
            return original_input * mask + (1 - mask) * res.pad_token_id

        def to_interp_rep_transform_custom(curr_sample, original_input, **kwargs: Any):
            return curr_sample

        res.explainer = LimeBase(forward_func=res.forward_func,
                                 interpretable_model=SkLearnLinearModel("linear_model.Ridge"),
                                 similarity_func=token_similarity_kernel,
                                 perturb_func=perturb_func,
                                 perturb_interpretable_space=False,
                                 from_interp_rep_transform=None,
                                 to_interp_rep_transform=to_interp_rep_transform_custom)
        return res

    def explain(self, batch):
        batch = {k: v.to(self.device) for k, v in batch.items()}
        inputs, additional_forward_args = self.get_inputs_and_additional_args(name_model=self.name_model, batch=batch)
        # TODO: Check if removing PAD tokens is necessary
        #additional_forward_args += (inputs[0] != tokenizer.pad_token_id)
        predictions = self.forward_func(inputs, *additional_forward_args)
        target = torch.argmax(predictions, dim=1)
        base_line = self.get_baseline(batch=batch)
        attributions = self.explainer.attribute(inputs=inputs,
                                                n_samples=self.n_samples,
                                                additional_forward_args=additional_forward_args,
                                                target=target,
                                                baselines=base_line)
        return attributions, predictions
