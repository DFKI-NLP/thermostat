import torch
from captum.attr import LimeBase
from captum._utils.models.linear_model import SkLearnLinearModel
from functools import partial
from typing import Any, Dict

from thermostat.explain import ExplainerAutoModelInitializer
from thermostat.utils import detach_to_list


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

        def token_similarity_kernel(
                original_input: torch.Tensor,
                perturbed_input: torch.Tensor,
                perturbed_interpretable_input: torch.Tensor,
                **kwargs) -> torch.Tensor:
            """
            Following https://github.com/copenlu/ALPS_2021
            """
            return torch.sum(original_input == perturbed_input) / len(original_input)

        # Sampling function
        def perturb_func(
                original_input: torch.Tensor,
                **kwargs: Any) -> torch.Tensor:
            """
            Following https://github.com/copenlu/ALPS_2021
            """
            # Build mask for replacing random tokens with [PAD] token
            mask_value_probs = torch.tensor([1, 7], dtype=torch.float)  # 1/8 chance of masking with 0
            mask_multinomial_binary = torch.multinomial(mask_value_probs,
                                                        len(original_input[0]),
                                                        replacement=True)

            # Additionally remove special_token_ids
            mask_special_token_ids = torch.Tensor([1 if id_ in res.special_token_ids else 0
                                                   for id_ in detach_to_list(original_input[0])]).int()

            # Merge the binary mask (12.5% masks) with the special_token_ids mask
            mask = torch.tensor([m + s if s == 0 else s for m, s in zip(
                detach_to_list(mask_multinomial_binary), detach_to_list(mask_special_token_ids))]).to(res.device)

            # Apply mask to original input
            perturbed_input = original_input * mask + (1 - mask) * res.pad_token_id
            return perturbed_input

        def to_interp_rep_transform_custom(curr_sample, original_input, **kwargs: Any):
            return curr_sample

        res.interpretable_model = SkLearnLinearModel("linear_model.Ridge")

        res.explainer = LimeBase(forward_func=res.forward_func,
                                 interpretable_model=res.interpretable_model,  # TODO: Other model?
                                 similarity_func=token_similarity_kernel,
                                 perturb_func=perturb_func,
                                 perturb_interpretable_space=False,  # TODO: Understand
                                 from_interp_rep_transform=None,  # TODO: Understand
                                 to_interp_rep_transform=to_interp_rep_transform_custom)  # TODO: Understand
        return res

    def explain(self, batch):
        batch = {k: v.to(self.device) for k, v in batch.items()}
        inputs, additional_forward_args = self.get_inputs_and_additional_args(name_model=self.name_model, batch=batch)

        predictions = self.forward_func(inputs, *additional_forward_args)
        target = torch.argmax(predictions, dim=1)
        attributions = self.explainer.attribute(inputs=inputs,
                                                n_samples=self.n_samples,
                                                additional_forward_args=additional_forward_args,
                                                target=target)
        return attributions, predictions
