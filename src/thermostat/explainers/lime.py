import torch
from captum.attr import KernelShap, Lime, LimeBase
from captum._utils.models.linear_model import SkLearnLasso, SkLearnLinearModel
from functools import partial
from math import exp
from typing import Any, Dict, List

from thermostat.explain import ExplainerAutoModelInitializer
from thermostat.utils import detach_to_list


class ExplainerLime(ExplainerAutoModelInitializer):
    def __init__(self):
        super().__init__()
        self.n_samples = None

    @staticmethod
    def token_similarity_kernel(
            original_input: torch.Tensor,
            perturbed_input: torch.Tensor,
            perturbed_interpretable_input: torch.Tensor,
            **kwargs) -> torch.Tensor:
        """
        Following https://github.com/copenlu/ALPS_2021
        """
        pad_token_id = 0
        kernel_width = 1.1
        assert original_input.shape[0] == perturbed_input.shape[0] == 1, 'This kernel assumes batch size 1'
        original_input_list = detach_to_list(original_input[0])
        perturbed_input_list = detach_to_list(perturbed_input[0])
        number_pad_tokens_original = original_input_list.count(pad_token_id)
        number_pad_tokens_perturbed = perturbed_input_list.count(pad_token_id)
        distance = (number_pad_tokens_perturbed - number_pad_tokens_original) / number_pad_tokens_original
        similarity = exp(-distance**2/kernel_width**2)
        return similarity

    @classmethod
    def from_config(cls, config):
        res = super().from_config(config)
        res.n_samples = config['explainer']['n_samples']
        res.explainer = Lime(forward_func=res.forward_func, interpretable_model=SkLearnLasso(alpha=0.0))
        return res

    def explain(self, batch):
        self.model.eval()
        assert len(batch['input_ids']) == 1, 'This implementation assumes that the batch size is 1.'
        batch = {k: v.to(self.device) for k, v in batch.items()}
        inputs, additional_forward_args = self.get_inputs_and_additional_args(base_model=type(self.model.base_model),
                                                                              batch=batch)
        base_line = self.get_baseline(batch)
        predictions = self.forward_func(inputs, *additional_forward_args)
        target = torch.argmax(predictions, dim=1)
        attributions = self.explainer.attribute(
            inputs=inputs,
            additional_forward_args=additional_forward_args,
            target=target.item(),
            n_samples=self.n_samples,
            baselines=base_line,
        )
        return attributions, predictions


class ExplainerLimeBase(ExplainerAutoModelInitializer):
    def __init__(self):
        super().__init__()
        self.n_samples = None

    def validate_config(self, config: Dict) -> bool:
        super().validate_config(config)
        assert 'n_samples' in config['explainer'], 'Set number of samples for attribution function.'
        assert 'mask_prob' in config['explainer'], 'Set probability of masking a token in perturbation function.'
        assert 0 <= config['explainer']['mask_prob'] <= 1, 'Mask probability must be between 0 and 1.'

    @classmethod
    def from_config(cls, config):
        res = super().from_config(config)
        res.validate_config(config)
        res.n_samples = config['explainer']['n_samples']
        res.mask_prob = config['explainer']['mask_prob']
        res.interpretable_model = SkLearnLinearModel("linear_model.Ridge")

        res.explainer = LimeBase(forward_func=res.forward_func,
                                 interpretable_model=res.interpretable_model,
                                 similarity_func=cls.token_similarity_kernel,
                                 perturb_func=partial(ExplainerLimeBase.perturb_func,
                                                      res.mask_prob,
                                                      res.special_token_ids,
                                                      res.pad_token_id,
                                                      res.device),
                                 perturb_interpretable_space=False,
                                 from_interp_rep_transform=None,
                                 to_interp_rep_transform=cls.to_interp_rep_transform_custom)
        return res

    @staticmethod
    def token_similarity_kernel(
            original_input: torch.Tensor,
            perturbed_input: torch.Tensor,
            perturbed_interpretable_input: torch.Tensor,
            **kwargs) -> torch.Tensor:
        """
        Following https://github.com/copenlu/ALPS_2021
        """
        assert original_input.shape[0] == perturbed_input.shape[0] == 1, 'Batch size for LIME needs to be 1'
        return torch.sum(original_input[0] == perturbed_input[0]) / len(original_input[0])

    @staticmethod
    def perturb_func(
            mask_prob: float,
            special_token_ids: List[int],
            pad_token_id: int,
            device: str,
            original_input: torch.Tensor,  # always needs to be last argument before **kwargs due to "partial"
            **kwargs: Any) -> torch.Tensor:
        """
        Sampling function
        Following https://github.com/copenlu/ALPS_2021
        """
        # Build mask for replacing random tokens with [PAD] token
        mask_value_probs = torch.tensor([mask_prob, 1 - mask_prob])
        mask_multinomial_binary = torch.multinomial(mask_value_probs,
                                                    len(original_input[0]),
                                                    replacement=True)

        # Additionally remove special_token_ids
        mask_special_token_ids = torch.Tensor([1 if id_ in special_token_ids else 0
                                               for id_ in detach_to_list(original_input[0])]).int()

        # Merge the binary mask (12.5% masks) with the special_token_ids mask
        mask = torch.tensor([m + s if s == 0 else s for m, s in zip(
            mask_multinomial_binary, mask_special_token_ids)]).to(device)
        # TODO: Would it be faster if the other tensors above were also on the device (cuda)?

        # Apply mask to original input
        perturbed_input = original_input * mask + (1 - mask) * pad_token_id
        return perturbed_input

    @staticmethod
    def to_interp_rep_transform_custom(curr_sample, original_input, **kwargs: Any):
        return curr_sample

    def explain(self, batch):
        batch = {k: v.to(self.device) for k, v in batch.items()}
        inputs, additional_forward_args = self.get_inputs_and_additional_args(base_model=type(self.model.base_model),
                                                                              batch=batch)

        predictions = self.forward_func(inputs, *additional_forward_args)
        target = torch.argmax(predictions, dim=1)
        attributions = self.explainer.attribute(inputs=inputs,
                                                n_samples=self.n_samples,
                                                additional_forward_args=additional_forward_args,
                                                target=target)
        return attributions, predictions


class ExplainerKernelShap(ExplainerAutoModelInitializer):
    def __init__(self):
        raise NotImplementedError
        super().__init__()
        self.internal_batch_size = None
        self.n_samples = None

    def validate_config(self, config: Dict) -> bool:
        super().validate_config(config)
        assert 'n_samples' in config['explainer'], 'Set number of samples for attribution function.'

    @classmethod
    def from_config(cls, config):
        res = super().from_config(config)
        res.validate_config(config)
        res.n_samples = config['explainer']['n_samples']
        res.explainer = KernelShap(forward_func=res.forward_func, interpretable_model=SkLearnLasso(alpha=0.0))
        return res

    def explain(self, batch):
        assert len(batch['input_ids']) == 1, 'This implementation assumes that the batch size is 1.'
        batch = {k: v.to(self.device) for k, v in batch.items()}
        inputs, additional_forward_args = self.get_inputs_and_additional_args(base_model=type(self.model.base_model),
                                                                              batch=batch)

        predictions = self.forward_func(inputs, *additional_forward_args)
        target = torch.argmax(predictions, dim=1)
        base_line = self.get_baseline(batch=batch)
        attributions = self.explainer.attribute(inputs=inputs,
                                                n_samples=self.n_samples,
                                                additional_forward_args=additional_forward_args,
                                                target=target,
                                                baseline=base_line)
        return attributions, predictions
