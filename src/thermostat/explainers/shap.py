import torch
from captum.attr import (
    LayerGradientShap, LayerDeepLiftShap,)
from transformers import XLNetForSequenceClassification
from typing import Dict

from thermostat.explain import ExplainerAutoModelInitializer
from thermostat.utils import HookableModelWrapper


class ExplainerLayerGradientShap(ExplainerAutoModelInitializer):
    def __init__(self):
        super().__init__()
        self.name_layer: str = None
        self.layer = None
        self.n_samples: int = None

    def validate_config(self, config: Dict) -> bool:
        super().validate_config(config)
        assert 'n_samples' in config['explainer'], \
            'Define how many samples to take along the straight line path from the baseline.'

    @classmethod
    def from_config(cls, config):
        res = super().from_config(config)
        res.validate_config(config)
        res.n_samples = config['explainer']['n_samples']
        
        res.model.eval() # setting the model to eval and zero grad 
        res.model.zero_grad() # so we do not have to loop it in explainer for each batch.
        
        res.explainer = LayerGradientShap(forward_func=res.forward_func,
                                                 layer=res.get_embedding_layer(res.model))
        return res

    def explain(self, batch):
        
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        inputs, additional_forward_args = self.get_inputs_and_additional_args(base_model=type(self.model.base_model),
                                                                              batch=batch)
        
        predictions = self.forward_func(inputs, *additional_forward_args)
        target = torch.argmax(predictions, dim=1)
        
        base_line = self.get_baseline(batch=batch)
        
        attributions = self.explainer.attribute(inputs=inputs,
                                                baselines=base_line,
                                                n_samples =self.n_samples,
                                                target=target,
                                                additional_forward_args=additional_forward_args)
                                                
        attributions = torch.sum(attributions, dim=2)
        
        if isinstance(self.model, XLNetForSequenceClassification):
            # for xlnet, attributions.shape = [seq_len, batch_dim]
            # but [batch_dim, seq_len] is assumed
            attributions = attributions.T 
            
        return attributions, predictions  # xlnet: [130, 1]



class ExplainerLayerDeepLiftShap(ExplainerAutoModelInitializer):
    def __init__(self):
        super().__init__()

    @classmethod
    def from_config(cls, config):
        res = super().from_config(config)
        
        res.model.eval() # setting the model to eval and zero grad 
        res.model.zero_grad() # so we do not have to loop it in explainer for each batch.
        
        res.explainer = LayerDeepLiftShap(model=HookableModelWrapper(res), layer=res.get_embedding_layer(res.model))
        
        return res

    def explain(self, batch):

        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        inputs, additional_forward_args = self.get_inputs_and_additional_args(base_model=type(self.model.base_model),
                                                                              batch=batch)
        base_line = self.get_baseline(batch)
        base_line = torch.stack((torch.zeros_like(inputs[0]), base_line[0])) # stacking to provide more than one example for base line
        
        predictions = self.forward_func(inputs, *additional_forward_args)
        target = torch.argmax(predictions, dim=1)
        
        attributions = self.explainer.attribute(
            inputs=inputs.float(),
            additional_forward_args=additional_forward_args,
            target=target,
            baselines=base_line.float(),
        )
        
        attributions = torch.sum(attributions, dim=2)
        
        if isinstance(self.model, XLNetForSequenceClassification):
            # for xlnet, attributions.shape = [seq_len, batch_dim]
            # but [batch_dim, seq_len] is assumed
            attributions = attributions.T
        
        return attributions, predictions
