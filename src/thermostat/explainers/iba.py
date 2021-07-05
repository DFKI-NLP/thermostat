""" From: https://github.com/bazingagin/IBA/ """

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm
from tqdm import tqdm

from thermostat.explain import ExplainerAutoModelInitializer


class IBASequential(nn.Sequential):
    def forward(self, *inp):
        for module in self._modules.values():
            inp = module(*inp)
        return inp


class Estimator:
    """
    Useful to calculate the empirical mean and variance of intermediate feature maps.
    """
    def __init__(self, layer):
        self.layer = layer
        self.M = None  # running mean for each entry
        self.S = None  # running std for each entry
        self.N = None  # running num_seen for each entry
        self.num_seen = 0  # total samples seen
        self.eps = 1e-5

    def feed(self, z: np.ndarray):

        # Initialize if this is the first datapoint
        if self.N is None:
            self.M = np.zeros_like(z, dtype=float)
            self.S = np.zeros_like(z, dtype=float)
            self.N = np.zeros_like(z, dtype=float)

        self.num_seen += 1

        diff = (z - self.M)
        self.N += 1
        self.M += diff / self.num_seen
        self.S += diff * (z - self.M)

    def feed_batch(self, batch: np.ndarray):
        for point in batch:
            self.feed(point)

    def shape(self):
        return self.M.shape

    def is_complete(self):
        return self.num_seen > 0

    def get_layer(self):
        return self.layer

    def mean(self):
        return self.M.squeeze()

    def p_zero(self):
        return 1 - self.N / (self.num_seen + 1)  # Adding 1 for stablility, so that p_zero > 0 everywhere

    def std(self, stabilize=True):
        if stabilize:
            # Add small numbers, so that dead neurons are not a problem
            return np.sqrt(np.maximum(self.S, self.eps) / np.maximum(self.N, 1.0))

        else:
            return np.sqrt(self.S / self.N)

    def estimate_density(self, z):
        z_norm = (z - self.mean()) / self.std()
        p = norm.pdf(z_norm, 0, 1)
        return p

    def normalize(self, z):
        return (z - self.mean()) / self.std()

    def load(self, what):
        state = what if not isinstance(what, str) else torch.load(what)
        # Check if estimator classes match
        if self.__class__.__name__ != state["class"]:
            raise RuntimeError("This Estimator is {}, cannot load {}".format(self.__class__.__name__, state["class"]))
        # Check if layer classes match
        if self.layer.__class__.__name__ != state["layer_class"]:
            raise RuntimeError("This Layer is {}, cannot load {}".format(self.layer.__class__.__name__,
                                                                         state["layer_class"]))
        self.N = state["N"]
        self.S = state["S"]
        self.M = state["M"]
        self.num_seen = state["num_seen"]


class InformationBottleneck(nn.Module):
    def __init__(self, mean: np.ndarray, std: np.ndarray, device=None):
        super().__init__()
        self.device = device
        self.initial_value = 5.0
        self.std = torch.tensor(std, dtype=torch.float, device=self.device, requires_grad=False)
        self.mean = torch.tensor(mean, dtype=torch.float, device=self.device, requires_grad=False)
        self.alpha = nn.Parameter(torch.full((1, *self.mean.shape), fill_value=self.initial_value, device=self.device))
        self.sigmoid = nn.Sigmoid()
        self.buffer_capacity = None

        self.reset_alpha()

    @staticmethod
    def _sample_t(mu, log_noise_var):
        log_noise_var = torch.clamp(log_noise_var, -10, 10)
        noise_std = (log_noise_var / 2).exp()
        eps = mu.data.new(mu.size()).normal_()
        return mu + noise_std * eps

    @staticmethod
    def _calc_capacity(mu, log_var):
        # KL[Q(t|x)||P(t)]
        return -0.5 * (1 + log_var - mu**2 - log_var.exp())

    def reset_alpha(self):
        with torch.no_grad():
            self.alpha.fill_(self.initial_value)
        return self.alpha

    def forward(self, bert_out):
        x = bert_out
        lamb = self.sigmoid(self.alpha)
        lamb = lamb.expand(x.shape[0], x.shape[1], -1)
        # We normalize x to simplify the computation of the KL-divergence
        x_norm = (x - self.mean) / self.std
        # Get sampling parameters
        noise_var = (1-lamb)**2
        scaled_signal = x_norm * lamb
        noise_log_var = torch.log(noise_var)
        # Sample new output values from p(t|x)
        t_norm = self._sample_t(scaled_signal, noise_log_var)
        self.buffer_capacity = self._calc_capacity(scaled_signal, noise_log_var)
        # Denormalize t to match x
        t = t_norm * self.std + self.mean
        return (t,)


class ExplainerIBA(ExplainerAutoModelInitializer):
    def __init__(self, beta=1e-5, steps=10, lr=1, batch_size=1, progbar=False):
        super().__init__()
        self.beta = beta
        self.batch_size = batch_size
        self.progbar = progbar
        self.lr = lr
        self.train_steps = steps
        self.ce_loss = []
        self.info_loss = []
        self.total_loss = []
        self.pred = []

        self.layer_idx = None
        self.original_layer = None
        self.shape = None
        self.bottleneck = None
        self.sequential = None
        self.estim = None

    @classmethod
    def from_config(cls, config):
        res = super().from_config(config)

        def extract_bert_layer(model, layer_idx):
            desired_layer = ''
            for name, submodule in model.named_children():
                for n, s in submodule.named_children():
                    if n == 'encoder':
                        for n2, s2 in s.named_children():
                            for n3, s3 in s2.named_children():
                                if n3 == str(layer_idx):
                                    desired_layer = s3
                                    return desired_layer

        res.layer_idx = config['explainer']['layer_idx']
        res.layer = extract_bert_layer(res.model, res.layer_idx)

        res.estim = Estimator(res.layer)

        res.original_layer = res.estim.get_layer()
        return res

    def attribute(self, inputs, target, segment=None):
        saliency = self.buff_cap(inputs, target, segment)
        saliency = saliency[0]
        saliency = saliency.cpu().detach().numpy()
        saliency = saliency.sum(axis=1)
        saliency = saliency - saliency.min()
        saliency = saliency / saliency.max()
        return torch.tensor(saliency).unsqueeze(0)

    def buff_cap(self, input_t, target, segment=None):
        target_t = torch.tensor([target]) if not isinstance(target, torch.Tensor) else target
        self._run_training(input_t, target_t, segment)
        return self.explainer.buffer_capacity

    def _run_training(self, input_t, target_t, segment=None):
        # Attach layer and train the bottleneck
        replace_layer(self.model, self.original_layer, self.sequential)
        input_t = input_t.to(self.device)
        target_t = target_t.to(self.device)
        self._train_bottleneck(input_t, target_t, segment)
        replace_layer(self.model, self.sequential, self.original_layer)

    def _train_bottleneck(self, input_t: torch.Tensor, target_t: torch.Tensor, segment=None):
        batch = input_t.expand(self.batch_size, -1), target_t.expand(self.batch_size)
        optimizer = torch.optim.Adam(lr=self.lr, params=self.explainer.parameters())
        # Reset from previous run or modifications
        self.explainer.reset_alpha()
        # Train
        self.model.eval()
        for _ in tqdm(range(self.train_steps), desc="Training Bottleneck",
                      disable=not self.progbar):
            optimizer.zero_grad()
            if segment is not None:
                out = self.model(batch[0], token_type_ids=segment)
            else:
                out = self.model(batch[0])
            # print(out[0])
            loss_t = self.calc_loss(outputs=out[0], labels=batch[1])
            loss_t.backward()
            optimizer.step(closure=None)
            self.pred = out[0]

    def calc_loss(self, outputs, labels):
        """ Calculate the combined loss expression for optimization of lambda """
        information_loss = self.explainer.buffer_capacity.mean()  # Taking the mean is equivalent of scaling with 1/K
        cross_entropy = F.cross_entropy(outputs, target=labels)
        total = cross_entropy + self.beta * information_loss
        self.ce_loss.append(cross_entropy.cpu().detach().numpy())
        self.info_loss.append(information_loss.cpu().detach().numpy())
        self.total_loss.append(total.cpu().detach().numpy())
        return total

    def return_loss(self):
        return self.ce_loss, self.info_loss, self.total_loss

    def explain(self, batch):
        batch = {k: v.to(self.device) for k, v in batch.items()}
        inputs, additional_forward_args = self.get_inputs_and_additional_args(base_model=type(self.model.base_model),
                                                                              batch=batch)
        predictions = self.forward_func(inputs, *additional_forward_args)
        target = torch.argmax(predictions, dim=1)

        feature_map = list(self.model.base_model(inputs)[1][0])[self.layer_idx]

        self.estim.feed(feature_map.detach().cpu().numpy())
        self.shape = self.estim.shape()
        self.explainer = InformationBottleneck(self.estim.mean(), self.estim.std(), device=self.device)
        self.sequential = IBASequential(self.original_layer, self.explainer)

        attributions = self.attribute(inputs=inputs,
                                      target=target)  # TODO: Add additional_forward_args
        return attributions, predictions


def replace_layer(model: nn.Module, target: nn.Module, replacement: nn.Module):
    """
    Replace a given module within a parent module with some third module
    Useful for injecting new layers in an existing model.
    """
    def replace_in(model: nn.Module, target: nn.Module, replacement: nn.Module):
        for name, submodule in model.named_children():
            if submodule == target:
                if isinstance(model, nn.ModuleList):
                    model[int(name)] = replacement
                elif isinstance(model, nn.Sequential):
                    model[int(name)] = replacement
                else:
                    model.__setattr__(name, replacement)
                return True
            elif len(list(submodule.named_children())) > 0:
                if replace_in(submodule, target, replacement):
                    return True
        return False

    if not replace_in(model, target, replacement):
        raise RuntimeError("Cannot substitute layer: Layer of type " + target.__class__.__name__ +
                           " is not a child of given parent of type " + model.__class__.__name__)
