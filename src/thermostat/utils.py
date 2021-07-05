import json
import _jsonnet
import logging
import os
import torch

from datetime import datetime
from os.path import expanduser
from typing import Dict


class Configurable:

    def __init__(self):
        """A  DataBroker is an abstract class for down loaders, tokenizers and other components that handle download,
        convert and write Datapoints.
        """
        pass

    def validate_config(self, config: Dict) -> bool:
        """Validate a config file. Is true if all required fields to configure this downloader are present.
        :param config: The configuration file to validate.
        :returns: True if all required fields exist, else False.
        """
        raise NotImplementedError

    @classmethod
    def from_config(cls, config: Dict):
        """Initializes the Preprocessor from a config file. The required fields in the config file are validated in
        validate_config.
        :param config: The config file to initialize this Downloader from.
        :return: The configured Downloader.
        """
        res = cls()
        res.validate_config(config)
        for k, v in config.items():
            assert k in res.__dict__, f'Unknown key: {k}'
            setattr(res, k, v)
        return res


class HookableModelWrapper(torch.nn.Module):
    def __init__(self, res):
        super().__init__()
        self.model = res.model
        self.model.zero_grad()
        self.forward = res.forward_func


def detach_to_list(t):
    return t.detach().cpu().numpy().tolist() if type(t) == torch.Tensor else t


def delistify(lst):
    return list(map(lambda x: x[0] if isinstance(x, list) else x, lst))


def get_logger(name: str, file_out: str = None, level: int = None):
    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(level)

    # Create handlers
    c_handler = logging.StreamHandler()
    if level is not None:
        c_handler.setLevel(level)
    c_format = logging.Formatter('%(asctime)s -%(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(c_format)
    logger.addHandler(c_handler)

    if file_out is not None:
        f_handler = logging.FileHandler(file_out, mode='a+')
        if level is not None:
            f_handler.setLevel(level)
        f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        f_handler.setFormatter(f_format)
        logger.addHandler(f_handler)

    return logger


def get_time():
    now = datetime.now()
    now = now.strftime("%Y-%m-%d-%H-%M-%S")
    return now


def read_config(config_path, home_dir=None) -> Dict:
    config = json.loads(_jsonnet.evaluate_file(config_path))

    # Config fields where $HOME needs to be resolved to a real directory
    config["path"] = read_path(config["path"], home=home_dir)
    config["dataset"]["root_dir"] = read_path(config["dataset"]["root_dir"], home=home_dir)
    config["model"]["path_model"] = read_path(config["model"]["path_model"], home=home_dir)

    if 'subset' in config['dataset']:
        dataset_name = f'{config["dataset"]["name"]}-{config["dataset"]["subset"]}'
    else:
        dataset_name = config['dataset']['name']

    # Set experiment path
    experiment_path = f'{config["path"]}' \
        f'/{dataset_name}/{"/".join(config_path.split("/")[2:]).split(".jsonnet")[0]}'
    if not os.path.exists(experiment_path):
        raise NotADirectoryError(f'{experiment_path}\nThis experiment path does not exist yet.')

    config['experiment_path'] = experiment_path
    return config


def read_path(path, home=None):
    """Replaces $HOME in a path (str) with the home directory"""
    if not home:
        home = expanduser("~")
    return path.replace("$HOME", home) if path else path


def lazy_property(fn):
    """ from: https://stevenloria.com/lazy-properties/
    Decorator that makes a property lazy-evaluated (only calculated when explicitly accessed). """
    attr_name = '_lazy_' + fn.__name__

    @property
    def _lazy_property(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)

    return _lazy_property
