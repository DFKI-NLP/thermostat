import json
import _jsonnet
import logging
import os

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


def detach_to_list(t):
    return t.detach().cpu().numpy().tolist()


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
        f_handler = logging.FileHandler(file_out, mode='w+')
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


def read_config(path) -> Dict:
    config = json.loads(_jsonnet.evaluate_file(path))
    config['experiment_path'] = set_experiment_path(config, path)
    return config


def set_experiment_path(config, config_path) -> str:
    experiment_path = f'{read_path(config["path"])}' \
        f'/{config["dataset"]["subset"] if "subset" in config["dataset"] else config["dataset"]["name"]}' \
        f'/{"/".join(config_path.split("/")[2:]).split(".jsonnet")[0]}'
    if not os.path.exists(experiment_path):
        raise NotADirectoryError(f'{experiment_path}\nThis experiment path does not exist yet.')
    return experiment_path


def read_path(path):
    """Replaces $HOME w/ home directory."""
    home = expanduser("~")
    return path.replace("$HOME", home)
