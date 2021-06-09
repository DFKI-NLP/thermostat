import argparse
import logging

from thermostat.data.readers import download_dataset
from thermostat.utils import get_logger, read_config


# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-c', help='Config file',
                    default='configs/xnli/bert/lgxa.jsonnet')
parser.add_argument('-home', help='Home directory', default=None)
args = parser.parse_args()
config_file = args.c
home_dir = args.home

logger = get_logger(name='download', file_out='./download.log', level=logging.INFO)

config = read_config(config_file, home_dir)
path_out = download_dataset(config['dataset'], logger)
