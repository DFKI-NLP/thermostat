import logging

from thermometer.data.readers import download_dataset
from thermometer.utils import get_logger, read_config


logger = get_logger(name='download', file_out='./download.log', level=logging.INFO)
config = read_config('configs/exp-a02_imdb_LayerIntegratedGradients_bert-base-cased.jsonnet')
path_out = download_dataset(config['dataset'], logger)
