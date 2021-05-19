import logging

from thermostat.data.readers import download_dataset
from thermostat.utils import get_logger, read_config


logger = get_logger(name='download', file_out='./download.log', level=logging.INFO)
config = read_config('configs/imdb/bert/LayerIntegratedGradients.jsonnet')
path_out = download_dataset(config['dataset'], logger)
