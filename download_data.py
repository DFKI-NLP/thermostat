import logging

from thermometer.data.downloaders import DownloaderHF
from thermometer.utils import get_logger, read_config


config = read_config('configs/download/imdb.jsonnet')

logger = get_logger(name='pipeline', file_out='./pipeline.log', level=logging.INFO)

# Download data
downloader = DownloaderHF.from_config(config=config)
path_out = downloader.process(split='test',
                              save_dir='$HOME/experiments/thermometer/',
                              logger=logger)
