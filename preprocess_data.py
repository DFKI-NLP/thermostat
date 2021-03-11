import logging

from thermostat.data.preprocessors import PreprocessorSingleTextAutoTokenizer
from thermostat.utils import get_logger, read_config


config = read_config('configs/preprocess/roberta-base.jsonnet')

logger = get_logger(name='pipeline', file_out='./pipeline.log', level=logging.INFO)

# Preprocess data
preprocessor = PreprocessorSingleTextAutoTokenizer.from_config(config=config)
_ = preprocessor.process(path_in='$HOME/experiments/thermometer/2021-01-26-14-57-46.download.imdb.test.jsonl',
                         save_dir='$HOME/experiments/thermometer/',
                         logger=logger)
