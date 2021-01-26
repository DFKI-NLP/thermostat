import json
import logging
import os
from datasets import load_dataset
from overrides import overrides
from tqdm import tqdm
from typing import Dict

from thermometer.data.dtypes import Datapoint
from thermometer.data.preprocessors import Processor
from thermometer.utils import get_logger, get_time, read_path, Configurable


class DownloaderIMDB(Configurable, Processor):

    def __init__(self):
        """
        This downloader gets the IMDB dataset and converts it into DataPoints.
        This object should be initialized from config.
        """
        super().__init__()
        self.path_dir_out = None
        self.split = None

    @overrides
    def validate_config(self, config: Dict):
        assert 'path_dir_out' in config, 'No output path specified'
        assert 'split' in config, 'No split specified'
        split = config['split']
        assert split in ['train', 'test'], f'Unknown split {split}'
        return True

    @overrides
    def process(self, logger) -> None:
        _now = get_time()
        path_out_file = os.path.join(read_path(self.path_dir_out), f'{_now}.download.imdb.{self.split}.jsonl')
        path_out_log = path_out_file + '.log'
        logger = get_logger(name=f'download.imdb.{self.split}', file_out=path_out_log, level=logging.INFO)
        logger.info(f'(Config) \n{json.dumps(self.__dict__, indent=2)}\n')
        logger.info(f'(Config) Will write log to {path_out_log}')
        logger.info(f'(Config) Output file: {path_out_file}')
        file_out = open(path_out_file, 'a+')
        counter_id = 0
        dataset = load_dataset('imdb', split=self.split)
        for entry in tqdm(dataset):
            counter_id = counter_id + 1
            datapoint = Datapoint(
                name_dataset='imdb',
                split=self.split,
                id=counter_id,
                data_raw=entry)
            line = str(datapoint) + os.linesep
            file_out.write(line)
            file_out.flush()
        return path_out_file
