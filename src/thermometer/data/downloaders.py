import json
import logging
import os
from datasets import load_dataset
from overrides import overrides
from tqdm import tqdm
from typing import Dict, List

from thermometer.data.dtypes import Datapoint
from thermometer.data.preprocessors import Processor
from thermometer.utils import get_logger, get_time, read_path, Configurable


class DownloaderHF(Configurable, Processor):

    def __init__(self):
        """
        This downloader gets a HF dataset and converts it into DataPoints.
        This object should be initialized from config.
        """
        super().__init__()
        self.name: str = ""
        self.input_keys: List = []
        self.splits: List = []

    @overrides
    def validate_config(self, config: Dict):
        return True

    @overrides
    def process(self, split: str, save_dir: str, logger) -> str:
        assert split in self.splits, f'Unknown split {split}'

        _now = get_time()
        path_out_file = os.path.join(read_path(save_dir), f'{_now}.download.{self.name}.{split}.jsonl')
        path_out_log = path_out_file + '.log'
        logger = get_logger(name=f'download.{self.name}.{split}', file_out=path_out_log, level=logging.INFO)
        logger.info(f'(Config) \n{json.dumps(self.__dict__, indent=2)}\n')
        logger.info(f'(Config) Will write log to {path_out_log}')
        logger.info(f'(Config) Output file: {path_out_file}')
        file_out = open(path_out_file, 'a+')
        counter_id = 0
        dataset = load_dataset(self.name, split=split)
        for entry in tqdm(dataset):
            counter_id = counter_id + 1
            datapoint = Datapoint(
                name_dataset=self.name,
                split=split,
                id=counter_id,
                data_raw=entry,
                version=str(dataset.version))
            line = str(datapoint) + os.linesep
            file_out.write(line)
            file_out.flush()
        return path_out_file
