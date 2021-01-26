import json
import logging
import numpy as np
import os
from overrides import overrides
from tqdm import tqdm
from transformers import AutoTokenizer
from typing import Dict, Union

from thermometer.data.dtypes import Datapoint, DatapointProcessed
from thermometer.utils import get_logger, get_time, read_path, Configurable


class Processor:

    def process(self):
        raise NotImplementedError


class PreprocessorSingleTextAutoTokenizer(Configurable, Processor):

    def __init__(self):
        """This preprocessor tokenizes a dataset from HF which holds a single text (as oppoed to e.g. premise +
        hypothesis; for this see below) which was previously downloaded into DataPoints.
        It takes a Datapoint and writes the field input_model, which makes the Datapoint a ProcessedDatapoint.
        This class should be initialized from config.
        """
        super().__init__()
        self.name_tokenizer: str = None
        self.tokenizer: AutoTokenizer = None
        self.padding: Union[bool, str] = None
        self.max_length: int = None
        self.return_tensors: str = None
        self.truncation: bool = None

    @overrides
    def validate_config(self, config: Dict) -> bool:
        assert 'name_tokenizer' in config, 'No tokenizer name specified'
        assert 'padding' in config, 'No padding strategy defined'
        assert 'max_length' in config, 'No max_length defined'
        assert 'return_tensors' in config, 'No tensor type defined'
        assert 'truncation' in config, 'No truncation strategy defined'
        return True

    @overrides
    def process(self, path_in: str, save_dir: str, logger) -> str:
        _now = get_time()
        _, path_file_in = os.path.split(os.path.realpath(path_in))
        path_file_in = '.'.join(path_file_in.split('.')[:-1])  # cut file extension
        path_out_file = os.path.join(read_path(save_dir),
                                     f'{_now}.preprocess.{self.name_tokenizer}.{path_file_in}.jsonl')
        path_out_log = path_out_file + '.log'

        logger = get_logger(name=f'preprocess.{self.name_tokenizer}.imdb',
                            file_out=path_out_log,
                            level=logging.INFO)
        logger.info(f'(Config) \n{json.dumps(self.__dict__, indent=2)}\n')
        logger.info(f'(Config) Will write log to {path_out_log}')
        logger.info(f'(Config) Output file: {path_out_file}')

        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.name_tokenizer)

        with open(path_out_file, 'a+') as file_out:
            with open(read_path(path_in), 'r+') as file_in:
                for line in tqdm(file_in.readlines()):
                    datapoint = Datapoint.from_dict(dct=json.loads(line))
                    # TODO: make tokenizer fields part of config
                    batch_encoding = self.tokenizer(datapoint.data['text'],
                                                    max_length=self.max_length,
                                                    padding=self.padding,
                                                    truncation=self.truncation,
                                                    return_tensors=self.return_tensors,
                                                    return_special_tokens_mask=True)
                    tensors = {'input_ids': np.squeeze(batch_encoding['input_ids']).tolist(),
                               'labels': [datapoint.data['label']],
                               'special_tokens_mask': np.squeeze(batch_encoding['special_tokens_mask']).tolist()}
                    if 'token_type_ids' in batch_encoding:
                        tensors['token_type_ids'] = np.squeeze(batch_encoding['token_type_ids']).tolist()
                    if 'attention_mask' in batch_encoding:
                        tensors['attention_mask'] = np.squeeze(batch_encoding['attention_mask']).tolist()

                    config_tokenizer = {'max_length': self.max_length,
                                        'padding': self.padding,
                                        'return_tensors': self.return_tensors,
                                        'truncation': self.truncation}

                    tokens = [self.tokenizer.decode([input_id]) for input_id in tensors['input_ids']]

                    input_model = {'name_tokenizer': self.name_tokenizer,
                                   'tokens': tokens,
                                   'config_tokenizer': config_tokenizer,
                                   'tensors': tensors}

                    datapoint = DatapointProcessed.from_parent_class(datapoint=datapoint,
                                                                     name_input=self.name_tokenizer,
                                                                     input_model=input_model)
                    line = str(datapoint) + os.linesep
                    file_out.write(line)
                    file_out.flush()
        return path_out_file
