import json
import torch
from overrides import overrides
from torch.utils.data import Dataset
from typing import Dict

from ..utils import read_path, Configurable


class DatasetProcessedDatapoints(Configurable, Dataset):

    def __init__(self):
        super().__init__()
        self.path_in = None
        self.size = None
        self.name_input = None
        self.name_model = None
        self.input_keys = None

    def __len__(self):
        return self.size

    @staticmethod
    def no_of_lines(path_in):
        fin = open(path_in, 'r+')
        result = len(fin.readlines())
        fin.close()
        return result

    @overrides
    def __getitem__(self, idx):
        file_in = open(self.path_in, 'r+')
        for idx_line, line in enumerate(file_in.readlines()):
            if idx_line == idx:
                json_line = json.loads(line)
                tensors = json_line['inputs'][self.name_input]['tensors']
                tensors['id'] = torch.LongTensor([json_line['id']])
                tensors = {k: torch.LongTensor(v) for k, v in tensors.items() if k in self.input_keys}
                return tensors

    @overrides
    def validate_config(self, config: Dict) -> bool:
        assert 'path_in' in config, 'Specify input path'
        assert 'name_input' in config, 'Specify input name'
        assert 'name_model' in config, 'Specify model name'
        assert 'input_keys' in config, 'Specify input keys'

    @classmethod
    def from_config(cls, config: Dict):
        res: DatasetProcessedDatapoints = super().from_config(config)
        res.path_in = read_path(res.path_in)
        res.size = res.no_of_lines(res.path_in)
        return res


class DatasetColoredDatapoints(Configurable, Dataset):
    """A dataset reader which gives additional access to attributions, i.e. to train a generative explainer."""

    def __init__(self):
        super().__init__()
        self.path_in = None # path_in
        self.size = None # self.no_of_lines(self.path_in)
        self.name_input = None # name_input
        self.name_explanation = None
        self.name_model = None # name_model
        self.mode = None # mode
        # self.keys = None # self.keys_options[f'{self.name_model}.{self.mode}']
        self.input_keys = None
        self.flip_attributions_on_index = None

    def __len__(self):
        return self.size

    @staticmethod
    def no_of_lines(path_in):
        fin = open(path_in, 'r+')
        result = len(fin.readlines())
        fin.close()
        return result

    @overrides
    def __getitem__(self, idx):
        file_in = open(self.path_in, 'r+')  # todo: very inefficient, consider keeping file reader open
        for idx_line, line in enumerate(file_in.readlines()):
            if idx_line == idx:
                json_line = json.loads(line)
                tensors = json_line['inputs'][self.name_input]['tensors']
                tensors['id'] = torch.LongTensor([json_line['id']])
                tensors = {k: torch.LongTensor(v) for k, v in tensors.items() if k in self.input_keys}
                attributions = json_line['explanation'][self.name_explanation]['attribution']
                if self.flip_attributions_on_index is not None:
                    assert isinstance(self.flip_attributions_on_index, int), "flip_attributions_on_index should be int"
                    prediction = json_line['explanation'][self.name_explanation]['prediction']
                    max_idx = prediction.index(max(prediction))
                    if max_idx == self.flip_attributions_on_index:
                        attributions = [score * -1 for score in attributions]
                tensors['attribution'] = torch.Tensor(attributions)
                return tensors

    @overrides
    def validate_config(self, config: Dict) -> bool:
        assert 'path_in' in config, 'Specify input path'
        assert 'name_input' in config, 'Specify input name'
        assert 'input_keys' in config, 'Specify input keys'

    @classmethod
    def from_config(cls, config: Dict):
        res: DatasetProcessedDatapoints = super().from_config(config)
        res.path_in = read_path(res.path_in)
        res.size = res.no_of_lines(res.path_in)
        assert "labels" not in res.input_keys, "Sanity check failed."
        return res


class JsonIterator:

    def __call__(self, path_in: str, batch_size: int):
        self.path_in = path_in,
        self.batch_size = batch_size
        self.file_in = open(path_in, 'r+')
        return self

    def __iter__(self):
        return self

    def __next__(self):
        batch = []
        for _ in range(self.batch_size):
            batch.append(self.file_in.readline())
        return batch


class ShapleyJsonToTensor(Configurable, Dataset):

    def __init__(self):
        self.path_in = None
        self.size = None

    def validate_config(self, config: Dict) -> bool:
        assert 'path_in_shapley' in config, 'No input path found.'
        assert 'name_model' in config, 'No model name found.'
        return True

    @classmethod
    def from_config(cls, config: Dict):
        res = cls()
        res.path_in = read_path(config['path_in_shapley'])
        res.name_model = config['name_model']
        res.size = DatasetColoredDatapoints.no_of_lines(res.path_in)
        return res

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        file_in = open(self.path_in, 'r+')  # todo: very inefficient, consider keeping file reader open
        for idx_line, line in enumerate(file_in.readlines()):
            if idx_line == idx:
                json_line = json.loads(line)
                assert (json_name := json_line['model_name']) == self.name_model,\
                    f'Model names differ {json_name} != {self.name_model} sanity check failed.'
                tensors = {'id': [json_line['id']],
                           'n_samples': [json_line['n_samples']],
                           'attribution': json_line['scores'],
                           'prediction': json_line['prediction']}
                tensors = {k: torch.Tensor(v) for k, v in tensors.items()}
                return tensors
