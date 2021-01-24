import json

from typing import Dict


class Datapoint:

    def __init__(self,
                 name_dataset: str = None,
                 id: int = None,
                 data_raw: Dict = None,
                 split=None,
                 version=None):
        """
        A Datapoint comes from a huggingface dataset which can be identified with the coordinates
        name_dataset, split and version.
        :param name_dataset: The name of the huggingface dataset.
        :param id: The id of this data_deprecated point.
        :param data_raw: The raw huggingface data_deprecated.
        :param split: The split, e.g. test or train.
        :param version: The version of the dataset.
        """
        self.name_dataset = name_dataset
        self.id = id
        self.data = data_raw
        self.split = split
        self.version = version

    def __repr__(self):
        return json.dumps(self.__dict__)

    def __str__(self):
        return self.__repr__()

    @classmethod
    def from_dict(cls, dct):
        res = cls()
        res.name_dataset = dct['name_dataset']
        res.id = dct['id']
        res.data = dct['data']
        res.split = dct['split']
        res.version = dct['version']
        return res

    @classmethod
    def from_parent_class(cls, *args, **kwargs):
        raise NotImplementedError


class DatapointProcessed(Datapoint):

    def __init__(self,
                 inputs: Dict[str, Dict] = {},
                 *args, **kwargs):
        """
        A processed datapoint is a datapoint with tokenized and tensorized inputs to an NLP model.
        :param input_model: The inputs to an NLP model, e.g. {'bert-base_cased': {'tensors': {'input_ids': ...}}, ...}
        :param name_input: An identifier for this model input, i.e. bert-base-cased-length-512.
        """
        super().__init__(*args, **kwargs)
        self.inputs = inputs

    def append_input(self, name_input: str, input_model: Dict):
        """
        Appends an input, i.e. the token ids etc. from an additional tokenizer to this data_deprecated point. 
        :param input_model: The input to append. 
        """""
        assert name_input not in self.inputs, f'Input {name_input} already contained.'
        self.inputs[name_input] = input_model

    @classmethod
    def from_parent_class(cls, name_input, input_model: Dict, datapoint: Datapoint):
        assert isinstance(datapoint, Datapoint), \
            f'Parent class is Datapoint but found {type(datapoint)}'
        res = DatapointProcessed(inputs={name_input: input_model})
        for k, v in datapoint.__dict__.items():
            setattr(res, k, v)
        return res

    @classmethod
    def from_dict(cls, dct):
        res = cls()
        res.name_dataset = dct['name_dataset']
        res.id = dct['id']
        res.data = dct['data']
        res.split = dct['split']
        res.version = dct['version']
        if 'inputs' in dct:
            res.inputs = dct['inputs']
        return res


class DatapointColored(DatapointProcessed):

    def __init__(self,
                 explanations: Dict[str, Dict] = {},
                 *args, **kwargs):
        """
        A colored datapoint is a datapoint with tokenized and tensorized inputs to an NLP model + saliency maps for
        at least one input.
        :param explanations: The saliency maps corresponding to a model input, e.g.
        {'integrated_gradients': {'config': {...}, 'saliencies': [...]}.
        :param name_input: An identifier for this saliency map, e.g. 'integrated_gradients_distilbert'.
        """
        super().__init__(*args, **kwargs)
        self.explanation = explanations # todo: is called explanationS

    def append_explanation(self, name_explanation: str, explanation: Dict):
        """
        Appends a saliency map to this point. 
        :param name_explanation: The name, which will be used as a key, to identify the saliency map with.  
        :param explanation: The saliency map. 
        """""
        assert name_explanation not in self.explanation, f'Saliency map {name_explanation} already contained.'
        # assert (name := explanation['name_input']) in self.inputs, f'Input not found: {name}'
        self.explanation[name_explanation] = explanation

    @classmethod
    def from_parent_class(cls, datapoint_processed: DatapointProcessed, name_explanation, explanation):
        assert isinstance(datapoint_processed, DatapointProcessed), \
            f'Parent class is DatapointProcessed but found {type(datapoint_processed)}'
        res = DatapointColored(explanations={name_explanation: explanation})
        for k, v in datapoint_processed.__dict__.items():
            setattr(res, k, v)
        return res

    @classmethod
    def from_dict(cls, dct):
        res = cls()
        res.name_dataset = dct['name_dataset']
        res.id = dct['id']
        res.data = dct['data']
        res.split = dct['split']
        res.version = dct['version']
        if 'inputs' in dct:
            res.inputs = dct['inputs']
        if 'explanation' in dct:
            res.explanation = dct['explanation']
        return res
