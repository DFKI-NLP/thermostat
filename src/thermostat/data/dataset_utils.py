from datasets import Dataset
from transformers import AutoTokenizer
from typing import List

from thermostat.visualize import run_visualize


def get_coordinate(thermostat_dataset: Dataset, coordinate: str) -> str:
    """ Determine a coordinate (dataset, model, or explainer) of a Thermostat dataset from its description """
    assert coordinate in ['Model', 'Dataset', 'Explainer']
    coord_prefix = f'{coordinate}: '
    assert coord_prefix in thermostat_dataset.description
    str_post_coord_prefix = thermostat_dataset.description.split(coord_prefix)[1]
    if '\n' in str_post_coord_prefix:
        coord_value = str_post_coord_prefix.split('\n')[0]
    else:
        coord_value = str_post_coord_prefix
    return coord_value


def get_tokens(thermostat_dataset: Dataset) -> List:
    """ Decode input ids from a Thermostat dataset and return tokens """
    model_id = get_coordinate(thermostat_dataset, coordinate='Model')
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return [tokenizer.decode(token_ids=token_ids) for token_ids in thermostat_dataset['input_ids']]


def to_html(thermostat_dataset: Dataset, out_html: str):
    """ Run the visualization script on a Thermostat dataset """
    config = dict()
    config["path_html"] = out_html
    config["dataset"] = {"name": get_coordinate(thermostat_dataset, coordinate="Dataset"),
                         "split": "test",  # TODO: Check if hard-coding this makes sense
                         }
    config["model"] = {"name": get_coordinate(thermostat_dataset, coordinate="Model")}
    config["visualization"] = {"columns": ["attributions", "predictions", "input_ids", "labels"],
                               "gamma": 2.0,
                               "normalize": True}

    run_visualize(config=config, dataset=thermostat_dataset)
