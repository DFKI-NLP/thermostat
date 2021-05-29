from collections import defaultdict
from datasets import Dataset
from transformers import AutoTokenizer
from typing import List

from thermostat.visualize import Sequence, normalize_attributions, run_visualize, zero_special_tokens


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


def get_heatmap(thermostat_dataset: Dataset) -> List:
    """ Generate a list of tuples in the form of <token,color> for each data point of a Thermostat dataset """
    model_id = get_coordinate(thermostat_dataset, coordinate='Model')
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    heatmap = []
    for instance in thermostat_dataset:
        atts = zero_special_tokens(instance['attributions'], instance['input_ids'], tokenizer)
        atts = normalize_attributions(atts)
        tokens = [tokenizer.decode(token_ids=token_ids) for token_ids in instance['input_ids']]

        sequence = Sequence(words=tokens, scores=atts)
        words_rgbs = sequence.words_rgb(token_pad=tokenizer.pad_token,
                                        position_pad=tokenizer.padding_side,
                                        gamma=2.0)
        instance_heatmap = []
        for word, rgb in words_rgbs:
            instance_heatmap.append((word, rgb.hex()))
        heatmap.append(instance_heatmap)
    return heatmap


# TODO: Wrappable function
def _to_html(dataset, model_name, path_out, gamma, normalize):
    raise NotImplementedError


def to_html(thermostat_dataset: Dataset, out_html: str, gamma=1.0):
    """ Run the visualization script on a Thermostat dataset """
    # TODO: Pass filehandler and check if valid

    config = dict()
    config["path_html"] = out_html
    config["dataset"] = {"name": get_coordinate(thermostat_dataset, coordinate="Dataset"),
                         "split": "test",  # TODO: Check if hard-coding this makes sense
                         }
    config["model"] = {"name": get_coordinate(thermostat_dataset, coordinate="Model")}
    config["visualization"] = {"columns": ["attributions", "predictions", "input_ids", "labels"],
                               "gamma": gamma,
                               "normalize": True}

    run_visualize(config=config, dataset=thermostat_dataset)


def avg_attribution_stat(thermostat_dataset: Dataset) -> List:
    """ Given a Thermostat dataset, calculate the average attribution for each token across the whole dataset """
    model_id = get_coordinate(thermostat_dataset, coordinate='Model')
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    token_atts = defaultdict(list)
    for row in thermostat_dataset:
        for input_id, attribution_score in zip(row['input_ids'], row['attributions']):
            # Distinguish between the labels
            if row['label'] == 0:
                # Add the negative attribution score for label 0
                # to the list of attribution scores of a single token
                token_atts[tokenizer.decode(input_id)].append(-attribution_score)
            else:
                token_atts[tokenizer.decode(input_id)].append(attribution_score)

    avgs = defaultdict(float)
    # Calculate the average attribution score from the list of attribution scores of each token
    for token, scores in token_atts.items():
        avgs[token] = sum(scores)/len(scores)
    return sorted(avgs.items(), key=lambda x: x[1], reverse=True)


def explainer_agreement_stat(thermostat_datasets: List) -> List:
    """ Calculate agreement on token attribution scores between multiple Thermostat datasets/explainers """
    assert len(thermostat_datasets) > 1
    all_explainers_atts = {}
    for td in thermostat_datasets:
        assert type(td) == Dataset
        explainer_id = get_coordinate(td, coordinate='Explainer')
        # Add all attribution scores to a dictionary with the key being the name of the explainer
        all_explainers_atts[explainer_id] = td['attributions']

    model_id = get_coordinate(thermostat_datasets[0], coordinate='Model')
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Dissimilarity dict for tokens and their contexts
    tokens_dissim = {}
    for row in zip(thermostat_datasets[0]['input_ids'],
                   *list(all_explainers_atts.values())):
        # Decode all tokens of one data point
        tokens = tokenizer.decode(list(row)[0], skip_special_tokens=True)
        for idx, input_id in enumerate(zip(*list(row))):
            if list(input_id)[0] in tokenizer.all_special_ids:
                continue

            att_explainers = list(input_id)[1:]
            max_att = max(att_explainers)
            min_att = min(att_explainers)

            # Key: All tokens (context), single token in question, index of token in context
            tokens_dissim[(tokenizer.decode(list(input_id)[0]), tokens, idx)]\
                = {'dissim': max_att - min_att,  # Maximum difference in attribution
                   'atts': dict(zip(all_explainers_atts.keys(), att_explainers))}
    return sorted(tokens_dissim.items(), key=lambda x: x[1]['dissim'], reverse=True)

