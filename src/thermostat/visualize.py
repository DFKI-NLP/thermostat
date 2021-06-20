import math
import numpy as np
import os
import torch
from datasets import tqdm
from transformers import AutoTokenizer
from typing import Dict

from thermostat.data import get_local_explanations
from thermostat.utils import detach_to_list, read_path


class ColorToken:
    def __init__(self, token, attribution, text_field, token_index, thermounit_vars: Dict):
        self.token = token
        self.attribution = attribution
        self.text_field = text_field
        self.token_index = token_index
        for var_name, value in thermounit_vars.items():
            setattr(self, var_name, value)

        # White color per default
        self.red = '255'
        self.green = '255'
        self.blue = '255'

        assert not math.isnan(self.attribution), 'Attribution of token {} is NaN'.format(self.token)

    def add_color(self, gamma, threshold=0):
        """ Needs to be explicitly called to calculate the color of a token """
        score = self.gamma_correction(self.attribution, gamma)
        if score >= threshold:
            r = str(int(255))
            g = str(int(255 * (1 - score)))
            b = str(int(255 * (1 - score)))
        else:
            b = str(int(255))
            r = str(int(255 * (1 + score)))
            g = str(int(255 * (1 + score)))

        # TODO: Add more color schemes from: https://colorbrewer2.org/#type=diverging&scheme=RdBu&n=5
        self.red = r
        self.green = g
        self.blue = b
        setattr(self, 'score', round(score, ndigits=3))

    def __repr__(self):
        if "score" in vars(self):
            return f'{self.token} (Score: {self.score}, Color: {self.hex()}, Text field: {self.text_field})'
        else:
            return f'{self.token} (Attribution: {self.attribution}, Color: {self.hex()}, Text field: {self.text_field})'

    @staticmethod
    def gamma_correction(score, gamma):
        return np.sign(score) * np.power(np.abs(score), gamma)

    def hex(self):
        return '#%02x%02x%02x' % (int(self.red), int(self.green), int(self.blue))


class Heatmap(list):
    def __init__(self, color_tokens, gamma=1.0):
        super().__init__(color_tokens)
        for ctoken in self:
            ctoken.add_color(gamma=gamma)

    def __repr__(self):
        return '\n'.join([str(ctok) for ctok in self])


def token_to_html(token, rgb):
    return f"<span style=\"background-color: {rgb}\"> {token.replace('<', '').replace('>', '')} </span>"


def summarize(summary: Dict):
    res = "<h4>"
    for k, v in summary.items():
        res += f"{k}: {summary[k]} <br/>"
    res += "</h4>"
    return res


def append_heatmap(tokens, scores, latex, gamma, caption, pad_token, formatting="colorbox", truncate_pad=True):
    """
    Produce a heatmap for LaTeX
    Format options: colorbox, text"""
    if gamma != 1:
        raise NotImplementedError
    latex += "\n\\begin{figure}[!htb]"
    for token, score in zip(tokens, scores):
        if token == pad_token and truncate_pad:
            continue
        color = "blue"
        if score >= 0:
            color = "red"
        latex += f"\\{formatting}" + "{" + f"{color}!{abs(score) * 100}" + "}" + "{" + token + "}"
    latex += "\\caption{" + f"{caption}" + "}"
    latex += "\\end{figure}\n"
    return latex


def normalize_attributions(attributions):
    max_abs_score = max(max(attributions), abs(min(attributions)))
    return [(score / max_abs_score) for score in attributions]


def run_visualize(config: Dict, dataset=None):
    raise NotImplementedError("Deprecated due to Heatmap and ColorToken refactoring")
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    visualization_config = config['visualization']

    if not dataset:
        dataset = get_local_explanations(config=visualization_config)
    dataset_name = f'{config["dataset"]["name"]}' \
                   f': {config["dataset"]["subset"]}' if 'subset' in config['dataset'] else \
        config['dataset']['name']
    str_dataset_name = f'{dataset_name} ({config["dataset"]["split"]})'

    file_out = open(read_path(config['path_html']), 'w+')

    tokenizer_str = str(type(tokenizer)).split('.')[-1].strip("'>")
    for idx_instance in tqdm(range(len(dataset))):
        instance = dataset[idx_instance]

        html = f"<html><h3>"
        html += f"<h2>Instance: {instance['idx']} | Dataset: {str_dataset_name} |" \
                f" Model: {config['model']['name']} | Tokenizer: {tokenizer_str}"
        html += '</h3><div style=\"border:3px solid #000;\">'

        html += "<div>"

        tokens = [tokenizer.decode(token_ids=token_ids) for token_ids in instance['input_ids']]
        atts = detach_to_list(instance['attributions'])

        if visualization_config['normalize']:
            atts = normalize_attributions(atts)

        heatmap = Heatmap(words=tokens, scores=atts, gamma=visualization_config['gamma'])

        summary = {'Sum of Attribution Scores': str(sum(atts))}

        if 'dataset' in dataset:
            label_names = dataset['dataset'][0]['label_names']
        else:
            label_names = dataset.info.features['label'].names
        if 'labels' in instance or 'label' in instance:
            if 'labels' in instance:
                label = detach_to_list(instance['labels'])
            else:
                label = instance['label']
            summary['True Label Index'] = str(label)
            summary['True Label'] = str(label_names[label])
        if 'predictions' in instance:
            preds = instance['predictions']
            summary['Logits'] = detach_to_list(preds)
            preds_max = torch.argmax(preds) if type(preds) == torch.Tensor else preds.index(max(preds))
            preds_max_detached = detach_to_list(preds_max)
            summary['Predicted Label'] = str(label_names[preds_max_detached])
        html += summarize(summary)

        for instance in heatmap:  # brackets to reuse iterator
            html += token_to_html(instance['token'], instance['color'])
        html += "</br></br>"
        html += "</div>"
        html += "</div></br></br></br></html>"
        file_out.write(html + os.linesep)
