import math
import numpy as np
import os
import pandas as pd
import torch
from tqdm import tqdm
from spacy import displacy
from spacy.util import is_in_jupyter
from transformers import AutoTokenizer
from typing import Dict

from thermostat.data import get_local_explanations
from thermostat.utils import delistify, detach_to_list, read_path


class ColorToken:
    def __init__(self, token, attribution, text_field, token_index, thermounit_vars: Dict):
        self.token = token
        self.attribution = attribution
        self.text_field = text_field
        self.token_index = token_index
        for var_name, value in thermounit_vars.items():
            if var_name not in ['texts']:
                setattr(self, var_name, value)

        # White color per default
        self.red = '255'
        self.green = '255'
        self.blue = '255'
        self.score = None

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
            score_str = f'Score: {self.score}'
        else:
            score_str = f'Attribution: {self.attribution}'
        return f'{self.token} (Index: {self.token_index}, {score_str}, Color: {self.hex()}, ' \
               f'Text field: {self.text_field})'

    def __str__(self):
        return repr(self)

    @staticmethod
    def gamma_correction(score, gamma):
        return np.sign(score) * np.power(np.abs(score), gamma)

    def hex(self):
        return '#%02x%02x%02x' % (int(self.red), int(self.green), int(self.blue))


class TextField(list):
    def __init__(self, color_tokens):
        super().__init__(color_tokens)

    def __repr__(self):
        return ' '.join([ctok.token for ctok in self])


class Heatmap(TextField):
    def __init__(self, color_tokens, attributions=None, gamma=1.0):
        super().__init__(color_tokens)
        for i in range(len(self)):
            if attributions:
                self[i].attribution = attributions[i]
            self[i].add_color(gamma=gamma)
        self.table = pd.DataFrame({
            'token_index': delistify(self['token_index']),
            'token': delistify(self['token']),
            'attribution': delistify(self['attribution']),
            'text_field': delistify(self['text_field'])}
        ).set_index('token_index').T

    def __getitem__(self, idx):
        if isinstance(idx, str):
            """ String indexing """
            if idx in list(self[0].__dict__.keys()):
                return [getattr(u, idx) for u in self]
        return list(self)[idx]

    def __repr__(self):
        return repr(self.table)

    def render(self, labels=False):
        """ Uses the displaCy visualization tool to render a HTML from the heatmap """

        # Call this function once for every text field
        if len(set([t.text_field for t in self])) > 1:
            for field in self[0].text_fields:
                print(f'Heatmap "{field}"')
                Heatmap([t for t in self if t.text_field == field]).render(labels=labels)
            return

        ents = []
        colors = {}
        ii = 0
        for color_token in self:
            ff = ii + len(color_token.token)

            # One entity in displaCy contains start and end markers (character index) and optionally a label
            # The label can be added by setting "attribution_labels" to True
            ent = {
                'start': ii,
                'end': ff,
                'label': str(color_token.score),
            }

            ents.append(ent)
            # A "colors" dict takes care of the mapping between attribution labels and hex colors
            colors[str(color_token.score)] = color_token.hex()
            ii = ff

        to_render = {
            'text': ''.join([t.token for t in self]),
            'ents': ents,
        }

        if labels:
            template = """
            <mark class="entity" style="background: {bg}; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 2; 
            border-radius: 0.35em; box-decoration-break: clone; -webkit-box-decoration-break: clone">
                {text}
                <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: 
                uppercase; vertical-align: middle; margin-left: 0.5rem">{label}</span>
            </mark>
            """
        else:
            template = """
            <mark class="entity" style="background: {bg}; padding: 0.15em 0.3em; margin: 0 0.2em; line-height: 2.2;
            border-radius: 0.25em; box-decoration-break: clone; -webkit-box-decoration-break: clone">
                {text}
            </mark>
            """

        html = displacy.render(
            to_render,
            style='ent',
            manual=True,
            jupyter=is_in_jupyter(),
            options={'template': template,
                     'colors': colors,
                     }
        )
        return html if not is_in_jupyter() else None


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
