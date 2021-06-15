import math
import numpy as np
import os
import torch
from datasets import tqdm
from transformers import AutoTokenizer
from typing import Dict, List

from thermostat.data import get_local_explanations
from thermostat.utils import detach_to_list, read_path


class RGB:
    def __init__(self, red, green, blue, score):
        self.red = red
        self.green = green
        self.blue = blue
        self.score = round(score, ndigits=3) if score is not None else score
        self.hex = self.hex()

    def __str__(self):
        return 'rgb({},{},{})'.format(self.red, self.green, self.blue)

    def hex(self):
        return '#%02x%02x%02x' % (int(self.red), int(self.green), int(self.blue))


class Sequence:
    def __init__(self, words, scores):
        assert (len(words) == len(scores))
        self.words = words
        self.scores = scores
        self.size = len(words)

    def words_rgb(self, gamma=1.0, token_pad=None, position_pad='right', return_zip_object=False):
        rgbs = list(map(lambda tup: self.rgb(word=tup[0], score=tup[1], gamma=gamma), zip(self.words, self.scores)))
        words_rgbs = None
        if token_pad is not None:
            if token_pad in self.words:
                if position_pad == 'right':
                    words_rgbs = zip(self.words[:self.words.index(token_pad)], rgbs)
                elif position_pad == 'left':
                    first_token_index = list(reversed(self.words)).index(token_pad)
                    words_rgbs = zip(self.words[-first_token_index:], rgbs[-first_token_index:])
                else:
                    return NotImplementedError('Invalid position_pad value.')
        if not words_rgbs:
            words_rgbs = zip(self.words, rgbs)
        return words_rgbs if return_zip_object else [{'token': word, 'color': rgb}
                                                     for word, rgb in words_rgbs]

    def compute_length_without_pad_tokens(self, special_tokens: List[str]):
        counter = 0
        for word in self.words:
            if word not in special_tokens:
                counter = counter + 1
        return counter

    @staticmethod
    def gamma_correction(score, gamma):
        return np.sign(score) * np.power(np.abs(score), gamma)

    def rgb(self, word, score, gamma, threshold=0):
        assert not math.isnan(score), 'Score of word {} is NaN'.format(word)
        score = self.gamma_correction(score, gamma)
        if score >= threshold:
            r = str(int(255))
            g = str(int(255 * (1 - score)))
            b = str(int(255 * (1 - score)))
        else:
            b = str(int(255))
            r = str(int(255 * (1 + score)))
            g = str(int(255 * (1 + score)))
        return RGB(r, g, b, score)
        # TODO: Add more color schemes from: https://colorbrewer2.org/#type=diverging&scheme=RdBu&n=5


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


def zero_special_tokens(attributions, input_ids, tokenizer):
    atts_special_tokens_zero = []
    for att, inp in zip(attributions, input_ids):
        if inp in tokenizer.all_special_ids:
            atts_special_tokens_zero.append(0.0)
        else:
            atts_special_tokens_zero.append(att)
    return atts_special_tokens_zero


def normalize_attributions(attributions):
    max_abs_score = max(max(attributions), abs(min(attributions)))
    return [(score / max_abs_score) for score in attributions]


def run_visualize(config: Dict, dataset=None):
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

        if 'special_tokens_attribution' not in visualization_config:
            atts = zero_special_tokens(atts, instance['input_ids'], tokenizer)
        if visualization_config['normalize']:
            atts = normalize_attributions(atts)

        sequence = Sequence(words=tokens, scores=atts)
        words_rgb = sequence.words_rgb(token_pad=tokenizer.pad_token,
                                       position_pad=tokenizer.padding_side,
                                       gamma=visualization_config['gamma'],
                                       return_zip_object=True)

        summary = {'Sum of Attribution Scores': str(sum(atts))}
        number_of_non_special_tokens = sequence.compute_length_without_pad_tokens(
            special_tokens=tokenizer.all_special_tokens)
        summary['Non-special tokens'] = number_of_non_special_tokens

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

        for word, rgb in words_rgb:  # brackets to reuse iterator
            html += token_to_html(word, rgb)
        html += "</br></br>"
        html += "</div>"
        html += "</div></br></br></br></html>"
        file_out.write(html + os.linesep)
