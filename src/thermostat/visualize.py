import json
import linecache
import math
import numpy as np
import os
from datasets import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from typing import Dict, List

from thermostat.data import get_local_explanations
from thermostat.utils import detach_to_list, Configurable


class RGB:
    def __init__(self, red, green, blue, score):
        self.red = red
        self.green = green
        self.blue = blue
        self.score = round(score, ndigits=3) if score is not None else score

    def __str__(self):
        return 'rgb({},{},{})'.format(self.red, self.green, self.blue)


class Sequence:
    def __init__(self, words, scores):
        assert (len(words) == len(scores))
        self.words = words
        self.scores = scores
        self.size = len(words)

    def words_rgb(self, gamma=1.0, token_pad=None, position_pad='right'):
        rgbs = list(map(lambda tup: self.rgb(word=tup[0], score=tup[1], gamma=gamma), zip(self.words, self.scores)))
        if token_pad is not None:
            if token_pad in self.words:
                if position_pad == 'right':
                    return zip(self.words[:self.words.index(token_pad)], rgbs)
                elif position_pad == 'left':
                    first_token_index = list(reversed(self.words)).index(token_pad)
                    return zip(self.words[-first_token_index:], rgbs[-first_token_index:])
                else:
                    return NotImplementedError
        return zip(self.words, rgbs)

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


def token_to_html(token, rgb):
    return f"<span style=\"background-color: {rgb}\"> {token.replace('<', '').replace('>', '')} </span>"


def append_heatmap(tokens, scores, latex, gamma, caption, pad_token, formatting="colorbox", truncate_pad=True):
    """Format options: colorbox, text"""
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


class ExplanationConfig(Configurable):
    def __init__(self):
        self.name = None
        self.flip_attributions_on_index = None

    def validate_config(self, config: Dict) -> bool:
        assert 'name' in config, 'Please provide an identifier, a name, which can be used to retrieve the saliencies.'
        assert 'flip_attributions_on_index' in config, 'flip_attributions_on_index is required. ' \
                                                       'Set to <0 if flip shall not be performed.'


def write_html(path_in: str,
               path_out: str,
               explanation_configs: List[ExplanationConfig],
               name_input: str,
               labels_list: List[str],
               dataset: str,
               name_model: str,
               mneumonic_method_names: Dict[str, str],
               line_index_start: int = -1,
               line_index_end: int = -1,
               gamma: float = 1.0,
               normalize: bool = False,
               token_pad: str = '[PAD]',
               position_pad: str = 'back',
               debug_mode: bool = False):
    if line_index_start >= 0:
        assert line_index_start < line_index_end, 'Start line index should be smaller than end line index'
        if line_index_start == 0:
            line_index_start = 1
    else:
        line_index_start = 1
        line_index_end = len(open(path_in, 'r+').readlines())
    if gamma != 1.0:
        raise NotImplementedError
    if os.path.isfile(path_out):
        raise FileExistsError
    fout = open(path_out, 'w+')
    for line_idx in tqdm(range(line_index_start, line_index_end + 1)):
        line = linecache.getline(path_in, line_idx)
        jsn = json.loads(line)
        tokens: List[str] = jsn['inputs'][name_input]['tokens']
        label = jsn['data']['label']
        id = jsn['id']
        html = f"<html><h3>"
        html += f" Dataset: {dataset} | "
        html += f" Instance: {line_idx} | "
        if debug_mode:
            html += f" Instance Id: {id} | "
        html += f" Gold Label: {labels_list[label]} | "
        html += f" Model: {name_model} | "
        prediction_set = False
        for idx, explanation_config in enumerate(explanation_configs):
            saliencies: List[float] = jsn['explanation'][explanation_config.name]['attribution']
            if normalize:
                max_abs_score = max(max(saliencies), abs(min(saliencies)))
                saliencies = [(score / max_abs_score) for score in saliencies]
            prediction = jsn['explanation'][explanation_config.name]['prediction']

            if not prediction_set:
                # todo: this assumes that the first explanation contains the predictions,
                #  which is not the case for empirical explanations which do not contain predictions
                assert idx == 0, 'Sanity check failed'
                html += f' Predicted Label: {labels_list[prediction.index(max(prediction))]} |'
                html += f' Logits: {[round(pred, 3) for pred in prediction]} '
                html += '</h3><div style=\"border:3px solid #000;\">'
                prediction_set = True

            flip_attributions = False
            if explanation_config.flip_attributions_on_index >= 0:
                if prediction.index(max(prediction)) == explanation_config.flip_attributions_on_index:
                    saliencies = [score * -1 for score in saliencies]
                    flip_attributions = True
            # todo: normalize?
            sequence = Sequence(words=tokens, scores=saliencies)
            words_rgb = sequence.words_rgb(token_pad=token_pad, gamma=gamma,
                                           position_pad=position_pad)  # todo: <pad> is xlnet pad token
            headline = f'</p><h4> ' \
                       f' {mneumonic_method_names[explanation_config.name]} '
            if debug_mode:
                headline += f' Flipped: {flip_attributions} </h4></p>'
            else:
                headline += '</h4></p>'
            html += headline
            html += " "
            for word, rgb in words_rgb:
                html += token_to_html(word, rgb)
            html += "</br></br>"
        html += "</div></br></br></br></html>"
        fout.write(html + os.linesep)
    fout.close()


def run_visualize(config: Dict):
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])

    visualization_config = config['visualization']
    dataset = get_local_explanations(config=visualization_config)
    dataloader = DataLoader(dataset=dataset, batch_size=1)

    file_out = open(config['path_html'], 'w+')
    for idx_instance, instance in enumerate(tqdm(dataloader)):
        tokens = [tokenizer.decode(token_ids=token_id) for token_id in instance['input_ids'][0]]
        atts = detach_to_list(instance['attributions'][0])
        # TODO: normalize https://github.com/rbtsbg/gxai/blob/a9df8ab6cdc453513bfd4193585ed37160ed8224/visualize.py#L145
        atts_sum = sum(atts)
        atts = [att/atts_sum for att in atts]
        sequence = Sequence(words=tokens, scores=atts)
        html = f"<html><h3>"
        html += f" Label: {instance['labels'].item()} | "
        html += f" Prediction: {detach_to_list(instance['predictions'][0])}"
        html += f" Sum o. Atts: {atts_sum}"
        html += f" Model: {config['model']['name']} | "

        # TODO: Allow token_pad and position_pad to be set by the config
        #  (tokenizer should determine this in most cases, though)
        words_rgb = sequence.words_rgb(token_pad=tokenizer.pad_token,
                                       position_pad=tokenizer.padding_side,
                                       gamma=visualization_config['gamma'])

        html += '</h3><div style=\"border:3px solid #000;\">'
        for word, rgb in words_rgb:
            html += token_to_html(word, rgb)
        html += "</br></br>"
        html += "</div></br></br></br></html>"
        file_out.write(html + os.linesep)
