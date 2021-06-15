from collections import defaultdict
from datasets import Dataset
from overrides import overrides
from spacy import displacy
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


class Thermopack(Dataset):
    def __init__(self, hf_dataset):
        super().__init__(hf_dataset.data, info=hf_dataset.info, split=hf_dataset.split)
        self.dataset = hf_dataset

        # Model
        self.model_name = get_coordinate(hf_dataset, 'Model')
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Dataset
        self.dataset_name = get_coordinate(hf_dataset, 'Dataset')
        self.label_names = hf_dataset.info.features['label'].names

        # Explainer
        self.explainer_name = get_coordinate(hf_dataset, 'Explainer')

        # Create Thermounits for every instance
        self.units = []
        for instance in hf_dataset:
            # Decode labels and predictions
            true_label_index = instance['label']
            true_label = {'index': true_label_index,
                          'name': self.label_names[true_label_index]}

            predicted_label_index = instance['predictions'].index(max(instance['predictions']))
            predicted_label = {'index': predicted_label_index,
                               'name': self.label_names[predicted_label_index]}

            self.units.append(Thermounit(
                instance, true_label, predicted_label,
                self.model_name, self.dataset_name, self.explainer_name, self.tokenizer))

    @overrides
    def __getitem__(self, idx):
        """ Indexing a Thermopack returns a Thermounit """
        return self.units[idx]

    @overrides
    def __str__(self):
        return self.info.description


class Thermounit:
    """ Processed single instance of a Thermopack (Thermostat dataset/configuration) """
    def __init__(self, instance, true_label, predicted_label, model_name, dataset_name, explainer_name, tokenizer):
        self.instance = instance
        self.index = self.instance['idx']
        self.true_label = true_label
        self.predicted_label = predicted_label

        self.model_name = model_name
        self.dataset_name = dataset_name
        self.explainer_name = explainer_name
        self.tokenizer = tokenizer

        # "tokens" includes all special tokens, later used for the heatmap when aligning with attributions
        self.tokens = self.tokenizer.convert_ids_to_tokens(self.instance['input_ids'])
        # Cleaned text
        # Note: decode + clean_up_tokenization_spaces did not remove the "##" artifacts
        self.text = self.tokenizer.decode(token_ids=self.instance['input_ids'], clean_up_tokenization_spaces=True,
                                          skip_special_tokens=True)

        self.heatmap = None
        self.set_heatmap(flip_attributions_idx=0)

    def __str__(self):
        """ String representation is the cleaned text """
        return self.text

    def __len__(self):
        """ Number of non-special tokens """
        return len([t for t in self.tokens if t not in self.tokenizer.all_special_tokens])

    def set_heatmap(self, gamma=2.0, normalize=True, flip_attributions_idx=None, drop_special_tokens=True,
                    fuse_subword_tokens=True):
        """ Generate a list of tuples in the form of <token,color> for a single data point of a Thermostat dataset """

        atts = zero_special_tokens(self.instance['attributions'],
                                   self.instance['input_ids'],
                                   self.tokenizer)
        if normalize:
            atts = normalize_attributions(atts)

        if flip_attributions_idx == self.predicted_label['index']:
            atts = [att * -1 for att in atts]

        if drop_special_tokens:
            special_tokens_indices = [i for i, w in enumerate(self.tokens) if w in self.tokenizer.all_special_tokens]
            tokens = [i for w, i in enumerate(self.tokens) if w not in special_tokens_indices]
            atts = [i for a, i in enumerate(atts) if a not in special_tokens_indices]
        else:
            tokens = self.tokens

        if fuse_subword_tokens:
            fused_tokens = []
            fused_atts = []
            _token = ""
            _att = 0
            _counter = 0
            for token, att in zip(tokens, atts):
                if not token.startswith('##'):
                    if len(_token) > 0 and _counter > 0:  # Previous token finished
                        fused_tokens.append(_token)
                        fused_atts.append(_att/_counter)
                    _token = token
                    _att = att
                    _counter = 0
                else:
                    _token += token.replace('##', '')
                    _att += att
                    _counter += 1

        sequence = Sequence(words=tokens, scores=atts)
        self.heatmap = sequence.words_rgb(token_pad=self.tokenizer.pad_token,
                                          position_pad=self.tokenizer.padding_side,
                                          gamma=gamma)

    def render(self, attribution_labels=False):
        """ """  # TODO

        ents = []
        colors = {}
        ii = 0
        for token_rgb in self.heatmap:
            token, rgb = token_rgb.values()
            att_rounded = str(rgb.score)

            ff = ii + len(token)

            # One entity in displaCy contains start and end markers (character index) and optionally a label
            # The label can be added by setting "attribution_labels" to True
            ent = {
                'start': ii,
                'end': ff,
                'label': att_rounded,
            }

            ents.append(ent)
            # A "colors" dict takes care of the mapping between attribution labels and hex colors
            colors[att_rounded] = rgb.hex
            ii = ff

        to_render = {
            'text': ''.join([t['token'] for t in self.heatmap]),
            'ents': ents,
        }

        if attribution_labels:
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
            <mark class="entity" style="background: {bg}; padding: 0.3em 0.45em; margin: 0 0.25em; line-height: 2.25;
            border-radius: 0.25em; box-decoration-break: clone; -webkit-box-decoration-break: clone">
                {text}
            </mark>
            """

        html = displacy.render(
            to_render,
            style='ent',
            manual=True,
            jupyter=False,
            options={'template': template,
                     'colors': colors,
                     }
        )
        return html


def to_html(thermostat_dataset: Dataset, out_html: str, gamma=1.0):
    """ Run the visualization script on a Thermostat dataset
        FIXME: soon to be deprecated. """
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
