"""References:
https://colab.research.google.com/drive/1pgAbzUF2SzF0BdFtGpJbZPWUOhFxT2NZ
"""

import torch

from captum.attr import LayerIntegratedGradients
from captum.attr import visualization as viz
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# Load pre-trained model fine-tuned on IMDb
model_id = 'textattack/roberta-base-imdb'
model = AutoModel.from_pretrained(model_id)
model.to(device)
model.eval()
model.zero_grad()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load dataset
data = load_dataset('imdb', split='test[:5%]')

text = data[0]['text']
batch_encoding = tokenizer(text, padding='max_length')


text_ids = batch_encoding['input_ids']

# construct input token ids
input_ids = torch.tensor([text_ids]).to(device)


def predict(inputs):
    return model(inputs)[0]


def binary_sentiment_fwd(inputs, target_label=0):
    """ target_label=0 for negative attribution,
        target_label=1 for positive attribution """
    preds = predict(inputs)
    return torch.softmax(preds, dim=1)[:, target_label]


lig = LayerIntegratedGradients(binary_sentiment_fwd, model.embeddings)

attributions_ig, delta = lig.attribute(inputs=input_ids,
                                       n_steps=50,
                                       internal_batch_size=1,
                                       target=0,
                                       return_convergence_delta=True)

attributions_sum = attributions_ig.sum(dim=-1).squeeze(0)
attributions_sum_norm = attributions_sum / torch.norm(attributions_sum)

pred = binary_sentiment_fwd(input_ids)
pred_prob = pred.max(0).values.squeeze(0).mean(0)
pred_class = pred.max(0).indices

indices = input_ids[0].detach().tolist()
all_tokens = tokenizer.convert_ids_to_tokens(indices)

score_vis = viz.VisualizationDataRecord(attributions_sum,
                                        pred_prob,
                                        pred_class,
                                        1,
                                        text,
                                        attributions_sum.sum(),
                                        all_tokens,
                                        delta)

viz.visualize_text([score_vis])
