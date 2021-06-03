# compute spearman's rand and kendal's tau of explainer pairs
import numpy as np
from datasets import load_dataset
from scipy.stats import spearmanr, kendalltau
import itertools

explainers = ['imdb-bert-lgxa', 'imdb-bert-lig']

seen = []
for explainer_1, explainer_2 in itertools.product(explainers, explainers):
    if (explainer_1, explainer_2) in seen \
            or (explainer_2, explainer_1) in seen \
            or explainer_1 == explainer_2:
        continue
    attributions_1 = np.array(load_dataset("thermostat", explainer_1, split="test[:100]")['attributions']).flatten()
    attributions_2 = np.array(load_dataset("thermostat", explainer_2, split="test[:100]")['attributions']).flatten()
    coef_s, p_s = spearmanr(attributions_1 / np.linalg.norm(attributions_1),
                            attributions_2 / np.linalg.norm(attributions_2))
    print(f"Spearman Correlation btw. {explainer_1} and {explainer_2}: {coef_s}, p value: {p_s}")
    coef_k, p_k = kendalltau(attributions_1 / np.linalg.norm(attributions_1),
                            attributions_2 / np.linalg.norm(attributions_2))
    print(f"Kendall Tau btw. {explainer_1} and {explainer_2}: {coef_k}, p value: {p_k}")
    seen.append((explainer_1, explainer_2))