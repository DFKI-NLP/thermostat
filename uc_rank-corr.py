import numpy as np
import thermostat
from scipy.stats import spearmanr, kendalltau

imdb_lime = thermostat.load("imdb-bert-lime")
imdb_lime = np.array(imdb_lime["attributions"]).flatten()
imdb_lime = imdb_lime / np.linalg.norm(imdb_lime)

imdb_intg = thermostat.load("imdb-bert-lig")
imdb_intg = np.array(imdb_intg["attributions"]).flatten()
imdb_intg = imdb_intg / np.linalg.norm(imdb_intg)

mnli_lime = thermostat.load("multi_nli-bert-lime")
mnli_lime = np.array(mnli_lime["attributions"]).flatten()
mnli_lime = mnli_lime / np.linalg.norm(mnli_lime)

mnli_intg = thermostat.load("multi_nli-bert-lig")
mnli_intg = np.array(mnli_intg["attributions"]).flatten()
mnli_intg = mnli_intg / np.linalg.norm(mnli_intg)

spearman_imdb = spearmanr(imdb_lime, imdb_intg)
kendall_imdb = kendalltau(imdb_lime, imdb_intg)
spearman_mnli = spearmanr(mnli_lime, mnli_intg)
kendall_mnli = kendalltau(mnli_lime, mnli_intg)

print(spearman_imdb, kendall_imdb)
print(spearman_mnli, kendall_mnli)
