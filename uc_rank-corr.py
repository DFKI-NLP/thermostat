import thermostat
from scipy.stats import kendalltau

imdb_lime = thermostat.load("imdb-bert-lime", cache_dir='/home/nfel/datasets')
il_atts = imdb_lime.attributions.flatten()
imdb_intg = thermostat.load("imdb-bert-lig")
ii_atts = imdb_intg.attributions.flatten()
kendall_imdb = kendalltau(il_atts, ii_atts)
print(kendall_imdb)

mnli_lime = thermostat.load("multi_nli-bert-lime")
ml_atts = mnli_lime.attributions.flatten()
mnli_intg = thermostat.load("multi_nli-bert-lig")
mi_atts = mnli_intg.attributions.flatten()
kendall_mnli = kendalltau(ml_atts, mi_atts)
print(kendall_mnli)
