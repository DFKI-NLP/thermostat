import thermostat

# This will use the dataset script ("thermostat.py") in the "thermostat" directory
#data = thermostat.load("imdb-bert-lig")

xlnet = thermostat.load("multi_nli-xlnet-occ")
roberta = thermostat.load("multi_nli-roberta-occ")
bert = thermostat.load("multi_nli-bert-occ")
#albert = thermostat.load("multi_nli-albert-occ")

xlnet_hm = xlnet[0].heatmap
roberta_hm = roberta[0].heatmap
bert_hm = bert[0].heatmap

bert[0].render()
roberta[0].render()
