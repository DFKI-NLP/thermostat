import thermostat

# This will use the dataset script ("thermostat.py") in the "thermostat" directory
#data = thermostat.load("imdb-bert-lig")

xlnet = thermostat.load("multi_nli-xlnet-occ")
roberta = thermostat.load("multi_nli-roberta-occ")
albert = thermostat.load("multi_nli-albert-occ")

roberta_hm = roberta[0].heatmap
xlnet_hm = xlnet[0].heatmap
albert_hm = albert[0].heatmap

example.render()
