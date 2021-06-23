import thermostat

bert = thermostat.load("xnli-bert-occ")
albert = thermostat.load("xnli-albert-occ")
electra = thermostat.load("xnli-electra-occ")
roberta = thermostat.load("xnli-roberta-occ")
xlnet = thermostat.load("xnli-xlnet-occ")

print(xlnet[0])
model_miscls = {}
miscl_indices = []

for model_data in [albert, bert, electra, roberta, xlnet]:
    miscl_indices += [instance.index for instance in model_data if instance.true_label != instance.predicted_label]

miscl_indices = sorted(list(set(miscl_indices)))

for model_data in [albert, bert, electra, roberta, xlnet]:
    model_miscls[model_data.config_name] = [instance for instance in model_data if instance.index in miscl_indices]
    miscl_indices = None
