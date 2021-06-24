import thermostat

bert = thermostat.load("multi_nli-bert-occ")
electra = thermostat.load("multi_nli-electra-occ")

for model_name, data in zip(["bert", "electra"], [bert, electra]):
    print(model_name)
    data.classification_report()
    print('=====================\n\n')

disagreement = [(b, e) for (b, e) in zip(bert, electra) if b.predicted_label != e.predicted_label]

# good examples in disagreement: 16
contradictions = [(i, unit)[0] for i, unit in enumerate(disagreement) if unit[0].true_label == 'contradiction']

index = contradictions[16]
u_b, u_e = disagreement[index]

print(u_b.predicted_label, u_b.true_label)
u_b.render()
print(u_e.predicted_label, u_e.true_label)
u_e.render()
