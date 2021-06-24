import thermostat

unit_index = 378
u_occ = thermostat.load("multi_nli-bert-occ")[unit_index]
u_intg = thermostat.load("multi_nli-bert-lig")[unit_index]
u_lime = thermostat.load("multi_nli-bert-lime")[unit_index]

print(u_occ.predicted_label, u_occ.true_label)
u_occ.render()
print(u_intg.predicted_label, u_intg.true_label)
u_intg.render()
print(u_lime.predicted_label, u_lime.true_label)
u_lime.render()
