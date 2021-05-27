
from datasets import load_dataset

from thermostat.data.dataset_utils import explainer_agreement_stat

# This will use the dataset script ("thermostat.py") in the "thermostat" directory
lgxa = load_dataset("thermostat", "imdb-bert-lgxa", split="test[:1%]")
lig = load_dataset("thermostat", "imdb-bert-lig", split="test[:1%]")

#tdis = explainer_agreement_stat([lgxa, lig])

print(lig[0])
