
from datasets import load_dataset

from thermostat.data.dataset_utils import get_heatmap

# This will use the dataset script ("thermostat.py") in the "thermostat" directory
#lgxa = load_dataset("thermostat", "imdb-bert-lgxa", split="test[:1%]")
#lig = load_dataset("thermostat", "imdb-bert-lig", split="test[:1%]")
agnews = load_dataset("thermostat", "ag_news-albert-lgxa", split="test[:2%]")

hm = get_heatmap(lgxa)
#print(hm)
