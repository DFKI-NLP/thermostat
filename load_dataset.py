
from datasets import load_dataset

from thermostat.data.dataset_utils import get_heatmap
from thermostat import to_html

# This will use the dataset script ("thermostat.py") in the "thermostat" directory
lgxa = load_dataset("thermostat", "imdb-bert-lgxa", split="test")
lgxa_head = lgxa.select(range(20, 40))
#to_html(lgxa_head, "/home/nfel/datasets/imdb-bert-lgxa.html")

#lig = load_dataset("thermostat", "imdb-bert-lig", split="test[:1%]")
#agnews = load_dataset("thermostat", "ag_news-albert-lgxa", split="test[:2%]")

hm = get_heatmap(lgxa_head)
print(hm)
