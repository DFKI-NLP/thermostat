from datasets import load_dataset

from thermostat import Thermopack

# This will use the dataset script ("thermostat.py") in the "thermostat" directory
lgxa = load_dataset("thermostat", "imdb-bert-lgxa", split="test")
#lig = load_dataset("thermostat", "imdb-bert-lig", split="test[:1%]")
#agnews = load_dataset("thermostat", "ag_news-albert-lgxa", split="test[:2%]")

lgxa_head = lgxa.select(range(20, 40))

tp = Thermopack(lgxa_head)
tu = tp[0]
html = tu.render()
print(tp)
