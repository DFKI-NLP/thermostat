from datasets import load_dataset

# This will use the dataset script ("thermostat.py") in the "thermostat" directory
#lgxa = load_dataset("thermostat", "imdb-bert-lgxa", split="test")
lig = load_dataset("thermostat", "imdb-bert-lig", split="test[:1%]")

print(lig[0])
