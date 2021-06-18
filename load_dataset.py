import thermostat

# This will use the dataset script ("thermostat.py") in the "thermostat" directory
data = thermostat.load("imdb-bert-lig")

html = data[0].render()
print(html)
