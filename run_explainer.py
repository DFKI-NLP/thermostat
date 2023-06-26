import argparse
from thermostat.explain import explain_custom_data


# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-c', help='Config file', default='configs/imdb/bert/lig.jsonnet')
parser.add_argument('-home', help='Home directory', default=None)
args = parser.parse_args()
config_file = args.c
home_dir = args.home
explain_custom_data(config_file, home_dir)
