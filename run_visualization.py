import os

from thermostat.utils import read_config, read_path
from thermostat.visualize import run_visualize


config = read_config('configs/exp-a01_imdb_LayerIntegratedGradients_textattack-roberta-base-imdb.jsonnet')

# Choose latest created file in experiment path
experiment_path = read_path(config['path'])
expath_files = sorted([os.path.join(experiment_path, file)
                       for file in os.listdir(experiment_path)],
                      key=os.path.getctime)
path_in = [f for f in expath_files if os.path.isfile(f)][-1]
config['visualization']['path_in'] = path_in  # Add filename to vis config

# Create visualization dir in experiment path if it does not exist already
vis_dir = os.path.join(experiment_path, 'vis')
if not os.path.isdir(vis_dir):
    os.makedirs(vis_dir)
    print(f'Created {vis_dir}')

# Add HTML output file to config
config['path_html'] = os.path.join(vis_dir, f'{path_in.split("/")[-1]}.html')

run_visualize(config=config)
