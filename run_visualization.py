import json
import logging
import os

from thermostat.utils import get_logger, read_config, read_path
from thermostat.visualize import run_visualize


config = read_config('configs/imdb_LimeBase_bert-base-cased-ignite-imdb.jsonnet')

# Choose latest created file in experiment path
experiment_path = read_path(config['path'])

expath_files = sorted([os.path.join(experiment_path, file)
                       for file in os.listdir(experiment_path)],
                      key=os.path.getctime)
path_in = [f for f in expath_files if os.path.isfile(f)][-1]
config['visualization']['path_explanations'] = path_in  # Add filename to vis config

# Create visualization dir in experiment path if it does not exist already
vis_dir = os.path.join(experiment_path, 'vis')
if not os.path.isdir(vis_dir):
    os.makedirs(vis_dir)
    print(f'Created {vis_dir}')

# Add HTML output file to config
config['path_html'] = os.path.join(vis_dir, f'{path_in.split("/")[-1]}.html')

# Log config
logger = get_logger(name='vis', file_out='./vis.log', level=logging.INFO)
logger.info(f'(Config) Config: \n{json.dumps(config, indent=2)}')

run_visualize(config=config, logger=logger)
