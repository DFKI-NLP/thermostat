import json
import logging
import os

from thermostat.utils import get_logger, read_config
from thermostat.visualize import run_visualize


# Log config
logger = get_logger(name='vis', file_out='./vis.log', level=logging.INFO)

# Config handling
config_file = 'configs/sst2/GradientXActivation_bert.jsonnet'
config = read_config(config_file)
logger.info(f'(Config) Config: \n{json.dumps(config, indent=2)}')


# Choose latest created file in experiment path
experiment_path = config['experiment_path']
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

logger.info("(Progress) Generating visualizations")
run_visualize(config=config)
