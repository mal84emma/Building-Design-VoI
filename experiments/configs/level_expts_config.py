"""Configuration settings for shape uncertainty experiments."""

import os
from experiments.configs.general_config import prob_config

results_dir = os.path.join('experiments','results','level')
n_buildings = 1

prob_config.update({
    'mean_load_mean': 100.0,
    'mean_load_std': 25.0,
    'mean_load_msr_error': 0.1,
    'peak_load_min': 200.0,
    'peak_load_max': 400.0,
    'peak_load_msr_error': 0.075,
    'thin_factor': 10
})