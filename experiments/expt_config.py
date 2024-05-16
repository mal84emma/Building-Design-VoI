"""Common configuration for experiments."""

import os

# Directorioes and file patterns
save_dir = os.path.join('experiments','results')
dataset_dir = os.path.join('data','processed')
building_fname_pattern = 'ly_{id}-{year}.csv'

# Available years and building ids
years = list(range(2012, 2018))
ids = [0, 4, 8, 19, 25, 40, 58, 102, 104] # 118

# Simulation parameters
n_buildings = 8
num_reduced_scenarios = 10 # no. of reduced scenarios used in Bryn's thesis
n_post_samples = 320 # determined from MC convergence plots

# Cost parameters
cost_dict = {
    'carbon': 1.0, #5e-1, # $/kgCO2
    'battery': 1e3, #1e3, # $/kWh
    'solar': 1e3, #2e3, # $/kWp
    'grid_capacity': 25e-2/0.95, # $/kW/day - note, this is waaay more expensive that current
    'grid_excess': 100e-2/0.95, # $/kW/day - note, this is a waaay bigger penalty than current
    'opex_factor': 20,
    'battery_power_ratio': 0.4
}