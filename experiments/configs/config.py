"""Common configuration for experiments."""

import os

# Machine settings
n_concurrent_designs = None
n_processes = 10

# Directories and file patterns
dataset_dir = os.path.join('data','processed')
solar_fname_pattern = 'sy_{year}.csv'
building_fname_pattern = 'ly_{id}-{year}.csv'
results_dir = os.path.join('results')

# Available years and building ids
solar_years = list(range(2010, 2020))
load_years = list(range(2012, 2018))
ids = [0, 4, 8, 19, 25, 40, 58, 102, 104] # 118

# Simulation parameters
num_reduced_scenarios = 10 # no. of reduced scenarios used in Bryn's thesis
n_post_samples = 256 # determined from MC convergence plots

# Solar constraint
solar_constraint = 150.0

# Cost parameters
cost_dict = {
    'carbon': 1.0, # $/kgCO2
    'battery': 750.0, # $/kWh
    'solar': 1500.0, # $/kWp
    'grid_capacity': 25e-2/0.95, # $/kW/day - note, this is waaay more expensive that current
    'grid_excess': 100e-2/0.95, # $/kW/day - note, this is a waaay bigger penalty than current
    'opex_factor': 20,
    'battery_power_ratio': 0.4, # kW/kWh
    'grid_con_safety_factor': 1.25, # safety factor for grid connection capacity
    'cntrl_grid_cap_margin': 0.01 # margin for grid capacity in control scheme to prevent drift
}

# Probability model parameters
prob_config = {
    'ids': ids,
    'solar_years': solar_years,
    'load_years': load_years,
    'mean_load_mean': 100.0,
    'mean_load_std': 25.0,
    'mean_load_msr_error': 0.1,
    'peak_load_min': 200.0,
    'peak_load_max': 400.0,
    'peak_load_msr_error': 0.075,
    'thin_factor': 10
}