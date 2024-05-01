"""Optimise district energy system for set of sampled scenarios.

Steps:
1. Load data for scenario load profiles.
2. Perform scenario reduction.
3. Construct Stochastic Program.
4. Solve and report result.
"""

import os
import numpy as np
import pandas as pd
import cvxpy as cp

from load_scenario_reduction import reduce_load_scenarios
#from linmodel import ...


# NOTE: see VoI_in_CAS repo for reference and for code to steal


def optimise_system(sampled_scenarios, data_dir, building_file_pattern, costs_dict, solver='HiGHS', num_reduced_scenarios=None):
    # NOTE: scenarios must be vector of Nx2 vectors, (building_id, year) tuple for each building
    # Use num_red_scenarios = None for deterministic cases

    ## Get load profiles for all scenarios (reduces load time)
    building_year_pairs = np.unique(np.concatenate(sampled_scenarios,axis=0),axis=0)
    load_profiles = {
        f'{building_id}-{year}': pd.read_csv(
            os.path.join(data_dir, building_file_pattern.format(id=building_id, year=year)),
            usecols=['Equipment Electric Power [kWh]']
            )['Equipment Electric Power [kWh]'].to_numpy()\
                for (building_id, year) in building_year_pairs
    }

    ## Perform scenario reduction
    if num_reduced_scenarios is not None:
        reduced_scenarios = reduce_load_scenarios(sampled_scenarios, load_profiles, num_reduced_scenarios)

    ## Construct Stochastic Program
    ...
    # need to think about how linprog script is structured and how best to do this
    # I can iterate through scenario and construct schemas for each scenario using building-year pairs
    ...

    ## Solve
    ...
    # check available solvers (try Gurobi, if not use HiGHS)

    ## Format results
    ...

    return ... # capacities and cost components

def evaulate_system(sampled_scenarios, data_dir, building_file_pattern, costs_dict, tau=48):
    # see sys_eval.py in VOI_in_CAS repo for reference

    return ...

if __name__ == '__main__':

    years = list(range(2012, 2018))
    ids = [0, 4, 8, 19, 25, 40, 58, 102, 104, 118]
    n_buildings = 8

    np.random.seed(0)
    n_samples = 10000
    scenarios = np.array([list(zip(np.random.choice(ids, n_buildings),np.random.choice(years, n_buildings))) for _ in range(n_samples)])

    ...