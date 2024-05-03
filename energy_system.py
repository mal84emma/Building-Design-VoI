"""Optimise district energy system for set of sampled scenarios.

Steps:
1. Load data for scenario load profiles.
2. Perform scenario reduction.
3. Construct Stochastic Program.
4. Solve and report result.
"""

import os
import json
import numpy as np
import pandas as pd
import cvxpy as cp

from load_scenario_reduction import reduce_load_scenarios
from utils import build_schema
from linmodel import LinProgModel
from citylearn.citylearn import CityLearnEnv


# NOTE: see VoI_in_CAS repo for reference and for code to steal


def optimise_system(
        sampled_scenarios,
        data_dir,
        building_file_pattern,
        cost_dict,
        solver_kwargs=None,
        num_reduced_scenarios=None,
        sim_duration=None,
        t_start=0,
        process_id=None
    ):
    """ToDO"""
    # NOTE: scenarios must be vector of Nx2 vectors, (building_id, year) tuple for each building
    # Use num_red_scenarios = None for deterministic cases

    ## Get load profiles for all scenarios (reduces load time)
    # ========================================================
    building_year_pairs = np.unique(np.concatenate(sampled_scenarios,axis=0),axis=0)
    load_profiles = {
        f'{building_id}-{year}': pd.read_csv(
            os.path.join(data_dir, building_file_pattern.format(id=building_id, year=year)),
            usecols=['Equipment Electric Power [kWh]']
            )['Equipment Electric Power [kWh]'].to_numpy()\
                for (building_id, year) in building_year_pairs
    }

    ## Perform scenario reduction
    # ===========================
    if num_reduced_scenarios is not None:
        reduced_scenarios = reduce_load_scenarios(sampled_scenarios, load_profiles, num_reduced_scenarios)

    ## Construct Stochastic Program
    # =============================
    with open(os.path.join('resources','base_system_params.json')) as jfile:
        base_params = json.load(jfile)

    # load data for each scenario and build schemas
    envs = []
    schema_paths = []
    for m, lys in enumerate(reduced_scenarios):
        params = base_params.copy()
        params['building_names'] = [f'TB{i}' for i in range(len(lys))]
        params['load_data_paths'] = [f'ly_{b}-{y}.csv' for b,y in lys]
        params['battery_efficiencies'] = [params['base_battery_efficiency']]*len(params['building_names'])
        params['schema_name'] = f'SP_schema_s{m}'
        if process_id is not None:
            params['schema_name'] = f'p{process_id}_' + params['schema_name']
        schema_path = build_schema(**params)

        schema_paths.append(schema_path)
        envs.append(CityLearnEnv(schema=schema_path))

        if m == 0: # initialise lp object
            lp = LinProgModel(env=envs[m])
        else:
            lp.add_env(env=envs[m])

    # set data and generate LP
    lp.set_time_data_from_envs(t_start=t_start,tau=sim_duration)
    lp.generate_LP(cost_dict,design=True)
    lp.set_LP_parameters()

    ## Solve and report results
    # =========================
    if solver_kwargs is None: # default to using HiGHS
        solver_kwargs = {'solver': 'SCIPY'}

    lp_results = lp.solve_LP(verbose=True,ignore_dpp=True)

    ## Clean up schemas
    # =================
    for path in schema_paths:
        if os.path.normpath(path).split(os.path.sep)[-1] != 'schema.json':
            os.remove(path)

    return lp_results

def evaulate_system(
        sampled_scenarios,
        data_dir,
        building_file_pattern,
        costs_dict,
        tau=48,
        process_id=None
    ):
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