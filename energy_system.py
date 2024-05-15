"""Design and control district energy system for set of sampled scenarios.

System design steps:
1. Load data for scenario load profiles.
2. Perform scenario reduction.
3. Construct Stochastic Program.
4. Solve and report result.

System control steps:
1. Load data for scenario.
2. Construct controller LP model.
3. Iterate through time steps and schedule battery with LP.
4. Compute cost components from system history.
"""

import os
import json
import numpy as np
import pandas as pd
import cvxpy as cp
import gurobipy as gp
import multiprocess as mp

import time
import warnings
from tqdm import tqdm
from functools import partial
from load_scenario_reduction import reduce_load_scenarios
from utils import build_schema, get_Gurobi_WLS_env
from utils.plotting import init_profile_fig, add_profile
from linmodel import LinProgModel
from citylearn.citylearn import CityLearnEnv



def design_system(
        sampled_scenarios,
        data_dir,
        building_file_pattern,
        cost_dict,
        solver_kwargs={},
        num_reduced_scenarios=None,
        sim_duration=None,
        t_start=0,
        process_id=None,
        show_progress=False
    ):
    """Use Stochastic Program formulation to design district energy system for
    given set of scenarios.

    Args:
        sampled_scenarios (List): List of Nx2 vectors defining scenarios. I.e.
            iterable of vectors containing (building_id, year) tuples for each
            building in each scenario.
        data_dir (str or Path): Path to directory containing Citylearn compatible
            data.
        building_file_pattern (fstr): Pattern of building load data files. Must
            contain {id} and {year} placeholders.
        cost_dict (dict):Dictionary of cost parameters for Stochastic Program.
            Keys are as specified in `linmodel.py`.
        solver_kwargs (dict, optional): Solver options for LP. Defaults to {}.
        num_reduced_scenarios (int, optional): Number of scenarios to be used
            in Stochastic Program, selected via Scenario Reduction. If None,
            all provided scenarios are used, assuming equal probability.
            Defaults to None.
        sim_duration (int, optional): Number of timesteps to include in
            Stocasthic Program. If None, all available timesteps are used.
            Defaults to None.
        t_start (int, optional): Start timestep for data. Defaults to 0.
        process_id (int, optional): Unique process ID for tagging schemas when
            fn called via multiprocessing. Defaults to None.
        show_progress (bool, optional): Whether to display progress. Defaults to False.

    Returns:
        dict: Dictionary of LP results return from `LinProgModel.solve_LP`.
    """

    ## Get load profiles for all scenarios (reduces load time)
    # ========================================================
    if show_progress: print("Loading data...")
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
    if show_progress: print("Reducing scenarios...")
    if num_reduced_scenarios is not None:
        reduced_scenarios,reduced_probs = reduce_load_scenarios(sampled_scenarios, load_profiles, num_reduced_scenarios)
    else:
        reduced_scenarios = sampled_scenarios
        reduced_probs = np.ones(len(reduced_scenarios))/len(reduced_scenarios)
    if show_progress:
        print("Reduced scenarios:\n", reduced_scenarios)
        print("Reduced probabilities:\n", reduced_probs)

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
        params['load_data_paths'] = [building_file_pattern.format(id=b,year=y) for b,y in lys]
        params['battery_efficiencies'] = [params['base_battery_efficiency']]*len(params['building_names'])
        params.pop('base_battery_efficiency', None)
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
    lp.generate_LP(cost_dict,design=True,scenario_weightings=reduced_probs,use_parameters=False)

    ## Solve and report results
    # =========================
    if solver_kwargs is None: # default to using HiGHS
        solver_kwargs = {'solver': 'SCIPY'}

    if show_progress: print("Solving LP...")
    lp_results = lp.solve_LP(**solver_kwargs,verbose=show_progress,canon_backend=cp.SCIPY_CANON_BACKEND)
    # SCIPY compilation backend provides somewhat better performance on sparse problem

    results = lp_results.copy()
    results['reduced_scenarios'] = reduced_scenarios
    results['reduced_probs'] = reduced_probs

    ## Clean up schemas
    # =================
    for path in schema_paths:
        if os.path.normpath(path).split(os.path.sep)[-1] != 'schema.json':
            os.remove(path)
    if show_progress: print("Design complete.")

    return results


def evaulate_system(
        schema_path,
        cost_dict,
        grid_con_capacity,
        design=False,
        tau=48,
        clip_level='m',
        solver_kwargs={},
        show_progress=False,
        plot=False
    ):
    """Simulate district energy system with given design (schema) for given
    scenario and evaluate its performance (operational/system cost).

    Args:
        schema_path (str, Path): Path to schema file defining energy system.
        cost_dict (dict):Dictionary of cost parameters for Linear Program.
            Keys are as specified in `linmodel.py`.
        grid_con_capacity (float): Capacity of grid connection in system
            design (kW).
        design (bool, optional): Whether to evaluate total system cost,
            or just operational cost. Defaults to False.
        tau (int, optional): Planning horizon of Linear MPC controller.
            Defaults to 48.
        clip_level (str, optional): Level at which to clip system costs.
            See `linmodel.py` for more info. Defaults to 'm'.
        solver_kwargs (dict, optional): Kwargs to pass to LP solver.
            Defaults to {}.
        show_progress (bool, optional): Whether to display simulation progress
            in console. Defaults to False.
        plot (bool, optional): Whether to plot system profiles for simulated
            scenario. Defaults to False.

    Returns:
        dict: Dictionary containing total system cost from simulation and cost components.
    """

    # Initialise CityLearn environment object.
    env = CityLearnEnv(schema=schema_path)

    # Initialise Linear MPC object.
    lp = LinProgModel(env=env)
    lp.tau = tau
    lp.generate_LP(cost_dict,design=False,use_parameters=True)

    # Initialise control loop.
    lp_solver_time_elapsed = 0
    num_steps = 0
    done = False

    # Initialise environment.
    observations = env.reset()
    soc_obs_index = 22
    current_socs = np.array([
        [charge*capacity for charge,capacity in\
         zip(np.array(observations)[:,soc_obs_index],lp.battery_capacities.flatten())]
        ]) # get initial SoCs
    max_grid_usage = grid_con_capacity

    # Execute control loop.
    with tqdm(total=env.time_steps,disable=(not show_progress)) as pbar:

        while not done:
            if num_steps%100 == 0: pbar.update(100)

            # Compute MPC action.
            # ====================================================================
            if (num_steps <= (env.time_steps - 1) - tau):
                # setup and solve predictive Linear Program model of system
                lp_start = time.perf_counter()
                lp.set_time_data_from_envs(t_start=num_steps, tau=tau, initial_socs=current_socs) # load ground truth data
                lp.set_LP_parameters(max_grid_usage=max_grid_usage)
                results = lp.solve_LP(**solver_kwargs,ignore_dpp=False,verbose=False)
                actions: np.array = results['battery_inflows'][0][:,0].reshape((lp.N,1))/lp.battery_capacities
                lp_solver_time_elapsed += time.perf_counter() - lp_start

            else: # if not enough time left to grab a full length ground truth forecast: do nothing
                actions = np.zeros((lp.N,1))

            # Apply action to environment.
            # ====================================================================
            observations, _, done, _ = env.step(actions)

            # Update battery states-of-charge & max grid usage
            # ====================================================================
            current_socs = np.array([
                [charge*capacity for charge,capacity in\
                 zip(np.array(observations)[:,soc_obs_index],lp.battery_capacities.flatten())]
                ])
            max_grid_usage = np.max([
                np.max(np.clip(np.sum([b.net_electricity_consumption for b in env.buildings],axis=0),0,None))/lp.delta_t,
                max_grid_usage
                ])

            # Iterate step counter
            num_steps += 1

    if show_progress: print("Evaluation complete.")

    # Plot system profiles
    # ====================
    if plot:
        fig = init_profile_fig(
            y_titles={'primary': 'Building energy usage (kWh)', 'secondary': 'Battery SoC (kWh)'}
            )

        for b in env.buildings:
            fig = add_profile(fig, b.net_electricity_consumption, name=f'{b.name} energy', secondary_y=False)
            fig = add_profile(fig, b.electrical_storage.soc, name=f'{b.name} SoC', secondary_y=True)

        fig.write_html(f'{os.path.splitext(os.path.basename(schema_path))[0]}_plot.html')
        fig.show()

    # Compute objective fn of simulation
    # ==================================
    # compute useful variables
    elec_prices = env.buildings[0].pricing.electricity_pricing
    carbon_intensities = env.buildings[0].carbon_intensity.carbon_intensity
    positive_building_draws = [np.clip(b.net_electricity_consumption,0,None) for b in env.buildings]
    positive_grid_draw = np.clip(np.sum([b.net_electricity_consumption for b in env.buildings],axis=0),0,None)

    objective_contributions = []
    # Add electricity price contribution
    if clip_level in ['d']:
        objective_contributions += [positive_grid_draw @ elec_prices]
    elif clip_level in ['b','m']:
        objective_contributions += [np.sum([pos_draw @ elec_prices for pos_draw in positive_building_draws])]
    # Add carbon price contribution
    if clip_level in ['d','m']:
        objective_contributions += [positive_grid_draw @ carbon_intensities * cost_dict['carbon']]
    elif clip_level in ['b']:
        objective_contributions += [np.sum([pos_draw @ carbon_intensities for pos_draw in positive_building_draws]) * cost_dict['carbon']]
    # Add grid connection exceedance cost
    objective_contributions += [np.max((np.max(positive_grid_draw)/lp.delta_t - grid_con_capacity),0) * cost_dict['grid_excess'] * (env.time_steps*lp.delta_t)/24]

    if design: # Multiply opex costs up to design lifetime & add capex costs
        objective_contributions = [contr*cost_dict['opex_factor'] for contr in objective_contributions] # extend opex costs to design lifetime
        objective_contributions += [grid_con_capacity * cost_dict['grid_capacity'] * cost_dict['opex_factor'] * (env.time_steps*lp.delta_t)/24]
        objective_contributions += [np.sum([b.electrical_storage.capacity_history[0] for b in env.buildings]) * cost_dict['battery']]
        objective_contributions += [np.sum([b.pv.nominal_power for b in env.buildings]) * cost_dict['solar']]

    return {'objective': np.sum(objective_contributions), 'objective_contrs': objective_contributions}


def evaulate_multi_system_scenarios(
        sampled_scenarios,
        system_design,
        data_dir,
        building_file_pattern,
        design,
        cost_dict,
        tau=48,
        solver_kwargs={},
        n_processes=None,
        show_progress=False,
        plot=False
    ):
    """Evaluate performance of given system design over multiple scenarios (simulations).

    Args:
        sampled_scenarios (List): List of Nx2 vectors defining scenarios. I.e.
            iterable of vectors containing (building_id, year) tuples for each
            building in each scenario.
        system_design (dict): Dictionary of system design parameters.
            Key-values are:
                - 'battery_capacities': List of battery energy capacity in each
                    building (kWh)
                - 'solar_capacities': List of solar power capacity in each
                    building (kWp)
                - 'grid_con_capacity': Grid connection capacity (kW)
        building_file_pattern (fstr): Pattern of building load data files. Must
            contain {id} and {year} placeholders.
        design (bool, optional): Whether to evaluate total system cost,
            or just operational cost. Defaults to False.
        cost_dict (dict):Dictionary of cost parameters for Linear Program.
            Keys are as specified in `linmodel.py`.
        tau (int, optional): Planning horizon of Linear MPC controller.
            Defaults to 48.
        solver_kwargs (dict, optional): Kwargs to pass to LP solver.
            Defaults to {}.
        n_processes (int, optional): Number of processes to use for running
            simulations in parallel using `multiprocess.Pool`. If None, sims
            are run sequentially. Defaults to None.
        show_progress (bool, optional): Whether to display simulation progress
            in console. Defaults to False.
        plot (bool, optional): Whether to plot system profiles for each scenario.
            Defaults to False.

    Returns:
        Overll mean system cost and list of results for each scenario.
    """

    if show_progress: print("Generating scenarios...")
    # Load base system parameters
    with open(os.path.join('resources','base_system_params.json')) as jfile:
        base_params = json.load(jfile)

    base_params['data_dir_path'] = data_dir

    # Set system design parameters
    base_params['battery_efficiencies'] = [base_params['base_battery_efficiency']]*len(sampled_scenarios[0])
    base_params.pop('base_battery_efficiency', None)
    base_params['battery_energy_capacities'] = system_design['battery_capacities'].flatten()
    base_params['battery_power_capacities'] = [energy*cost_dict['battery_power_ratio'] for energy in system_design['battery_capacities'].flatten()]
    base_params['pv_power_capacities'] = system_design['solar_capacities'].flatten()

    # Build schema for each scenario
    scenario_schema_paths = []
    for m, lys in enumerate(sampled_scenarios):
        params = base_params.copy()
        params['building_names'] = [f'TB{i}' for i in range(len(lys))]
        params['load_data_paths'] = [building_file_pattern.format(id=b,year=y) for b,y in lys]
        params['schema_name'] = f'EVAL_schema_s{m}'
        schema_path = build_schema(**params)
        scenario_schema_paths.append(schema_path)

    # Evaluate system performance for each scenario
    if show_progress: print("Evaluating scenarios...")
    if n_processes is None:
        eval_results = [
            evaulate_system(
                schema_path, cost_dict, system_design['grid_con_capacity'], design=design,
                    tau=tau, solver_kwargs=solver_kwargs, show_progress=show_progress, plot=plot)\
                        for schema_path in tqdm(scenario_schema_paths, disable=(not show_progress))
        ]
    else:
        eval_wrapper = partial(evaulate_system,
                               cost_dict=cost_dict, grid_con_capacity=system_design['grid_con_capacity'],
                               design=design, tau=tau,
                               solver_kwargs=solver_kwargs,
                               show_progress=False, plot=False
                              )
        with mp.Pool(n_processes) as pool:
            eval_results = list(tqdm(pool.imap(eval_wrapper, scenario_schema_paths), total=len(scenario_schema_paths), disable=(not show_progress)))

    # Clean up schemas
    for path in scenario_schema_paths:
        if os.path.normpath(path).split(os.path.sep)[-1] != 'schema.json':
            os.remove(path)
    if show_progress: print("Evaluation complete.")

    return np.mean([res['objective'] for res in eval_results]), eval_results


if __name__ == '__main__':
    # give each of the fns a test run

    with warnings.catch_warnings():
        # filter pandas warnings, `DeprecationWarning: np.find_common_type is deprecated.`
        warnings.simplefilter("ignore", category=DeprecationWarning)

        start = time.time()

        try:
            m = gp.Model()
            e = get_Gurobi_WLS_env()
            solver_kwargs = {'solver': 'GUROBI', 'env': e}
        except:
            solver_kwargs = {}

        dataset_dir = os.path.join('data','processed')
        building_fname_pattern = 'ly_{id}-{year}.csv'

        years = list(range(2012, 2018))
        ids = [0, 4, 8, 19, 25, 40, 58, 102, 104] # 118
        n_buildings = 8

        cost_dict = {
            'carbon': 1.0, #5e-1, # $/kgCO2
            'battery': 1e3, #1e3, # $/kWh
            'solar': 1e3, #2e3, # $/kWp
            'grid_capacity': 25e-2/0.95, # $/kW/day - note, this is waaay more expensive that current
            'grid_excess': 100e-2/0.95, # $/kW/day - note, this is a waaay bigger penalty than current
            'opex_factor': 20,
            'battery_power_ratio': 0.4
        }

        np.random.seed(0)
        n_samples = 1000
        scenarios = np.array([list(zip(np.random.choice(ids, n_buildings),np.random.choice(years, n_buildings))) for _ in range(n_samples)])

        # test system design
        design_results = design_system(scenarios, dataset_dir, building_fname_pattern, cost_dict,
                                        solver_kwargs=solver_kwargs, num_reduced_scenarios=10,
                                        show_progress=True
                                    )

        for key in ['objective','objective_contrs','battery_capacities','solar_capacities','grid_con_capacity']:
            print(design_results[key])

        system_design = {
            'battery_capacities': design_results['battery_capacities'],
            'solar_capacities': design_results['solar_capacities'],
            'grid_con_capacity': design_results['grid_con_capacity']
        }

        solver_kwargs = {} # HiGHS better for operational LP
        # test system evaluation
        mean_cost, eval_results = evaulate_multi_system_scenarios(
                scenarios[:20], system_design, dataset_dir, building_fname_pattern,
                design=True, cost_dict=cost_dict, tau=48, n_processes=None,
                solver_kwargs=solver_kwargs, show_progress=True, plot=False
            )

        print('Mean system cost:', mean_cost)
        print('Mean system cost components:', np.mean([res['objective_contrs'] for res in eval_results],axis=0))

        # compare objective fn returned by LP to actual cost from simulation (LP over-optimism)
        print('LP objective:', design_results['objective'])
        print('Simulation cost:', mean_cost)
        print('Difference:', mean_cost - design_results['objective'])

        end = time.time()
        print('Total run time:', end-start)