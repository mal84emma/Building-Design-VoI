"""Helper functions for performing system design & evaluation."""

import os
import time
import utils
import utils.data_handling as data_handling
from energy_system import design_system, evaluate_multi_system_scenarios



def posterior_design(
        scenario_id_tuple,
        out_dir,
        prob_config,
        posterior_model,
        info_type,
        n_post_samples,
        data_dir,
        building_file_pattern,
        cost_dict,
        sizing_constraints={},
        solver_kwargs={},
        num_reduced_scenarios=None,
        show_progress=False
    ):
    """Wrapper function for designing system based on posterior samples.

    Args:
        scenario_id_tuple (tuple): Tuple containing scenario number (index of
            samples from prior) and scenario vector for sampled scenario.
            Single argument to unpack for ease of multiprocessing.
        out_dir (path): Directory to save design results to.
        prob_config (dict): Parameters defining probability model configuration.
        posterior_model (function): Function to samples scenarios from
            posterior distribution conditioned on measured scenario.
        info_type (str): One of ['profile','type','mean','peak','mean+peak'].
            Type of information provided by sampled scenario, determining
            posterior dist. to use, i.e. sampling procedure in `posterior_model`.
        n_post_samples (int): No. of samples to draw from posterior.
        NOTE: all args below are passed to `design_system` function. See docstring
            for details.
        data_dir (path):
        building_file_pattern (str):
        cost_dict (dict):
        sizing_constraints (dict, optional): Defaults to {}.
        solver_kwargs (dict, optional): Defaults to {}.
        num_reduced_scenarios (int, optional): Defaults to None.
        show_progress (bool, optional): Defaults to False.

    Returns:
        dict: Not used.
    """

    scenario_num,measured_scenario = scenario_id_tuple

    # Set up solver environment.
    if solver_kwargs['solver'] == 'GUROBI':
        solver_kwargs['env'] = utils.get_Gurobi_WLS_env(silence=not show_progress if scenario_num != None else True)

    # Sample scenarios from posterior model
    sampled_scenarios = posterior_model(measured_scenario[:,0],measured_scenario[:,2],measured_scenario[:,3],n_post_samples,prob_config,info_type)

    # Design system.
    start = time.time()
    design_results = design_system(
            sampled_scenarios,
            data_dir,
            building_file_pattern,
            cost_dict,
            sizing_constraints=sizing_constraints,
            solver_kwargs=solver_kwargs,
            num_reduced_scenarios=num_reduced_scenarios,
            show_progress=show_progress if scenario_num != None else False,
            process_id=scenario_num
        )
    end = time.time()

    # Save results.
    out_path = os.path.join(out_dir,f's{scenario_num}_posterior_design_results.csv')
    data_handling.save_design_results(design_results, out_path)

    # Report finish.
    print(f'Scenario {scenario_num} design completed in {end-start:.2f}s @ {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}.')

    return design_results


def posterior_evaluation(
        scenario_id_tuple,
        design_results_path_pattern,
        out_dir,
        prob_config,
        posterior_model,
        info_type,
        n_post_samples,
        data_dir,
        building_file_pattern,
        cost_dict,
        solver_kwargs={},
        n_processes=None,
        show_progress=True
    ):
    """Wrapper function for evaluating system based on posterior samples.

    Args:
        scenario_id_tuple (_type_): Tuple containing scenario number
            (index of samples from prior), and scenario vector for 
            sampled scenario. Single argument to unpack for ease of
            multiprocessing.
        design_results_path_pattern (str): Path pattern for loading
            system design results. Formatted with scenario number.
        out_dir (path): Directory to save design results to.
        prob_config (dict): Parameters defining probability model configuration.
        posterior_model (function): Function to samples scenarios from
            posterior distribution conditioned on measured scenario.
        info_type (str): One of ['profile','type','mean','peak','mean+peak'].
            Type of information provided by sampled scenario, determining
            posterior dist. to use, i.e. sampling procedure in `posterior_model`.
        n_post_samples (int): No. of samples to draw from posterior.
        NOTE: all args below are passed to `evaluate_multi_system_scenarios`
            function. See docstring for details.
        data_dir (path):
        building_file_pattern (str):
        cost_dict (dict):
        solver_kwargs (dict, optional): Defaults to {}.
        num_reduced_scenarios (int, optional): Defaults to None.
        n_processes (int, optional): Defaults to None.
        show_progress (bool, optional): Defaults to False.

    Returns:
        dict: Not used.
    """

    scenario_num,measured_scenario = scenario_id_tuple

    # Load system design.
    system_design = data_handling.load_design_results(design_results_path_pattern.format(j=scenario_num))

    # Sample scenarios from posterior model
    sampled_scenarios = posterior_model(measured_scenario[:,0],measured_scenario[:,2],measured_scenario[:,3],n_post_samples,prob_config,info_type)

    # Evaluate system.
    if show_progress: print(f'\nStarting scenario {scenario_num} evaluation @ {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}.')
    start = time.time()
    mean_cost, eval_results = evaluate_multi_system_scenarios(
            sampled_scenarios,
            system_design,
            data_dir,
            building_file_pattern,
            design=True,
            cost_dict=cost_dict,
            solver_kwargs=solver_kwargs,
            n_processes=n_processes,
            show_progress=show_progress,
        )
    end = time.time()

    # Save results.
    out_path = os.path.join(out_dir,f's{scenario_num}_posterior_eval_results.csv')
    data_handling.save_eval_results(eval_results, system_design, sampled_scenarios, out_path)

    # Report finish.
    print(f'\nScenario {scenario_num} evaluation completed in {end-start:.2f}s @ {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}.')

    return eval_results