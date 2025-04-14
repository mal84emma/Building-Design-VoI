"""Check results files have been generated correctly."""

import os
import sys
from experiments.configs.experiments import parse_experiment_args


if __name__ == "__main__":

    # Get experiment settings
    expt_type = str(sys.argv[1])
    [expt_id,n_buildings,info_id] = [int(sys.argv[i]) for i in range(2,5)]

    assert expt_type in ['post_design','post_eval'], f"Experiment type must be one of 'prior','post_design','post_eval', {expt_type} given."

    from experiments.configs.config import *

    expt_name, sizing_constraints, info_type = parse_experiment_args(expt_id, n_buildings, info_id)

    nsamples = 255

    # Set up file paths
    results_dir = os.path.join(results_dir,f'posterior_{expt_name}_{n_buildings}b_{info_type}_info')

    if expt_type == 'post_design':
        fpattern = f's%s_posterior_design_results.csv'
        path_pattern = os.path.join(results_dir,'designs',fpattern)
    elif expt_type == 'post_eval':
        fpattern = f's%s_posterior_eval_results.csv'
        path_pattern = os.path.join(results_dir,'evals',fpattern)

    # Check if files exist
    for i in range(nsamples+1):
        if not os.path.exists(path_pattern % i):
            print(f"{i}: File {path_pattern % i} does not exist.")