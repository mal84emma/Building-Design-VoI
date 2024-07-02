"""Sample and save (common) scenarios used for VoI calculations."""

# Hack to emulate running files from root directory.
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# run using `python -m experiments.{fname}`

import numpy as np
from prob_models import shape_prior_model, level_prior_model
import utils.data_handling as data_handling


if __name__ == '__main__':

    # Get run options
    expt_type = str(sys.argv[1])

    from experiments.configs.general_config import *
    if expt_type == 'shape':
        from experiments.configs.shape_expts_config import *
        prior_model = shape_prior_model
    elif expt_type == 'level':
        from experiments.configs.level_expts_config import *
        prior_model = level_prior_model
    else:
        raise ValueError('Invalid run option for `expt_type`. Please provide valid CLI argument.')


    n_samples = 1000

    save_path = os.path.join(results_dir,'sampled_scenarios.csv')

    # Perform sampling.
    np.random.seed(0)
    scenarios = prior_model(n_buildings, n_samples, ids, years)

    # Save data.
    data_handling.save_scenarios(scenarios, save_path)