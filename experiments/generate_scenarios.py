"""Sample and save (common) scenarios used for VoI calculations."""

# Hack to emulate running files from root directory.
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# run using `python -m experiments.{fname}`

import numpy as np
from prob_models import prior_model
import utils.data_handling as data_handling


if __name__ == '__main__':

    # Get run options
    n_buildings = int(sys.argv[1])

    from experiments.configs.config import *

    n_samples = 1000

    save_path = os.path.join(results_dir,f'sampled_scenarios_{n_buildings}b.csv')

    # Perform sampling.
    np.random.seed(0)
    scenarios, measurements = prior_model(n_buildings, n_samples, prob_config)

    # Save data.
    data_handling.save_scenarios(scenarios, measurements, save_path)