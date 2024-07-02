"""Sample and save (common) scenarios used for VoI calculations."""

# Hack to emulate running files from root directory.
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# run using `python -m experiments.{fname}`

import numpy as np
from prob_models import prior_model
from utils import data_handling


if __name__ == '__main__':

    from experiments.expt_config import *

    n_samples = 1000

    save_path = os.path.join(save_dir,'sampled_scenarios.csv')

    # Perform sampling.
    np.random.seed(0)
    scenarios = prior_model(n_buildings, n_samples, ids, years)

    # Save data.
    data_handling.save_scenarios(scenarios, save_path)