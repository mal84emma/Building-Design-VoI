"""Quantify benefit derived from each part of the system for prior case."""

import os
import sys
import numpy as np
from utils import data_handling


if __name__ == '__main__':

    n_buildings = int(sys.argv[1])

    from experiments.configs.config import *

    results_dir = os.path.join('experiments','results')

    cases = ['unconstr','constr_solar','battery_only','solar_only','neither']
    results = {}

    # Load eval results for each system case.
    # =======================================
    for case in cases:
        eval_results_path = os.path.join(results_dir,'prior',f'{case}_{n_buildings}b_eval_results.csv')
        eval_results = data_handling.load_eval_results(eval_results_path)
        costs = [res['objective'] for res in eval_results]
        case_results = {
            'mean': np.mean(costs),
            'std': np.std(costs),
            'min':np.min(costs),
            'max':np.max(costs)
        }
        results[case] = case_results
        print(case,case_results)

    # Compute benefits dervied from each part of the system.
    # ======================================================
    ...
