"""Visualise convergence of MC estimate of prior solution mean cost."""

# Hack to emulate running files from root directory.
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# run using `python -m experiments.{fname}`

import numpy as np
from utils import data_handling
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # Load prior eval results.
    eval_results_path = os.path.join('experiments','results','prior_eval_results.csv')
    eval_results = data_handling.load_eval_results(eval_results_path)

    # Compute MC convergence of mean cost.
    MC_estimates = [np.mean([res['objective'] for res in eval_results[:i+1]]) for i in range(len(eval_results))]

    # Plot convergence of MC estimate of mean cost.
    fig = plt.figure()
    plt.plot(range(1,len(MC_estimates)+1), np.array(MC_estimates)/1e6, 'k-')
    plt.xlabel('Number of scenarios')
    plt.ylabel('Mean cost ($m)')
    plt.xlim(0,len(MC_estimates)+1)
    plt.show()