"""Helper functions for loading and saving scenarios."""

import csv
import ast
import numpy as np

def save_scenarios(scenarios, out_path):

    header = ['Scenario no.']
    header.extend([f'SB{i}' for i in range(scenarios.shape[1])])

    with open(out_path, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        for scenario_no, scenario in enumerate(scenarios):
            writer.writerow([scenario_no] + [(b,y) for b,y in scenario])

def load_scenarios(in_path):

    with open(in_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        scenarios = []
        for row in reader:
            scenarios.append([ast.literal_eval(t) for t in row[1:]])

    return np.array(scenarios)

def save_LP_design_results(results, out_path):
    # for saving system design and LP objective results
    ...

def load_LP_design_results(in_path):

    ...

def save_eval_results(results, out_path):
    # for saving the cost results from n evaulations of a system design
    ...

def load_eval_results(in_path):

    ...