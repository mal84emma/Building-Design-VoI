"""Reduce load profiles scenarios using stats of aggregate load profile."""

import numpy as np
from scenarioReducer import Fast_forward
from utils.data_processing import scale_profile


def rescale_array(a, invert=False, bounds=None):
    assert not (invert and bounds is None), 'bounds must be provided when inverting scaling.'
    old_bounds = (a.min(), a.max()) if not invert else (0, 1)
    new_bounds = (0, 1) if not invert else bounds
    return np.interp(a, old_bounds, new_bounds)

def get_scenario_stats(building_scenario_vector, load_profiles_dict):
    """Compute mean, standard deviation, and peak of aggregate load for a given scenario,
    i.e. set of building-year profiles (profiles for each building in scenario)."""

    load_profiles = []

    for building_tuple in building_scenario_vector:
        if len(building_tuple) == 2:
            building_id,year = building_tuple
        if len(building_tuple) == 4:
            building_id,year = building_tuple[:2]
            mean,peak = building_tuple[2:]

        load_profile = load_profiles_dict[f'{int(building_id)}-{int(year)}']

        if len(building_tuple) == 4: # if mean and peak provide, perform profile scaling
            load_profile = scale_profile(load_profile, mean, peak)

        load_profiles.append(load_profile)

    aggregate_load = np.sum(load_profiles, axis=0)

    return np.mean(aggregate_load), np.std(aggregate_load), np.max(aggregate_load)

def reduce_load_scenarios(sampled_scenarios, load_profiles_dict, num_scenarios=10):
    """Determine reduced set of scenarios based on stats of aggregate load profiles.

    Scenarios are described by vectors of building id-year pairs (tuple) for each building in the scenario."""

    # compute stats of aggregate load profiles for each scenario
    scenario_stats = np.array([get_scenario_stats(scenario, load_profiles_dict) for scenario in sampled_scenarios])
    bounds = [(np.min(scenario_stats[:,j]), np.max(scenario_stats[:,j])) for j in range(3)]

    # rescale stats to [0,1] (independently on each axis) for clustering
    scaled_scenario_stats = np.array([rescale_array(scenario_stats[:,j]) for j in range(3)]).T

    # set up scenario reducer object - assume all scenarios have equal probability
    probs = np.ones(shape=sampled_scenarios.shape[0])/sampled_scenarios.shape[0]
    FFreducer = Fast_forward(scaled_scenario_stats.T, probs)

    # perform scenario reduction using 1-norm distance metric
    reduced_scenario_stats, reduced_probs, reduced_indices = FFreducer.reduce(distance=1,n_scenarios=num_scenarios)

    return sampled_scenarios[reduced_indices], reduced_probs