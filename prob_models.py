"""Definition of probabilistic models for decision problem."""

import numpy as np


def prior_model(n_buildings,n_samples,building_ids,years):
    """Sample scenarios of building-year load profile pairs from prior distribution.
    Scenarios are Nx2 arrays of building-year tuples for each of the N buildings in the system.
    Prior model assumes all buildings and years are equally likely and independent.

    Args:
        n_buildings (int): Number of buildings in system.
        n_samples (int): Number of scenarios to samples from prior.
        building_ids (list[int]): List of valid building ids to sample from.
        years (list[int]): List of valid years to sample from.

    Returns:
        np.array: Array of scenarios (Nx2 arrays of building-year tuples).
    """

    return np.array([list(zip(np.random.choice(building_ids, n_buildings),np.random.choice(years, n_buildings))) for _ in range(n_samples)])

def posterior_model(sampled_ids,n_samples,years):
    """Sample scenarios of building-year load profile pairs from posterior distribution,
    i.e. with given (sampled) set of building ids.
    Scenarios are Nx2 arrays of building-year tuples for each of the N buildings in the system.
    Posterior model takes in list of building ids, and samples years only, assuming equal
    likelihood.

    Args:
        sampled_ids (list[int]): List of building ids in posterior scenarios.
        n_samples (int): Number of scenarios to samples from prior.
        years (list[int]): List of valid years to sample from.

    Returns:
        np.array: Array of posterior scenarios (Nx2 arrays of building-year tuples).
    """

    return np.array([list(zip(sampled_ids,np.random.choice(years,len(sampled_ids)))) for _ in range(n_samples)])


if __name__ == '__main__':
    # give it a spin

    n_buildings = 4
    n_samples = 3
    years = list(range(2012, 2018))
    ids = [0, 4, 8, 19, 25, 40, 58, 102, 104, 118]

    prior_scenarios = prior_model(n_buildings, n_samples, ids, years)
    print("Prior scenarios:")
    print(prior_scenarios)

    posterior_scenarios = posterior_model(prior_scenarios[0][:,0], n_samples, years)
    print("Posterior scenarios:")
    print(posterior_scenarios)
