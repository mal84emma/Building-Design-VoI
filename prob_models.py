"""Definition of probabilistic models for decision problem."""

import numpy as np


def shape_prior_model(n_buildings,n_samples,building_ids,years):
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

def shape_posterior_model(sampled_ids,n_samples,years):
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


def level_prior_model(n_buildings,n_samples,building_ids,years):
    """Sample scenarios of (building-year) & (mean-peak) pairs from prior distribution.
    Scenarios are Nx4 arrays of building-year-mean-peak tuples for each of the N buildings in the system.
    Prior model assumes all buildings and years are equally likely and independent.
    Mean and peak values (in kWh/hr) are modelled as ... (ToDo).

    Args:
        n_buildings (int): Number of buildings in system.
        n_samples (int): Number of scenarios to samples from prior.
        building_ids (list[int]): List of valid building ids to sample from.
        years (list[int]): List of valid years to sample from.

    Returns:
        np.array: Array of scenarios (Nx2 arrays of building-year tuples).
    """

    bs = np.random.choice(building_ids, (n_samples,n_buildings))
    ys = np.random.choice(years, (n_samples,n_buildings))
    mus = np.round(np.random.uniform(50, 150, (n_samples,n_buildings)),1) # ToDo - confirm model
    ps = np.round(np.random.uniform(250, 500, (n_samples,n_buildings)),1) # ToDo - confirm model

    return np.array([list(zip(bs[i],ys[i],mus[i],ps[i])) for i in range(n_samples)])

def level_posterior_model(sampled_ids,sampled_mus,sampled_peaks,n_samples,years,info='both'):
    """ToDo"""

    n_buildings = len(sampled_ids)

    ys = np.random.choice(years, (n_samples,n_buildings))

    if info in ['mu','both']:
        mus = np.round(np.tile(sampled_mus,(n_samples,n_buildings)),1)
    else:
        mus = np.round(np.random.uniform(50, 150, (n_samples,n_buildings)),1) # ToDo - confirm model
    
    if info in ['peak','both']:
        ps = np.round(np.tile(sampled_peaks,(n_samples,n_buildings)),1)
    else:
        ps = np.round(np.random.uniform(250, 500, (n_samples,n_buildings)),1) # ToDo - confirm model
    
    return np.array([list(zip(sampled_ids,ys[i],mus[i],ps[i])) for i in range(n_samples)])



if __name__ == '__main__':
    # give it a spin

    n_buildings = 4
    n_samples = 10
    years = list(range(2012, 2018))
    ids = [0, 4, 8, 19, 25, 40, 58, 102, 104, 118]

    shape_prior_scenarios = shape_prior_model(n_buildings, n_samples, ids, years)
    print("Shape prior scenarios:")
    print(shape_prior_scenarios)

    shape_posterior_scenarios = shape_posterior_model(shape_prior_scenarios[0][:,0], n_samples, years)
    print("Shape posterior scenarios:")
    print(shape_posterior_scenarios)

    level_prior_scenarios = level_prior_model(n_buildings, n_samples, ids, years)
    print("Level prior scenarios:")
    print(level_prior_scenarios)

    level_posterior_scenarios = level_posterior_model(level_prior_scenarios[0][:,0], level_prior_scenarios[0][:,2], level_prior_scenarios[0][:,3], n_samples, years)
    print("Level posterior scenarios:")
    print(level_posterior_scenarios)