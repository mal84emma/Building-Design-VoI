"""Definition of probabilistic models for decision problem."""

import os
import numpy as np
from cmdstanpy import CmdStanModel

# Turn off Stan logging
import logging
logger = logging.getLogger('cmdstanpy')
logger.addHandler(logging.NullHandler())
logger.propagate = False
logger.setLevel(logging.WARNING)



def shape_prior_model(n_buildings,n_samples,prob_config):
    """Sample scenarios of building-year load profile pairs from prior distribution.
    Scenarios are Nx2 arrays of building-year tuples for each of the N buildings in the system.
    Prior model assumes all buildings and years are equally likely and independent.

    Args:
        n_buildings (int): Number of buildings in system.
        n_samples (int): Number of scenarios to samples from prior.
        prob_config (dict): Parameters defining probability model configuration.

    Returns:
        np.array: Array of scenarios (Nx2 arrays of building-year tuples).
        np.array: Array of measurements (building ids only, no year information).
    """

    bs = np.random.choice(prob_config['ids'], (n_samples,n_buildings))
    ys = np.random.choice(prob_config['years'], (n_samples,n_buildings))

    return np.array([list(zip(bs[i],ys[i])) for i in range(n_samples)]), np.array([list(zip(bs[i],[-1]*n_buildings)) for i in range(n_samples)])

def shape_posterior_model(sampled_ids,n_samples,prob_config):
    """Sample scenarios of building-year load profile pairs from posterior distribution,
    i.e. with given (sampled) set of building ids.
    Scenarios are Nx2 arrays of building-year tuples for each of the N buildings in the system.
    Posterior model takes in list of building ids, and samples years only, assuming equal
    likelihood.

    Args:
        sampled_ids (list[int]): List of building ids (measurements) in posterior scenarios.
        n_samples (int): Number of scenarios to samples from prior.
        prob_config (dict): Parameters defining probability model configuration.

    Returns:
        np.array: Array of posterior scenarios (Nx2 arrays of building-year tuples).
    """

    return np.array([list(zip(sampled_ids,np.random.choice(prob_config['years'],len(sampled_ids)))) for _ in range(n_samples)])


def level_prior_model(n_buildings,n_samples,prob_config):
    """Sample scenarios of (building-year) & (mean-peak) pairs from prior distribution.
    Scenarios are Nx4 arrays of building-year-mean-peak tuples for each of the N buildings in the system.
    Prior model assumes all buildings and years are equally likely and independent.
    Mean and peak values (in kWh/hr) are modelled as Gaussian and uniform distributions respectively.

    Args:
        n_buildings (int): Number of buildings in system.
        n_samples (int): Number of scenarios to samples from prior.
        prob_config (dict): Parameters defining probability model configuration.

    Returns:
        np.array: Array of scenarios (Nx4 arrays of building-year-mean-peak tuples).
        np.array: Array of measurements (building id, and mean & peak load values, no year information).
    """

    bs = np.random.choice(prob_config['ids'], (n_samples,n_buildings))
    ys = np.random.choice(prob_config['years'], (n_samples,n_buildings))
    mus = np.round(np.random.normal(prob_config['mean_load_mean'], prob_config['mean_load_std'], (n_samples,n_buildings)),1)
    ps = np.round(np.random.uniform(prob_config['peak_load_min'], prob_config['peak_load_max'], (n_samples,n_buildings)),1)

    # sample measured values
    msrd_mus = np.round(np.random.normal(mus,mus*prob_config['mean_load_msr_error']),1)
    msrd_ps = np.round(np.random.normal(ps,ps*prob_config['peak_load_msr_error']),1)

    return np.array([list(zip(bs[i],ys[i],mus[i],ps[i])) for i in range(n_samples)]), np.array([list(zip(bs[i],[-1]*n_buildings,msrd_mus[i],msrd_ps[i])) for i in range(n_samples)])

def level_posterior_model(sampled_ids,sampled_mus,sampled_peaks,n_samples,prob_config,info='mean+peak'):
    """Sample scenarios of (building-year) & (mean-peak) pairs from posterior distribution.
    i.e. with given (sampled) set of building ids, and mean & peak loads.
    Scenarios are Nx4 arrays of building-year-mean-peak tuples for each of the N buildings in the system.
    Posterior assumes perfect information provided by sample for building ids, and imperfect info
    mean and/or peak load.
    The mean and peak load samples used are the **measured** values, not the true values.

    Args:
        sampled_ids (list[int]): List of sampled building ids (used for all
            posterior scenarios).
        NOTE: for the imperfect info posterior, the mean and peak load samples
        must be the **measured values**, not the true values.
        sampled_mus (list[float]): List of measured mean loads (kW).
        sampled_peaks (list[float]): List of measured peak loads (kW).
        n_samples (int): Number of scenarios to samples from prior.
        prob_config (dict): Parameters defining probability model configuration.
        info (str, optional): Type of information provided to posterior by
            sample. One of ['mean', 'peak', 'mean+peak']. Defaults to 'mean+peak'.

    Returns:
        np.array: Array of scenarios, theta|z (Nx4 arrays of building-year-mean-peak tuples).
    """

    n_buildings = len(sampled_ids)

    ys = np.random.choice(prob_config['years'], (n_samples,n_buildings))

    if info in ['mean','mean+peak']:
        mean_post_file = os.path.join('stan_models','mean_load_posterior.stan')
        mean_stan_model = CmdStanModel(stan_file=mean_post_file)

        building_mus = []
        for mu in sampled_mus:
            # sample theta|z from posterior using Stan
            data = {'mu':prob_config['mean_load_mean'],'sigma':prob_config['mean_load_std'],'error':prob_config['mean_load_msr_error'],'z':mu}
            inits = {'theta':prob_config['mean_load_mean']}
            post_fit = mean_stan_model.sample(data=data, inits=inits, iter_warmup=n_samples, iter_sampling=n_samples*prob_config['thin_factor'], chains=1, show_progress=False)
            candidate_mus = np.round(post_fit.stan_variable('theta')[::prob_config['thin_factor']],1)
            building_mus.append(candidate_mus)
        mus = np.array(building_mus).T
    else:
        mus = np.round(np.random.normal(prob_config['mean_load_mean'], prob_config['mean_load_std'], (n_samples,n_buildings)),1)

    if info in ['peak','mean+peak']:
        peak_post_file = os.path.join('stan_models','peak_load_posterior.stan')
        peak_stan_model = CmdStanModel(stan_file=peak_post_file)

        building_ps = []
        for p in sampled_peaks:
            # sample theta|z from posterior using Stan
            data = {'low':prob_config['peak_load_min'],'high':prob_config['peak_load_max'],'error':prob_config['peak_load_msr_error'],'z':p}
            inits = {'theta':np.mean([prob_config['peak_load_min'],prob_config['peak_load_max']])}
            post_fit = peak_stan_model.sample(data=data, inits=inits, iter_warmup=n_samples, iter_sampling=n_samples*prob_config['thin_factor'], chains=1, show_progress=False)
            candidate_ps = np.round(post_fit.stan_variable('theta')[::prob_config['thin_factor']],1)
            building_ps.append(candidate_ps)
        ps = np.array(building_ps).T
    else:
        ps = np.round(np.random.uniform(prob_config['peak_load_min'], prob_config['peak_load_max'], (n_samples,n_buildings)),1)

    return np.array([list(zip(sampled_ids,ys[i],mus[i],ps[i])) for i in range(n_samples)])



if __name__ == '__main__':
    # give it a spin

    np.random.seed(0)

    n_buildings = 5
    n_samples = 3
    years = list(range(2012, 2018))
    ids = [0, 4, 8, 19, 25, 40, 58, 102, 104, 118]

    prob_config = {
    'ids': ids,
    'years': years,
    'mean_load_mean': 100.0,
    'mean_load_std': 25.0,
    'mean_load_msr_error': 0.1,
    'peak_load_min': 200.0,
    'peak_load_max': 400.0,
    'peak_load_msr_error': 0.075,
    'thin_factor': 10
    }

    shape_prior_scenarios, shape_prior_measurements = shape_prior_model(n_buildings, n_samples, prob_config)
    print("Shape prior scenarios:")
    print(shape_prior_scenarios)
    print(shape_prior_measurements)

    shape_posterior_scenarios = shape_posterior_model(shape_prior_measurements[0][:,0], n_samples, prob_config)
    print("Shape posterior scenarios:")
    print(shape_posterior_scenarios)

    level_prior_scenarios, level_prior_measurements = level_prior_model(n_buildings, n_samples, prob_config)
    print("Level prior scenarios:")
    print(level_prior_scenarios)
    print(level_prior_measurements)

    level_posterior_scenarios = level_posterior_model(level_prior_measurements[0][:,0], level_prior_measurements[0][:,2], level_prior_measurements[0][:,3], n_samples, prob_config)
    print("Level posterior scenarios:")
    print(level_posterior_scenarios)