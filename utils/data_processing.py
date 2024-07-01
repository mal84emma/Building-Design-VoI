"""Helper functions for manipulating load profile data."""

import numpy as np

def scale_profile(profile, mean, peak):
    """Scale energy usage profile to have a given mean and peak.
    Method uses iterative scaling of peak and mean to achieve both
    mean and peak, with peak within 0.1% of the desired value.

    Args:
        profile (np.array): 1D array of energy usage profile.
        mean (float): Mean load to scale to.
        peak (float): Peak load to scale to.

    Returns:
        np.array: Scaled load profile.
    """

    p = profile.copy()
    p *= mean/np.mean(p)

    while np.abs(np.max(p) - peak) >= 0.1/100*peak:
        mu = np.mean(p)
        p = np.where(p > mu, (p-mu)*peak/np.max(p)+mu, p)
        p *= mean/np.mean(p)

    return p