"""Compute EPVI for building energy system design via LPs."""

# NOTE: follow structure of EVPI function from VoI paper repo
# - compute actions using optimise_system
# - determine true cost of actions using evaluate_system
# - perform for both uncertain building type (ID, uniform prob.) and known building type

from energy_system import optimise_system, evaluate_system

def determine_EVPI(x):

    ...

    return ...


if __name__ == '__main__':

    # set up costs, params, dists, etc.
    ...

    determine_EVPI(...)

    print(...)