"""Compute EVPI for building energy system design via LPs."""

from tqdm import tqdm
from multiprocess import Pool

# NOTE: follow structure of EVPI function from VoI paper repo
# - compute actions using design_system
# - determine true cost of actions using evaluate_system
# - perform for both uncertain building type (ID, uniform prob.) and known building type

from energy_system import design_system, evaluate_system
from probs_models import prior_model, posterior_model


# aim is to provide really clear structure of EVPI calculations that is used by
# experiment scripts (for repeatability and self-docs)
def design_EVPI(
        building_ids,
        years,
        data_dir,
        building_file_pattern,
        n_buildings,
        n_scenarios,
        cost_dict,
        info_type='type', # alternative is 'profile'
        solver_kwargs=None,
        n_processes=None
    ):
    # handler function to bring together options for EVPI calc
    # i.e. set up sampling functions, etc.
    # provide options for:
    # - number of buildings
    # - number of scenarios, processes, etc.
    # - level of information for EVPI (i.e. building type vs exact profile)
    #   note for this, when there is exact profile info the posterior analysis simplifies

    ...

    return ...

def compute_EVPI(
        data_dir,
        building_file_pattern,
        cost_dict,
        prior_sampler,
        posterior_sampler,
        info_type='type',
        solver_kwargs=None,
        n_processes=None
    ): # pass kwargs through from handler fn
    # execution of EVPI calc
    # follow structre from VoI for building energy repo

    # Sample from prior distribution of uncertain parameter(s)
    # ========================================================
    ...

    # Perform Prior analysis
    # ======================
    ...

    # Perform Pre-Posterior analysis
    # ==============================
    ...
    # I think I can make this different pretty minimal and elegant
    # Just average over a single posterior sample for profile case?
    if info_type == 'type':
        ...
    elif info_type == 'profile':
        ...

    # Compute EVPI
    # ============
    ...

    return ...


if __name__ == '__main__':
    # give it a quick test run for both cases (i.e. few buildings & scenarios)

    # set up costs, params, dists, etc.
    ...

    determine_EVPI(...)

    print(...)