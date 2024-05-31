# Schedule for Experiment Scripts

Note:
- all results are logged to files in the `results` directory.
- `expt_config.py` contains configuration parameters used by all experiments.

## Experiment Scripts
- `generate_scenarios.py`, sample scenarios from prior distribution which are used by all other calculations requiring prior samples.
- `find_prior_soln.py` - solve Stochastic Program to determine prior optimal solution.
- `eval_prior_soln.py` - simulate system specified by prior design for scenarios sampled from prior distribution to evaluate expected utility (performance) of prior solution.
- `find_posterior_solns.py` - for prior scenario samples, sample from posterior solution, and solve Stochastic Programs to determine posterior optimal solutions for each sample.
- `eval_posterior_solns.py` - evaluate expected performance of posterior solutions (system designs) using samples from posterior distribution.
- `compute_voi.py` - load simulation results and compute Value of Information (VOI).
- `find_and_eval_partial_systems.py` - repeat system design and evaluation for the prior case, for the following scenarios:
    - system with constrained solar capacity (based on realistic roof size) - NOTE, also done using prior and posterior scripts
    - system with only battery (i.e. no solar)
    - system with only solar (i.e. no battery)
    - system with neither solar nor battery
- `quant_sys_benefit_contrs.py` - compute benefits derived from each part of the solar-battery system, i.e. load eval results for cases above and find differences

## Additional Scripts
- `analyse_peakiness.py` - plot load-duration curves for buildings with and without electrified heating to demonstrate increase in electricity usage variability caused by heat electrification.
- `analyse_pv.py` - compute capacity factors of PV generation data
- `plot_MC_conv.py` - plot convergence of Monte Carlo estimates of expected utilities to validate number of samples used in experiments.