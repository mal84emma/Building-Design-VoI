# Quantifying the benefit of load uncertainty reduction for the design of district energy systems under grid constraints using the Value of Information

This repository supports the article 'Quantifying the benefit of load uncertainty reduction for the design of district energy systems under grid constraints using the Value of Information', which is available online at [https://arxiv.org/abs/2412.16105](https://arxiv.org/abs/2412.16105).
It provides the code and data used to perform the numerical experiments in the paper.

## Technical requirements

A suitable environment for running this code can be initialised using [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#) as follows:

```
conda create --name myenv python=3.9
conda activate myenv
pip install -r requirements.txt
conda install -c conda-forge cvxpy==1.5.1 cmdstanpy==1.2.4
```

Note:
- `cvxpy` and `cmdstanpy` need to be installed from conda-forge to get the link C++ backends to install (compile) properly.
- before running any experiments, the script `prob_models.py` needs to be run independently to generate the required Stan model binaries.

## Solver license files

The experiments in the paper used the Gurobi solver for speed, using a WLS license. However, the code also supports the use of the open-source solver [HiGHS](https://highs.dev/).

You can get a free academic WLS license for Gurobi [here](https://www.gurobi.com/academia/academic-program-and-licenses/). Update the template license file `resources/gurobi.lic` to enable the Gurobi solver. To prevent your `gurobi.lic` file from being tracked by the repository, use the command 'git update-index --assume-unchanged FILE_NAME'.

## Running experiments

Run all experiment and analysis scripts should be run from root dir using the syntax,

```
python -m {experiments/analysis}.{fname} {options}
```

There are 4 experiment scripts that should be run in the following order:
1. `generate_scenarios` - samples building load scenarios from the prior and posterior distributions for a specified number of buildings and saves them to file
2. `find_and_eval_prior_systems` - optimises and simulates the energy system over the prior building load distribution for all system setups, for a specified number of buildings
3. `find_posterior_solns` - optimises the energy system over the posterior building load distribution, for a specified system setup, number of buildings, and information level for the posterior
4. `eval_posterior_solns` - simulates the energy system over the posterior building load distribution, for a specified system setup, number of buildings, and information level for the posterior
See `experiments.py` for the definition of the available system setups and information levels.