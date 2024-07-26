NOTE for setup instructions, cvxpy often needs to be conda installed via conda-forge to get the C++ based solvers to install (compile) properly.

Using `python=3.9`

`conda install -c conda-forge cmdstanpy`
It also appears that you have to run `prob_models.py` first to generate the Stan model binaries before they can be called in more complex scripts (e.g. while using concurrency).

To prevent `gurobi.lic` file from being uploaded to repo, using command 'git update-index --assume-unchanged FILE_NAME' to prevent tracking of changes to the file.