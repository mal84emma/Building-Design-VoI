"""Helper functions for running experiments."""


def parse_experiment_args(expt_id, n_buildings, info_id):

    # Design constraints
    if expt_id == 0:
        expt_name = 'unconstr'
        sizing_constraints = {'battery':None,'solar':None}
    elif expt_id == 1:
        expt_name = 'constr_solar'
        sizing_constraints = {'battery':None,'solar':150.0}
    else:
        raise ValueError('Invalid run option for `expt_id`. Please provide valid CLI argument.')

    # Number of buildings
    assert n_buildings > 0, 'Invalid number of buildings. Please provide a positive integer.'

    # Information type
    if info_id == 0:
        info_type = 'type'
    elif info_id == 1:
        info_type = 'mean'
    elif info_id == 2:
        info_type = 'peak'
    elif info_id == 3:
        info_type = 'type+mean+peak'
    else:
        raise ValueError('Invalid run option for `info_id`. Please provide valid CLI argument.')

    return expt_name, sizing_constraints, info_type