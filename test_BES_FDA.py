# done so far: brew install octave; conda install oct2py

import oct2py
from oct2py import octave

import numpy as np
from utils.plotting import init_profile_fig, add_profile

import logging
logging.basicConfig(level=logging.DEBUG)
oc = oct2py.Oct2Py(logger=logging.getLogger(), convert_to_float=False, backend='qt')

octave.addpath(octave.genpath('FDA-for-BES'))

fig = init_profile_fig(y_titles={'y1':'Building energy (kWh)'})

for peak in [10.0,20.0,30.0]:
    Test2019,Test2020 = octave.feval('FDA-for-BES/DesignPlugLoads/FDASimulationP', 10.0, peak, 2, nout=2)
    load_profile = Test2019[:,0]
    print(np.mean(load_profile))
    print(np.max(load_profile))
    fig = add_profile(fig, load_profile, name=f'peak {peak}')

fig.show()

# Got to the point where I can generate plug load profiles
# Problem is the generated data doesn't look anything like the measured
# building data I use in the other part, and it's difficult to control the
# actual peak load and mean load.