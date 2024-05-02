"""
Implementation of Linear Programming controller class for CityLearn model.
** Adapted from Annex 37 implementation. **

LinProgModel class is used to construct, hold, and solve LP models of the
CityLearn environment for use with either a Linear MPC controller, or as
an asset capcity design task (scenario optimisation supported).

Order of operations:
- Initalise model object with first scenario `env`
- Load additional scenario `envs`
- Set tau (for control use)
- Load data (for all scenarios)
- Generate LP
- Solve
"""

from citylearn.citylearn import CityLearnEnv
import numpy as np
import cvxpy as cp
from pathlib import Path
from typing import Any, List, Dict, Mapping, Tuple, Union


class LinProgModel():

    def __init__(self, schema: Union[str, Path] = None , env: CityLearnEnv = None) -> None:
        """Set up CityLearn environment from provided initial schema, and collected required data.

        NOTE: it is assumed all data is clean and appropriately formatted.
        NOTE: all further CityLearnEnvs/schemas must match the properties of the initial
        env used during setup (buildings and timings).

        Args:
            schema (Union[str, Path]): path to schema.json defining model setup
            env (CityLearnEnv): pre-constructred environment object to use
        """

        self.envs = []

        env = self.add_env(schema,env)

        self.b_names = [b.name for b in env.buildings]
        self.Tmax = env.time_steps # number of timesteps available
        self.delta_t = env.seconds_per_time_step/3600


    def add_env(self, schema: Union[str, Path] = None , env: CityLearnEnv = None) -> None:
        """Add a new CityLearnEnv object to the LP model - respresenting a new scenario.

        Args:
            schema (Union[str, Path]): path to schema.json defining model setup
            env (CityLearnEnv): pre-constructred environment object to use
        """

        if schema is None and env is None:
            raise ValueError("Must provide either a schema or a CityLearnEnv object.")
        if schema is not None and env is not None:
            raise ValueError("Cannot provide both a schema and a CityLearnEnv object.")

        if schema is not None:
            env = CityLearnEnv(schema)

        if len(self.envs) > 0:
            assert [b.name for b in env.buildings] == [b.name for b in self.envs[0].buildings]
            assert env.time_steps == self.envs[0].time_steps
            assert env.seconds_per_time_step == self.envs[0].seconds_per_time_step

            # NOTE: all schemas must have the same asset capacities
            assert [b.electrical_storage.capacity_history[0] for b in env.buildings] == [b.electrical_storage.capacity_history[0] for b in self.envs[0].buildings]
            assert [b.pv.nominal_power for b in env.buildings] == [b.pv.nominal_power for b in self.envs[0].buildings]

        self.envs.append(env)

        return env


    def set_time_data_from_envs(self, tau: int = None, t_start: int = None,
        current_socs: np.array = None) -> None:
        """Set time variant data for model from data given by CityLearnEnv objects for period
        [t_start+1, t_start+tau] (inclusive).

        Note: this corresponds to using perfect data for the prediction model of the system
        in a state at time t, with planning horizon tau. `current_socs` values are the
        state-of-charge at the batteries at the beginning of time period t.

        Args:
            tau (int, optional): number of time instances included in LP model. Defaults to None.
            t_start (int, optional): starting time index for LP model. Defaults to None.
            current_socs (np.array, optional): initial states of charge of batteries in
                period before t_start (kWh). Defaults to None.
        """

        if not t_start: self.t_start = 0
        else: self.t_start = t_start

        if not hasattr(self, 'tau'):
            if tau is None: self.tau = (self.Tmax - 1) - self.t_start
            else: self.tau = tau
        else:
            if (tau is not None) and (tau != self.tau): raise ValueError(f"Arugment `tau`={tau} does not match `self.tau`={self.tau} already set.")
        if self.tau > (self.Tmax - 1) - self.t_start: raise ValueError("`tau` cannot be greater than remaining time instances, (Tmax - 1) - t_start.")

        # initialise battery state for period before t_start
        if current_socs is not None:
            self.battery_initial_socs = current_socs
        else: # note this will default to zero if not specified in schema
            self.battery_initial_socs = np.array([[b.electrical_storage.initial_soc\
                for b in env.buildings] for env in self.envs])

        self.elec_loads = np.array(
            [[b.energy_simulation.non_shiftable_load[self.t_start+1:self.t_start+self.tau+1]\
                for b in env.buildings] for env in self.envs])
        self.solar_gens = np.array(
            [[b.pv.get_generation(b.energy_simulation.solar_generation)[self.t_start+1:self.t_start+self.tau+1]\
                for b in env.buildings] for env in self.envs])
        self.prices = np.array(
            [env.buildings[0].pricing.electricity_pricing[self.t_start+1:self.t_start+self.tau+1] for env in self.envs])
        self.carbon_intensities = np.array(
            [env.buildings[0].carbon_intensity.carbon_intensity[self.t_start+1:self.t_start+self.tau+1] for env in self.envs])


    def set_custom_time_data(self, elec_loads: np.array, solar_gens: np.array, prices: np.array,
        carbon_intensities: np.array, current_socs: np.array = None) -> None:
        """Set custom time variant data for each scenario to be used in model.

        This is used to load in forecast/prediction data in the LP model of the system for Linear MPC.

        Note: for a model prediction for the system in a state at time t, with planning horizon tau,
        the load, solar generation, pricing, and carbon intensity prediction values are for the period
        [t+1, t+tau], and the `current_socs` values are the state-of-charge values of the batteries at
        the start of time t.

        Args:
            elec_loads (np.array): electrical loads of buildings in each period (kWh) - shape (M,N,tau)
            solar_gens (np.array): energy generations of pv panels in each period (kWh) - shape (M,N,tau)
            prices (np.array): grid electricity price in each period ($/kWh) - shape (M,tau)
            carbon_intensities (np.array): grid electricity carbon intensity in each period (kgCO2/kWh) - shape (M,tau)
            current_socs (np.array, optional): initial states of charge of batteries in
                period before t_start (kWh) - shape (M,N). Defaults to None.
        """

        if not hasattr(self,'buildings'): raise NameError("Battery data must be contructed before providing time data.")

        assert elec_loads.shape[0] == solar_gens.shape[0] == prices.shape[0] == carbon_intensities.shape[0],\
            "Data provided must have consistent number of scenarios."
        assert elec_loads.shape[1] == solar_gens.shape[1] == len(self.buildings),\
            "Data must be provided for all buildings used in model."
        assert elec_loads.shape[2] == solar_gens.shape[2] == prices.shape[1] == carbon_intensities.shape[1],\
            "Data provided must have consistent time duration."

        if not hasattr(self,'tau'):
            self.tau = elec_loads.shape[2]
            assert self.tau > 0, "Must provide at least one period of data"
        else:
            assert elec_loads.shape[2] == self.tau, "Predicted time series must have length equal to specified planning horizon, tau."

        # initialise battery state for period before t_start
        if current_socs is not None:
            self.battery_initial_socs = current_socs
        else: # note this will default to zero if not specified in schema
            self.battery_initial_socs = np.array([[b.electrical_storage.initial_soc\
                for b in env.buildings] for env in self.envs])

        self.elec_loads = elec_loads
        self.solar_gens = solar_gens
        self.prices = prices
        self.carbon_intensities = carbon_intensities


    def generate_LP(self, clip_level: str = 'm',
                    pricing_dict: Dict[str,float] = {'carbon':5e-2,'battery':1e3,'solar':2e3},
                    opex_factor = 10.0,
                    design: bool = False,
                    scenario_weightings: List[float] = None
                    ) -> None:
        """Set up CVXPY LP of CityLearn model with data specified by stored `env`s, for
        desired buildings over specified time period.

        Note: we need to be extremely careful about the time indexing of the different variables (decision and data),
        see comments in implementation for details.

        Args:
            clip_level (Str, optional): str, either 'd' (district), 'b' (building), or 'm' (mixed),
                indicating the level at which to clip cost values in the objective function.
            pricing_dict (Dict[str,float], optional): dictionary containing pricing info for LP. Prices
                of carbon ($/kgCO2), battery capacity ($/kWh), solar capacity ($/kWp).
            opex_factor (float, optional): operational lifetime to consider OPEX costs over as factor
                of time duration considered in LP.
            design (Bool, optional): whether to construct the LP as a design problem - i.e. include
                asset capacities as decision variables
            scenario_weightings (List[float], optional): list of scenario OPEX weightings for objective
        """

        if not hasattr(self,'tau'): raise NameError("Planning horizon must be set before LP can be generated.")

        assert clip_level in ['d','b','m'], f"`clip_level` value must be either 'd' (district), 'b' (building), or 'm' (mixed), {clip_level} provided."

        assert all([type(val) == float for val in pricing_dict.values()])
        self.pricing_dict = pricing_dict
        assert opex_factor > 0

        self.design = design

        self.M = len(self.envs) # number of scenarios for optimisation
        self.N = len(self.envs[0].buildings) # number of buildings in model
        assert self.N > 0

        if scenario_weightings is not None:
            assert np.sum(scenario_weightings) == 1.0, "Scenario weightings must sum to 1."
        else:
            scenario_weightings = np.ones(self.M)/self.M


        # initialise decision variables
        self.SoC = {m: cp.Variable(shape=(self.N,self.tau), nonneg=True) for m in range(self.M)} # for [t+1,t+tau] - [kWh]
        self.battery_inflows = {m: cp.Variable(shape=(self.N,self.tau)) for m in range(self.M)} # for [t,t+tau-1] - [kWh]
        # NOTE: could create a stochastic control scheme by setting battery_inflows[:,0] (first control
        # action) equal for all scenarios - mutli-stage stochastic LP

        if clip_level == 'd':
            self.xi = {m: cp.Variable(shape=(self.tau), nonneg=True) for m in range(self.M)} # net power flow slack variable
        elif clip_level == 'b':
            self.bxi = {m: cp.Variable(shape=(self.N,self.tau), nonneg=True) for m in range(self.M)} # building level xi
        elif clip_level == 'm':
            self.xi = {m: cp.Variable(shape=(self.tau), nonneg=True) for m in range(self.M)} # net power flow slack variable
            self.bxi = {m: cp.Variable(shape=(self.N,self.tau), nonneg=True) for m in range(self.M)} # building level xi

        # initialise problem parameters
        self.current_socs = {m: cp.Parameter(shape=(self.N)) for m in range(self.M)}
        self.elec_loads_param = {m: cp.Parameter(shape=(self.N,self.tau)) for m in range(self.M)}
        self.solar_gens_param = {m: cp.Parameter(shape=(self.N,self.tau)) for m in range(self.M)}
        self.prices_param = {m: cp.Parameter(shape=(self.tau)) for m in range(self.M)}
        self.carbon_intensities_param = {m: cp.Parameter(shape=(self.tau)) for m in range(self.M)}

        # get battery data
        self.battery_efficiencies = np.array([[b.electrical_storage.efficiency\
            for b in env.buildings] for env in self.envs])
        self.battery_loss_coeffs = np.array([[b.electrical_storage.loss_coefficient\
            for b in env.buildings] for env in self.envs])
        self.battery_max_powers = np.array([[b.electrical_storage.available_nominal_power\
            for b in env.buildings] for env in self.envs])
        # TODO: add battery loss coefficient dynamics (self-discharge) to LP model

        # setup asset capacities & generation time series for design task
        if self.design:
            self.battery_capacities = cp.Variable(shape=(self.N), nonneg=True)
            self.rel_solar_capacities = cp.Variable(shape=(self.N), nonneg=True) # solar panel capacities as fraction of those specified in schemas
            #self.solar_gens_vals = {m: cp.multiply(cp.vstack([self.rel_solar_capacities]*self.tau).T,self.solar_gens_param[m]) for m in range(self.M)}
            self.solar_gens_vals = {m: cp.vstack([self.solar_gens_param[m][n,:]*self.rel_solar_capacities[n] for n in range(self.N)]) for m in range(self.M)}
            # NOTE: the mode shape for solar generation assumed depends on the solar capacity specified
            # in the schema due to the non-linearities of the panel model in CityLearn
        else:
            self.battery_capacities = np.array([b.electrical_storage.capacity_history[0]\
                for b in self.envs[0].buildings])
            # NOTE: batttery & solar capacities must be common to all scenarios
            self.solar_gens_vals = self.solar_gens_param


        # set up scenario constraints & objective contr.
        self.constraints = []
        self.e_grids = []
        self.building_power_flows = []
        self.scenario_objective_contributions = []

        for m in range(self.M): # for each scenario

            # initial storage dynamics constraint - for t=0
            self.constraints += [self.SoC[m][:,0] <= self.current_socs[m] +\
                cp.multiply(self.battery_inflows[m][:,0],\
                    np.sqrt(self.battery_efficiencies[m]))]
            self.constraints += [self.SoC[m][:,0] <= self.current_socs[m] +\
                cp.multiply(self.battery_inflows[m][:,0],\
                    1/np.sqrt(self.battery_efficiencies[m]))]

            # storage dynamics constraints - for t \in [t+1,t+tau-1]
            self.constraints += [self.SoC[m][:,1:] <= self.SoC[m][:,:-1] +\
                cp.multiply(self.battery_inflows[m][:,1:],\
                    np.tile((np.sqrt(self.battery_efficiencies[m]).reshape((self.N,1))),self.tau-1))]
            self.constraints += [self.SoC[m][:,1:] <= self.SoC[m][:,:-1] +\
                cp.multiply(self.battery_inflows[m][:,1:],\
                    np.tile((1/np.sqrt(self.battery_efficiencies[m]).reshape((self.N,1))),self.tau-1))]

            # storage power constraints - for t \in [t,t+tau-1]
            self.constraints += [-1*np.tile(self.battery_max_powers[m].reshape((self.N,1)),self.tau)*self.delta_t <=\
                self.battery_inflows[m]]
            self.constraints += [self.battery_inflows[m] <=\
                np.tile(self.battery_max_powers[m].reshape((self.N,1)),self.tau)*self.delta_t]

            # storage energy constraints - for t \in [t+1,t+tau]
            #self.constraints += [self.SoC[m] <= cp.vstack([self.battery_capacities]*self.tau).T]
            # NOTE: oddly for cvxpy the below is more vectorized and gives better compile times
            for n in range(self.N): self.constraints += [self.SoC[m][n,:] <= self.battery_capacities[n]]

            if clip_level == 'd':
                # aggregate costs at district level (CityLearn <= 1.6 objective)
                # costs are computed from clipped e_grids (net grid power flow) value - i.e. looking at portfolio elec. cost
                self.e_grids += [cp.sum(self.elec_loads_param[m] - self.solar_gens_vals[m] + self.battery_inflows[m], axis=0)] # for [t+1,t+tau]
                self.constraints += [self.xi[m] >= self.e_grids[m]] # for t \in [t+1,t+tau]

                self.scenario_objective_contributions.append([(self.xi[m] @ self.prices_param[m]),\
                    (self.xi[m] @ self.carbon_intensities_param[m]) * self.pricing_dict['carbon']])

            elif clip_level == 'b':
                # aggregate costs at building level and average (CityLearn >= 1.7 objective)
                # costs are computed from clipped building net power flow values - i.e. looking at mean building elec. cost
                self.building_power_flows += [self.elec_loads_param[m] - self.solar_gens_vals[m] + self.battery_inflows[m]] # for [t+1,t+tau]
                self.constraints += [self.bxi[m] >= self.building_power_flows[m]] # for t \in [t+1,t+tau]

                self.scenario_objective_contributions.append([(cp.sum(self.bxi[m], axis=0) @ self.prices_param[m]),\
                    (cp.sum(self.bxi[m], axis=0) @ self.carbon_intensities_param[m]) * self.pricing_dict['carbon']])

            elif clip_level == 'm':
                # aggregate electricity costs at building level (inflow metering only), but carbon emissions
                # costs at district level (estate level carbon reporting )
                # i.e. carbon credit system but no inter-building energy trading
                self.e_grids += [cp.sum(self.elec_loads_param[m] - self.solar_gens_vals[m] + self.battery_inflows[m], axis=0)] # for [t+1,t+tau]
                self.constraints += [self.xi[m] >= self.e_grids[m]] # for t \in [t+1,t+tau]
                self.building_power_flows += [self.elec_loads_param[m] - self.solar_gens_vals[m] + self.battery_inflows[m]] # for [t+1,t+tau]
                self.constraints += [self.bxi[m] >= self.building_power_flows[m]] # for t \in [t+1,t+tau]

                self.scenario_objective_contributions.append([(cp.sum(self.bxi[m], axis=0) @ self.prices_param[m]),\
                    (self.xi[m] @ self.carbon_intensities_param[m]) * self.pricing_dict['carbon']])

        # define overall objective
        self.objective_contributions = []

        self.objective_contributions += [scenario_weightings @ np.array([t[0] for t in self.scenario_objective_contributions])] # mean electricity price
        self.objective_contributions += [scenario_weightings @ np.array([t[1] for t in self.scenario_objective_contributions])] # mean carbon cost

        if self.design:
            self.objective_contributions = [contr*opex_factor for contr in self.objective_contributions] # extend opex costs to design lifetime
            self.objective_contributions += [cp.sum(self.battery_capacities) * self.pricing_dict['battery']] # battery CAPEX
            self.act_solar_capacities = cp.multiply(self.rel_solar_capacities, np.array([b.pv.nominal_power for b in self.envs[0].buildings]))
            self.objective_contributions += [cp.sum(self.act_solar_capacities) * self.pricing_dict['solar']] # solar CAPEX

        self.obj = cp.sum(self.objective_contributions)

        self.objective = cp.Minimize(self.obj)


        # construct problem
        self.problem = cp.Problem(self.objective,self.constraints)


    def set_LP_parameters(self):
        """Set value of CVXPY parameters using loaded data."""

        if not hasattr(self,'problem'): raise NameError("LP must be generated before parameters can be set.")
        if not hasattr(self,'elec_loads') or not hasattr(self,'solar_gens') or not hasattr(self,'prices')\
            or not hasattr(self,'carbon_intensities') or not hasattr(self,'battery_initial_socs'):
            raise NameError("Data must be loaded before parameters can be set.")

        # NOTE: clip parameter values at 0 to prevent LP solve issues
        # This requirement is for the current LP formulation and could be
        # relaxed with an alternative model setup.

        for m in range(self.M):
            self.current_socs[m].value = self.battery_initial_socs[m].clip(min=0)
            self.elec_loads_param[m].value = self.elec_loads[m].clip(min=0)
            self.solar_gens_param[m].value = self.solar_gens[m].clip(min=0)
            self.prices_param[m].value = self.prices[m].clip(min=0)
            self.carbon_intensities_param[m].value = self.carbon_intensities[m].clip(min=0)


    def solve_LP(self, **kwargs):
        """Solve LP model of specified problem.

        Args:
            **kwargs: optional keyword arguments for solver settings.

        Returns:
            results (Dict): formatted results dictionary with;
                - optimised objective
                - breakdown of objetive contributions
                - scenario opex costs (if appropriate)
                - optimised states-of-charge for batteries
                - optimised battery charging energy schedules
                - optimised battery energy capacities (if appropriate)
                - optimised solar panel power capacities (if appropriate)
        """

        if not hasattr(self,'problem'): raise ValueError("LP model has not been generated.")

        if 'solver' not in kwargs: kwargs['solver'] = 'SCIPY'
        if 'verbose' not in kwargs: kwargs['verbose'] = False
        if 'ignore_dpp' not in kwargs: kwargs['ignore_dpp'] = True
        if kwargs['solver'] == 'SCIPY': kwargs['scipy_options'] = {'method':'highs'}
        if kwargs['verbose'] == True: kwargs['scipy_options'].update({'disp':True})

        try:
            self.problem.solve(**kwargs)
        except cp.error.SolverError:
            print("Current SoCs: ", self.current_socs.value)
            print("Building loads:", self.elec_loads_param.value)
            print("Solar generations: ", self.solar_gens_vals.value)
            print("Pricing: ", self.prices_param.value)
            print("Carbon intensities: ", self.carbon_intensities_param.value)
            raise Exception("LP solver failed. Check your forecasts. Try solving in verbose mode. If issue persists please contact organizers.")

        # prepare results
        results = {
            'objective': self.objective.value,
            'objective_contrs': [val.value for val in self.objective_contributions],
            'scenario_contrs': [(t[0].value,t[1].value) for t in self.scenario_objective_contributions] if self.M > 1 else None,
            'SOC': {m: self.SoC[m].value for m in range(self.M)},
            'battery_inflows': {m: self.battery_inflows[m].value for m in range(self.M)},
            'battery_capacities': self.battery_capacities.value if self.design else None,
            'solar_capacities': self.act_solar_capacities.value if self.design else None
        }

        return results


    def get_LP_data(self, solver: str, **kwargs):
        """Get LP problem data used in CVXPY call to specified solver,
        as specified in https://www.cvxpy.org/api_reference/cvxpy.problems.html#cvxpy.Problem.get_problem_data

        Args:
            solver (str): desired solver.
            kwargs (dict): keywords arguments for cvxpy.Problem.get_problem_data().

        Returns:
            solver_data: data passed to solver in solve call, as specified in link to docs above.
        """

        if not hasattr(self,'problem'): raise NameError("LP model has not been generated.")

        return self.problem.get_problem_data(solver, **kwargs)