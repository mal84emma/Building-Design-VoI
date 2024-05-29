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
        self.delta_t = env.seconds_per_time_step/3600 # hours per timestep


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
        initial_socs: np.array = None) -> None:
        """Set time variant data for model from data given by CityLearnEnv objects for period
        [t_start+1, t_start+tau] (inclusive).

        Note: this corresponds to using perfect data for the prediction model of the system
        in a state at time t, with planning horizon tau. `initial_socs` values are the
        state-of-charge at the batteries at the beginning of time period t.

        Args:
            tau (int, optional): number of time instances included in LP model. Defaults to None.
            t_start (int, optional): starting time index for LP model. Defaults to None.
            initial_socs (np.array, optional): initial states of charge of batteries in
                period before t_start (kWh). Defaults to None.
        """

        if t_start is None: self.t_start = 0
        else: self.t_start = t_start

        if not hasattr(self, 'tau'):
            if tau is None: self.tau = (self.Tmax - 1) - self.t_start
            else: self.tau = tau
        else:
            if (tau is not None) and (tau != self.tau): raise ValueError(f"Arugment `tau`={tau} does not match `self.tau`={self.tau} already set.")
        if self.tau > (self.Tmax - 1) - self.t_start: raise ValueError("`tau` cannot be greater than remaining time instances, (Tmax - 1) - t_start.")

        # initialise battery state for period before t_start
        if initial_socs is not None:
            self.battery_initial_socs = initial_socs
        else: # note this will default to zero if not specified in schema
            self.battery_initial_socs = np.array([[b.electrical_storage.initial_soc for b in env.buildings] for env in self.envs])

        self.elec_loads = np.array(
                [[b.energy_simulation.non_shiftable_load[self.t_start+1:self.t_start+self.tau+1] for b in env.buildings] for env in self.envs]
            )
        self.solar_gens = np.array( # NOTE: this is the NORMALISED solar generation (W/kWp) -> [kW/kWp]
                [[b.energy_simulation.solar_generation[self.t_start+1:self.t_start+self.tau+1]/1e3 for b in env.buildings] for env in self.envs]
            )
        self.prices = np.array(
                [env.buildings[0].pricing.electricity_pricing[self.t_start+1:self.t_start+self.tau+1] for env in self.envs]
            )
        self.carbon_intensities = np.array(
                [env.buildings[0].carbon_intensity.carbon_intensity[self.t_start+1:self.t_start+self.tau+1] for env in self.envs]
            )


    def generate_LP(self,
                    cost_dict: Dict[str,float],
                    clip_level: str = 'm',
                    design: bool = False,
                    scenario_weightings: List[float] = None,
                    sizing_constraints: Dict[str,float] = None,
                    grid_con_capacity: float = None,
                    use_parameters = False
                    ) -> None:
        """Set up CVXPY LP of CityLearn model with data specified by stored `env`s, for
        desired buildings over specified time period.

        Note: we need to be extremely careful about the time indexing of the different variables (decision and data),
        see comments in implementation for details.

        Args:
            cost_dict (Dict[str,float/int]): dictionary containing info for LP costs
                Key-value pairs are:
                    - carbon: carbon price ($/kgCO2)
                    - battery: battery capacity cost ($/kWh)
                    - solar: solar capacity cost ($/kWp)
                    - opex_factor: OPEX extension factor, ratio of system lifetime to simulated duration (float/int)
                    - grid_capacity: grid connection capacity cost ($/kW/day)
                    - grid_excess: grid exceed capacity usage cost ($/kW excess/day)
                    - battery_power_ratio: ratio of power capacity to energy capacity for batteries (float)
                    - grid_con_safety_factor: safety factor for grid connection capacity during design (float)
                    - cntrl_grid_cap_margin: margin on tracked max grid used used by control LP to prevent
                        drift of max usage (float)
            clip_level (Str, optional): str, either 'd' (district), 'b' (building), or 'm' (mixed),
                indicating the level at which to clip cost values in the objective function.
            design (Bool, optional): whether to construct the LP as a design problem - i.e. include
                asset capacities as decision variables
            scenario_weightings (List[float], optional): list of scenario OPEX weightings for objective.
            sizing_constraints (Dict[str,float], optional): dictionary containing constraints for asset sizes in design LP.
            grid_con_capacity (float, optional): grid connection capacity (kW) in system design. Used for
                system simulation only. Defaults to None.
            use_parameters (Bool, optional): whether to use CVXPY parameters for data or directly load data.
                NOTE: parameters should be used for control LP where problem of identical structure is solved repeatedly,
                and problem size is small. For design problem, load data directly to drastically reduce compile time.
        """

        if not hasattr(self,'tau'): raise NameError("Planning horizon must be set before LP can be generated.")

        assert clip_level in ['d','b','m'], f"`clip_level` value must be either 'd' (district), 'b' (building), or 'm' (mixed), {clip_level} provided."

        for key in ['carbon','battery','solar','grid_capacity','grid_excess','opex_factor','battery_power_ratio','grid_con_safety_factor','cntrl_grid_cap_margin']:
            assert key in cost_dict.keys(), f"Key {key} missing from cost_dict."
        assert all([type(val) in [int,float] for val in cost_dict.values()])
        assert cost_dict['opex_factor'] > 0, "opex_factor must be greater than 0."
        self.cost_dict = cost_dict

        self.design = design

        if sizing_constraints is not None:
            assert self.design == True, "Sizing constraints can only be applied to design LP."
            assert all([key in ['battery','solar'] for key in sizing_constraints.keys()]), "Sizing constraints must be provided for battery and solar."
            assert all([val > 0 for val in sizing_constraints.values() if val is not None]), "Sizing constraints must be greater than 0."
        self.sizing_constraints = sizing_constraints

        if not self.design:
            assert grid_con_capacity is not None, "Grid connection capacity must be provided for operational LP."
            assert grid_con_capacity > 0, "Grid connection capacity must be greater than 0."

        self.M = len(self.envs) # number of scenarios for optimisation
        self.N = len(self.envs[0].buildings) # number of buildings in model
        assert self.N > 0, "Must have at least one building."

        if scenario_weightings is not None:
            assert np.isclose(np.sum(scenario_weightings), 1.0), "Scenario weightings must sum to 1."
        else: # assume scenarios equally probable
            scenario_weightings = np.ones(self.M)/self.M


        # initialise decision variables
        # =============================
        self.SoC = {m: cp.Variable(shape=(self.N,self.tau), nonneg=True) for m in range(self.M)} # for [t+1,t+tau] - (kWh)
        self.battery_inflows = {m: cp.Variable(shape=(self.N,self.tau), nonneg=True) for m in range(self.M)} # for [t,t+tau-1] - (kWh)
        self.battery_outflows = {m: cp.Variable(shape=(self.N,self.tau), nonneg=True) for m in range(self.M)} # for [t,t+tau-1] - (kWh)
        # NOTE: cvxpy does not like pos(e)*eta - neg(e)/eta, so split positive and negative battery flows into separate decision variables

        if self.design:
            self.battery_capacities = cp.Variable(shape=(self.N,1), nonneg=True) # battery energy capacities (kWh)
            self.solar_capacities = cp.Variable(shape=(self.N,1), nonneg=True) # solar panel capacities (kWp)
            self.grid_con_capacity = cp.Variable(nonneg=True) # grid connection capacity (kW)
        else:
            self.battery_capacities = np.array([[b.electrical_storage.capacity_history[0]] for b in self.envs[0].buildings])
            self.solar_capacities = np.array([[b.pv.nominal_power] for b in self.envs[0].buildings])
            # NOTE: batttery & solar capacities must be common to all scenarios
            self.grid_con_capacity = grid_con_capacity
            self.max_grid_usage = cp.Parameter(nonneg=True) # maximum grid usage so far in simulation (kW)

        if clip_level in ['d','m']:
            self.xi = {m: cp.Variable(shape=(self.tau), nonneg=True) for m in range(self.M)} # grid energy flow slack variable
        if clip_level in ['b','m']:
            self.bxi = {m: cp.Variable(shape=(self.N,self.tau), nonneg=True) for m in range(self.M)} # building level xi
        # NOTE: have to use slack variables, as cp.pos() is not DPP compliant

        # initialise problem parameters (control) or directly load data (design)
        # ======================================================================
        if use_parameters:
            self.initial_socs = {m: cp.Parameter(shape=(self.N)) for m in range(self.M)}
            self.elec_loads_param = {m: cp.Parameter(shape=(self.N,self.tau)) for m in range(self.M)}
            self.solar_gens_param = {m: cp.Parameter(shape=(self.N,self.tau)) for m in range(self.M)}
            self.prices_param = {m: cp.Parameter(shape=(self.tau)) for m in range(self.M)}
            self.carbon_intensities_param = {m: cp.Parameter(shape=(self.tau)) for m in range(self.M)}
        else:
            self.initial_socs = {m: self.battery_initial_socs[m].clip(min=0) for m in range(self.M)}
            self.elec_loads_param = {m: self.elec_loads[m].clip(min=0) for m in range(self.M)}
            self.solar_gens_param = {m: self.solar_gens[m].clip(min=0) for m in range(self.M)}
            self.prices_param = {m: self.prices[m].clip(min=0) for m in range(self.M)}
            self.carbon_intensities_param = {m: self.carbon_intensities[m].clip(min=0) for m in range(self.M)}

        # get battery data
        self.battery_efficiencies = np.array([[[b.electrical_storage.efficiency] for b in env.buildings] for env in self.envs])
        self.battery_loss_coeffs = np.array([[[b.electrical_storage.loss_coefficient] for b in env.buildings] for env in self.envs])
        # TODO: add battery loss coefficient dynamics (self-discharge) to LP model

        # set battery power capacities based off energy capacities & discharge/volume ratio
        self.battery_max_powers = self.battery_capacities * self.cost_dict['battery_power_ratio']
        # define solar nominal power generation variables (kWh)
        self.solar_gens_vals = {m: cp.multiply(self.solar_gens_param[m],self.solar_capacities) for m in range(self.M)}


        self.constraints = []

        # set up asset sizing constraints
        # ===============================
        if self.sizing_constraints is not None:
            if self.sizing_constraints['battery'] is not None:
                self.constraints += [self.battery_capacities <= self.sizing_constraints['battery']]
            if self.sizing_constraints['solar'] is not None:
                self.constraints += [self.solar_capacities <= self.sizing_constraints['solar']]

        # set up scenario constraints & objective contributions
        # =====================================================
        self.e_grids = []
        self.building_power_flows = []
        self.scenario_objective_contributions = []

        for m in range(self.M): # for each scenario

            # initial storage dynamics constraint - for t=0
            self.constraints += [self.SoC[m][:,0] == self.initial_socs[m] +\
                cp.multiply(self.battery_inflows[m][:,0],\
                    np.sqrt(self.battery_efficiencies[m]).flatten()) -\
                cp.multiply(self.battery_outflows[m][:,0],\
                    1/np.sqrt(self.battery_efficiencies[m]).flatten())]

            # storage dynamics constraints - for t \in [t+1,t+tau-1]
            self.constraints += [self.SoC[m][:,1:] == self.SoC[m][:,:-1] +\
                cp.multiply(self.battery_inflows[m][:,1:],\
                    np.sqrt(self.battery_efficiencies[m])) -\
                cp.multiply(self.battery_outflows[m][:,1:],\
                    1/np.sqrt(self.battery_efficiencies[m]))]

            # storage power constraints - for t \in [t,t+tau-1]
            self.constraints += [self.battery_inflows[m] <= self.battery_max_powers*self.delta_t]
            self.constraints += [self.battery_outflows[m] <= self.battery_max_powers*self.delta_t]

            # storage energy constraints
            self.constraints += [self.SoC[m] <= self.battery_capacities]
            # NOTE: oddly cvxpy CAN provide lower compile times if constraints are defined per building

            # define grid energy flow variables (energy drawn from grid at each timestep)
            self.e_grids += [cp.sum(self.elec_loads_param[m] - self.solar_gens_vals[m] + self.battery_inflows[m] - self.battery_outflows[m], axis=0)] # for [t+1,t+tau]

            if clip_level == 'd':
                # aggregate costs at district level (CityLearn <= 1.6 objective)
                # costs are computed from clipped e_grids (net grid power flow) value - i.e. looking at portfolio elec. cost
                self.constraints += [self.xi[m] >= self.e_grids[m]] # for t \in [t+1,t+tau]
                self.scenario_objective_contributions.append([
                    (self.xi[m] @ self.prices_param[m]),
                    (self.xi[m] @ self.carbon_intensities_param[m]) * self.cost_dict['carbon']
                ])

            elif clip_level == 'b':
                # aggregate costs at building level and average (CityLearn >= 1.7 objective)
                # costs are computed from clipped building net power flow values - i.e. looking at mean building elec. cost
                self.building_power_flows += [self.elec_loads_param[m] - self.solar_gens_vals[m] + self.battery_inflows[m] - self.battery_outflows[m]] # for [t+1,t+tau]
                self.constraints += [self.bxi[m] >= self.building_power_flows[m]] # for t \in [t+1,t+tau]
                self.scenario_objective_contributions.append([
                    (cp.sum(self.bxi[m], axis=0) @ self.prices_param[m]),
                    (cp.sum(self.bxi[m], axis=0) @ self.carbon_intensities_param[m]) * self.cost_dict['carbon']
                ])

            elif clip_level == 'm':
                # aggregate electricity costs at building level (inflow metering only), but carbon emissions
                # costs at district level (estate level carbon reporting )
                # i.e. carbon credit system but no inter-building energy trading
                self.constraints += [self.xi[m] >= self.e_grids[m]] # for t \in [t+1,t+tau]
                self.building_power_flows += [self.elec_loads_param[m] - self.solar_gens_vals[m] + self.battery_inflows[m] - self.battery_outflows[m]] # for [t+1,t+tau]
                self.constraints += [self.bxi[m] >= self.building_power_flows[m]] # for t \in [t+1,t+tau]
                self.scenario_objective_contributions.append([
                    (cp.sum(self.bxi[m], axis=0) @ self.prices_param[m]),
                    (self.xi[m] @ self.carbon_intensities_param[m]) * self.cost_dict['carbon']
                ])

            # add grid capacity exceedance cost
            if design:
                threshold_capacity = self.grid_con_capacity/self.cost_dict['grid_con_safety_factor']
                billing_period = self.tau
            else:
                threshold_capacity = self.max_grid_usage*(1-self.cost_dict['cntrl_grid_cap_margin'])
                # NOTE: the safety margin on grid usage prevents drift of the max grid usage, and corrects the inability of the myopic finite
                # horizon LP to properly account for the long term excess usage cost. The margin is a heuristic that should be tuned, but its
                # use was found to substantially improve overall performance of the control scheme.
                billing_period = self.Tmax

            self.scenario_objective_contributions[-1].append(
                cp.maximum((cp.maximum(*self.e_grids[m],*(-1*self.e_grids[m]))/self.delta_t - threshold_capacity),0) *\
                    self.cost_dict['grid_excess'] * (billing_period * self.delta_t)/24
            )
            # NOTE
            # For the operational LP (design==False), we use the simulation duration Tmax and the current,
            # maximum grid usage to compute the grid exceedance cost
            # For the design LP (design==True), the planning horizon tau is the effective simulation duration,
            # and so is used to compute grid costs

        # define overall objective
        # ========================
        self.objective_contributions = []

        # add up scenario costs with weightings
        for k in range(len(self.scenario_objective_contributions[0])):
            self.objective_contributions += [scenario_weightings @ np.array([m[k] for m in self.scenario_objective_contributions])]

        if self.design: # extend operational costs to full lifetime and add asset costs
            self.objective_contributions = [contr*self.cost_dict['opex_factor'] for contr in self.objective_contributions] # extend opex costs to design lifetime
            self.objective_contributions += [self.grid_con_capacity * self.cost_dict['grid_capacity'] * (self.tau * self.delta_t)/24 * self.cost_dict['opex_factor']] # grid capacity OPEX
            self.objective_contributions += [cp.sum(self.battery_capacities) * self.cost_dict['battery']] # battery CAPEX
            self.objective_contributions += [cp.sum(self.solar_capacities) * self.cost_dict['solar']] # solar CAPEX

        self.obj = cp.sum(self.objective_contributions)
        self.objective = cp.Minimize(self.obj)

        # construct problem
        self.problem = cp.Problem(self.objective,self.constraints)


    def set_LP_parameters(self, max_grid_usage: float = None):
        """Set value of CVXPY parameters using loaded data.

        Argument `max_grid_usage` (kW) is required for operational LP (for control task),
        and keeps track of the maximum grid usage so far in the simulation to allow
        the true cost impact of grid exceedance to be calculated.
        """

        if not hasattr(self,'problem'): raise NameError("LP must be generated before parameters can be set.")
        if not hasattr(self,'elec_loads') or not hasattr(self,'solar_gens') or not hasattr(self,'prices')\
            or not hasattr(self,'carbon_intensities') or not hasattr(self,'battery_initial_socs'):
            raise NameError("Data must be loaded before parameters can be set.")

        if not self.design:
            assert max_grid_usage is not None, "max_grid_usage must be provided for operational LP."
            self.max_grid_usage.value = max_grid_usage

        # NOTE: clip parameter values at 0 to prevent LP solve issues
        # This requirement is for the current LP formulation (??) and could be
        # relaxed with an alternative model setup.

        for m in range(self.M):
            self.initial_socs[m].value = self.battery_initial_socs[m].clip(min=0)
            self.elec_loads_param[m].value = self.elec_loads[m].clip(min=0)
            self.solar_gens_param[m].value = self.solar_gens[m].clip(min=0)
            self.prices_param[m].value = self.prices[m].clip(min=0)
            self.carbon_intensities_param[m].value = self.carbon_intensities[m].clip(min=0)


    def solve_LP(self, return_profiles=False, **kwargs):
        """Solve LP model of specified problem.

        Args:
            return_profiles (bool): whether to return the optimised energy profiles
                for each debugging. Defaults to False.
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
        elif kwargs['solver'] == 'GUROBI':
            assert 'env' in kwargs, "Gurobi environment must be provided for GUROBI solver."
        if 'verbose' not in kwargs: kwargs['verbose'] = False
        if 'ignore_dpp' not in kwargs: kwargs['ignore_dpp'] = True
        if kwargs['solver'] == 'SCIPY': kwargs['scipy_options'] = {'method':'highs'}
        if kwargs['verbose'] == True:
            if kwargs['solver'] == 'SCIPY':
                kwargs['scipy_options'].update({'disp':True})
        else:
            if kwargs['solver'] == 'GUROBI':
                kwargs['env'].setParam('LogToConsole',0)

        try:
            self.problem.solve(**kwargs)
        except cp.error.SolverError:
            print("Current SoCs: ", self.initial_socs.value)
            print("Building loads:", self.elec_loads_param.value)
            print("Solar generations: ", self.solar_gens_vals.value)
            print("Pricing: ", self.prices_param.value)
            print("Carbon intensities: ", self.carbon_intensities_param.value)
            raise Exception("LP solver failed. Check your forecasts. Try solving in verbose mode. If issue persists please contact organizers.")

        # prepare results
        results = {
            'objective': self.objective.value,
            'objective_contrs': [val.value for val in self.objective_contributions],
            'scenario_contrs': [[val.value for val in m] for m in self.scenario_objective_contributions] if self.M > 1 else None,
            'battery_capacities': self.battery_capacities.value if self.design else None,
            'solar_capacities': self.solar_capacities.value if self.design else None,
            'grid_con_capacity': self.grid_con_capacity.value if self.design else None
        }

        if return_profiles:
            results.update({
                'SOC': {m: self.SoC[m].value for m in range(self.M)},
                'battery_net_in_flows': {m: self.battery_inflows[m].value - self.battery_outflows[m].value for m in range(self.M)},
                'e_grids': {m: self.e_grids[m].value for m in range(self.M)}
            })

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