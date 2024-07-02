"""Build schema for a specified system design/realisation."""

import os
import csv
import json
from pathlib import Path
from typing import List, Union

import pandas as pd
from utils import scale_profile


def generate_temp_building_files(
        scenario_vector,
        data_dir_path: Union[str, Path],
        building_file_pattern: str,
        scenario_no: int,
        process_id: int = None
    ):
    """Generate temporary building files for a scenario if required
    (i.e. if profile scaling is needed).

    Args:
        scenario_vector (List): Nx2/4 vector defining scenarios. I.e. vector
            vector containing (building_id, year / mean, peak) tuples for each
            building in each scenario.
        data_dir_path (str or Path): Path to directory containing Citylearn compatible
            data.
        building_file_pattern (fstr): Pattern of building load data files. Must
            contain {id} and {year} placeholders.
        scenario_no (int): Scenario number of file identification
        process_id (int, optional): Process id for file identification. Defaults to None.
    
    Returns:
        List: List of paths to (temporary) building files for each building in scenario.
            Used to construct schema and for file cleanup.
    """

    if scenario_vector.shape[1] == 2: # no file generation needed
        load_files = [building_file_pattern.format(id=int(b),year=int(y)) for b,y in scenario_vector]

    elif scenario_vector.shape[1] == 4: # generate temporary files
        load_files = []

        for i,b_tup in enumerate(scenario_vector):
            building_id, year, mean, peak = b_tup

            temp_file = f'temp-prof_SB{i}_s{scenario_no}.csv'
            if process_id is not None:
                temp_file = f'p{process_id}_' + temp_file
            load_files.append(temp_file) # generate temp file name

            # grab original data, scale, and save to temp file
            original_building_file = pd.read_csv(os.path.join(data_dir_path, building_file_pattern.format(id=int(building_id), year=int(year))))
            profile = original_building_file['Equipment Electric Power [kWh]'].to_numpy()
            scaled_profile = scale_profile(profile, mean, peak)

            new_building_file = original_building_file.copy()
            new_building_file['Equipment Electric Power [kWh]'] = scaled_profile
            new_building_file.to_csv(os.path.join(data_dir_path,temp_file), index=False)

    return load_files


def build_schema(
        data_dir_path: Union[str, Path],
        building_names: List[str],
        load_data_paths: List[Union[str, Path]],
        weather_data_path: Union[str, Path],
        carbon_intensity_data_path: Union[str, Path],
        pricing_data_path: Union[str, Path],
        battery_efficiencies: List[float],
        schema_name: str = 'schema_temp',
        battery_energy_capacities: List[float] = None,
        battery_power_capacities: List[float] = None,
        pv_power_capacities: List[float] = None,
        simulation_duration: int = None
    ) -> Union[str, Path]:
    """Construct a schema.json for the specified system design/realisation
    and save to file.

    Note:
        - all data file paths must be specified relative to the schema
    location, so the schema must be saved to the dir containing the data.
        - the building names must be consistent between scnearios, i.e.
        all CityLearnEnv objects must use same building names for identification.

    Args:
        data_dir_path (Union[str, Path]): Path to directory containing data files.
        building_names (List[str]): List of building names in system.
        load_data_paths (List[Union[str, Path]]): List of paths to
            building data CSV for each building.
        weather_data_path (Union[str, Path]): Path to weather data CSV.
        carbon_intensity_data_path (Union[str, Path]): Path to carbon
            intensity data CSV.
        pricing_data_path (Union[str, Path]): Path to pricing data CSV.
        schema_name (str, optional): Name to be used for output schema file. Defaults
            to schema_temp.
        battery_efficiencies (List[float]): List of (round-trip)
            efficiencies of batteries in system.
        battery_energy_capacities (List[float], optional): List of energy
            capacities of batteries in system.
        battery_power_capacities (List[float], optional): List of power
            capacities of batteries in system.
        pv_power_capacities (List[float], optional): List of solar panel
            power capacities in system.
        simulation_duration (int, optional): Number of data time instances
            from simulation data to be used. Max is total number of data
            points in CSVs. Defaults to None.

    Returns:
        Union[str, Path]: Full path to created schema file.
    """

    # load base schema
    with open(os.path.join('resources','base_schema.json')) as base_schema:
        schema = json.load(base_schema)

    # if capacities not specified, set to 100 (not used by LP, but choose reasonable scale just in case)
    if battery_energy_capacities is None:
        battery_energy_capacities = [100]*len(building_names)
    if battery_power_capacities is None:
        battery_power_capacities = [100]*len(building_names)
    if pv_power_capacities is None:
        pv_power_capacities = [100]*len(building_names)

    if simulation_duration is None:
        with open(os.path.join(data_dir_path,pricing_data_path)) as file:
            reader_file = csv.reader(file)
            simulation_duration = len(list(reader_file))-1 # skip header row

    schema['simulation_end_time_step'] = simulation_duration - 1 # set length of simulation

    # write building attributes
    schema['buildings'] = {}
    for i,b_name in enumerate(building_names):

        building_dict = {
            'include': True,
            'energy_simulation': load_data_paths[i],
            'weather': weather_data_path,
            'carbon_intensity': carbon_intensity_data_path,
            'pricing': pricing_data_path,
            'inactive_observations': [],
            'inactive_actions': [],

            'electrical_storage': {
                'type': "citylearn.energy_model.Battery",
                'autosize': False,
                'attributes': {
                        'capacity': battery_energy_capacities[i],
                        'efficiency': battery_efficiencies[i],
                        'nominal_power': battery_power_capacities[i],
                        'capacity_loss_coefficient': 1e-05,
                        'loss_coefficient': 0,
                        'power_efficiency_curve': [[0,battery_efficiencies[i]],[1,battery_efficiencies[i]]],
                        'capacity_power_curve': [[0,1],[1,1]]
                }
            },

            'pv': {
                'type': "citylearn.energy_model.PV",
                'autosize': False,
                'attributes': {'nominal_power': pv_power_capacities[i]}
            }
        }

        schema['buildings'].update({b_name: building_dict})

    # write schema to file
    schema_path = os.path.join(data_dir_path,'%s.json'%schema_name)
    with open(schema_path, 'w') as file:
        json.dump(schema, file, indent=4)

    return schema_path


if __name__ == '__main__':

    building_year_pairs = [(0,2012),(19,2015),(40,2016)]

    b_names = [f'ly_{b}-{y}' for b,y in building_year_pairs]
    base_kwargs = {
        'data_dir_path': os.path.join('data','processed'),
        'building_names': b_names,
        'load_data_paths': [name + '.csv' for name in b_names],
        'weather_data_path': 'weather.csv',
        'carbon_intensity_data_path': 'carbon_intensity.csv',
        'pricing_data_path': 'pricing.csv',
        'schema_name': 'schema_build_test',
        'battery_efficiencies': [0.95,0.95,0.95],
        'battery_energy_capacities': None,
        'battery_power_capacities': None,
        'pv_power_capacities': None
    }

    schema_path = build_schema(**base_kwargs)