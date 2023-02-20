import os
import sys
import mmap
import random
import argparse

from time import sleep

import numpy as np

from beamngpy import BeamNGpy
from beamngpy import Scenario
from beamngpy import Vehicle
from beamngpy import set_up_simple_logging
from beamngpy.sensors import Lidar


def main():

    # Get the user defined arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=64256, help="The port used to connect.")
    parser.add_argument('--workspace', type=str, default="BeamNGWorkspace", help="The name of the workspace folder")
    args = parser.parse_args()

    # Get BeamNG's location and workspace
    current_user = os.environ.get("USERNAME")
    bng_loc = 'C:\\Users\\{}\\BeamNG\\BeamNG.tech-0.23.5.1'.format(current_user)
    bng_usr = 'C:\\Users\\{}\\BeamNG\\{}'.format(current_user, args.workspace)

    # Setup logging and random seed
    random.seed(1703)
    set_up_simple_logging()

    # Start BeamNG
    beamng = BeamNGpy('localhost', args.port, home=bng_loc, user=bng_usr)
    bng = beamng.open(launch=True)

    # Create the scenario
    scenario = Scenario('west_coast_usa', 'lidar_demo', description='Spanning the map with a lidar sensor')

    # Create the vehicle and attach a sensor
    vehicle = Vehicle('ego_vehicle', model='etk800', licence='RED', color='Red')
    lidar = Lidar(offset=(0, 0, 1.6))
    vehicle.attach_sensor('lidar', lidar)

    # Add the vehicle to the scenario
    scenario.add_vehicle(vehicle, pos=(-852.024 , -513.641, 106.620), rot=None, rot_quat=(0, 0, 0.926127, -0.377211))

    # Create the scenario in BeamNG
    scenario.make(bng)
    
    try:

        # Set simulator to be deterministic
        bng.set_deterministic() 
        # With 60hz temporal resolution
        bng.set_steps_per_second(60)  
        # Load the simulator
        bng.load_scenario(scenario)
        # Hide the HUD and start
        bng.hide_hud()
        bng.start_scenario()
        # Use the span AI mode
        vehicle.ai_set_mode('span')

        # Do this for 60 seconds
        count = 0
        print('Driving around for 60 seconds...')
        while count < 60:
            # Poll the sensors to update the lidar
            vehicle.poll_sensors()
            # Get the lidar data
            data_points = lidar.data["points"]
            # Display the lidar data
            print("Getting {} lidar points".format(np.shape(data_points)))
            sleep(1)

    finally:
        # Close BeamNG
        bng.close()

if __name__ == '__main__':
    main()