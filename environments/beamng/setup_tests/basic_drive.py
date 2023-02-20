import os
import argparse

from beamngpy import BeamNGpy, Scenario, Vehicle

# Get the user defined arguments
parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, default=64256, help="The port used to connect.")
parser.add_argument('--workspace', type=str, default="BeamNGWorkspace", help="The name of the workspace folder")
args = parser.parse_args()

# Get BeamNG's location and workspace
current_user = os.environ.get("USERNAME")
bng_loc = 'C:\\Users\\{}\\BeamNG\\BeamNG.tech-0.23.5.1'.format(current_user)
bng_usr = 'C:\\Users\\{}\\BeamNG\\{}'.format(current_user, args.workspace)

# Launch BeamNG
bng = BeamNGpy('localhost', args.port, home=bng_loc, user=bng_usr)
bng.open()

# Create a scenario in west_coast_usa called 'example'
scenario = Scenario('west_coast_usa', 'example')

# Create an ETK800 with the license plate 'PYTHON'
vehicle = Vehicle('ego_vehicle', model='etk800', licence='PYTHON')

# Add it to our scenario at this position and rotation
scenario.add_vehicle(vehicle, pos=(-717, 101, 118), rot=None, rot_quat=(0, 0, 0.3826834, 0.9238795))

# Place files defining our scenario for the simulator to read
scenario.make(bng)

# Load and start our scenario
bng.load_scenario(scenario)
bng.start_scenario()

# Make the vehicle's AI span the map
vehicle.ai_set_mode('span')
input('Hit enter when done...')