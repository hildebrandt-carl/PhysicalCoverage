from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Electrics
import time 

bng = BeamNGpy('localhost', 64256, home=r'C:\Program Files (x86)\Steam\steamapps\common\BeamNG.drive')
bng.open()

scenario = Scenario('west_coast_usa', 'example')

vehicle = Vehicle('ego_vehicle', model='etk800', licence='PYTHON')
electrics = Electrics()
vehicle.sensors.attach('electrics', electrics)

scenario.add_vehicle(vehicle, pos=(-852.024 , -513.641, 106.620), rot_quat=(0, 0, 0.3826834, 0.9238795))

scenario.make(bng)

bng.load_scenario(scenario)
bng.start_scenario()

my_file = open("right.txt","w+")
time.sleep(20)
print("Starting in 3")
time.sleep(1)
print("Starting in 2")
time.sleep(1)
print("Starting in 1")
time.sleep(1)
print("Go")
while True:
    time.sleep(0.1)
    vehicle.sensors.poll()
    my_file.write("{}\n".format(vehicle.state['pos']))

my_file.close()