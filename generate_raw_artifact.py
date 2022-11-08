import os
import glob
import shutil
import numpy as np

number_samples = 10
 
# Check if the artifact already exists
path = '/home/carl/Desktop/PhysicalCoverageDataSubSet'
already_exists = os.path.exists(path)
if already_exists:
    print("Artifact already exists. Please remove from desktop")
    exit()

# Otherwise make the directory
print("Creating directory")
new_paths = ['/home/carl/Desktop/PhysicalCoverageDataSubSet',
             '/home/carl/Desktop/PhysicalCoverageDataSubSet/beamng',
             '/home/carl/Desktop/PhysicalCoverageDataSubSet/highway',
             '/home/carl/Desktop/PhysicalCoverageDataSubSet/beamng/feasibility',
             '/home/carl/Desktop/PhysicalCoverageDataSubSet/highway/feasibility',
             '/home/carl/Desktop/PhysicalCoverageDataSubSet/beamng/random_tests/',
             '/home/carl/Desktop/PhysicalCoverageDataSubSet/highway/random_tests/',
             '/home/carl/Desktop/PhysicalCoverageDataSubSet/beamng/random_tests/code_coverage', 
             '/home/carl/Desktop/PhysicalCoverageDataSubSet/highway/random_tests/code_coverage',
             '/home/carl/Desktop/PhysicalCoverageDataSubSet/beamng/random_tests/code_coverage/raw', 
             '/home/carl/Desktop/PhysicalCoverageDataSubSet/highway/random_tests/code_coverage/raw',
             '/home/carl/Desktop/PhysicalCoverageDataSubSet/beamng/random_tests/physical_coverage',
             '/home/carl/Desktop/PhysicalCoverageDataSubSet/highway/random_tests/physical_coverage',
             '/home/carl/Desktop/PhysicalCoverageDataSubSet/beamng/random_tests/physical_coverage/lidar',
             '/home/carl/Desktop/PhysicalCoverageDataSubSet/highway/random_tests/physical_coverage/raw']
for p in new_paths:
    os.mkdir(p)

# Create the subdirectories
for i in range(10):
    external_vehicles = i + 1

    paths = ["/home/carl/Desktop/PhysicalCoverageDataSubSet/highway/random_tests/code_coverage/raw",
             "/home/carl/Desktop/PhysicalCoverageDataSubSet/beamng/random_tests/code_coverage/raw",
             "/home/carl/Desktop/PhysicalCoverageDataSubSet/highway/random_tests/physical_coverage/raw",
             "/home/carl/Desktop/PhysicalCoverageDataSubSet/beamng/random_tests/physical_coverage/lidar"]

    for p in paths:
        new_path = "{}/{}_external_vehicles".format(p, external_vehicles)
        os.mkdir(new_path)


# Find the feasible tests
print("Copying over feasible tests")
shutil.copytree('/media/carl/DataDrive/PhysicalCoverageData/highway/feasibility/raw','/home/carl/Desktop/PhysicalCoverageDataSubSet/highway/feasibility/raw')

# Find all the raw code coverage tests
beamng_code_coverage = '/media/carl/DataDrive/PhysicalCoverageData/beamng/random_tests/code_coverage/raw'
highway_code_coverage = '/media/carl/DataDrive/PhysicalCoverageData/highway/random_tests/code_coverage/raw'
beamng_physical_coverage = '/media/carl/DataDrive/PhysicalCoverageData/beamng/random_tests/physical_coverage/lidar'
highway_physical_coverage = '/media/carl/DataDrive/PhysicalCoverageData/highway/random_tests/physical_coverage/raw'

print("Copying raw files")
for i in range(10):
    external_vehicles = i + 1
    
    # Get the code coverage path
    beamng_cc_path     = "{}/{}_external_vehicles".format(beamng_code_coverage, external_vehicles)
    highway_cc_path    = "{}/{}_external_vehicles".format(highway_code_coverage, external_vehicles)

    # Get the physical coverage path
    beamng_pc_path     = "{}/{}_external_vehicles".format(beamng_physical_coverage, external_vehicles)
    highway_pc_path    = "{}/{}_external_vehicles".format(highway_physical_coverage, external_vehicles)
   
    # Get the list of code coverage files
    beamng_cc_files    = np.array(sorted(glob.glob(beamng_cc_path + "/*.txt")))
    highway_cc_files   = np.array(sorted(glob.glob(highway_cc_path + "/*.xml")))

    # Get the list of physical coverage files
    beamng_pc_files    = np.array(sorted(glob.glob(beamng_pc_path + "/*.csv")))
    highway_pc_files   = np.array(sorted(glob.glob(highway_pc_path + "/*.txt")))

    # Make sure you find the same number of files
    assert(len(beamng_cc_files) == len(beamng_pc_files))
    assert(len(highway_cc_files) == len(highway_pc_files))
    assert(len(beamng_pc_files) > 1)
    assert(len(highway_pc_files) > 1)

    # Generate random indicies
    local_state     = np.random.RandomState()
    highway_indices = local_state.choice(len(highway_pc_files), number_samples, replace=False) 
    beamng_indices  = local_state.choice(len(beamng_pc_files), number_samples, replace=False) 

    # Copy these files over
    for index in highway_indices:
        # Make sure they are the same file
        highway_cc_file_name = os.path.basename(highway_cc_files[index])
        highway_pc_file_name = os.path.basename(highway_pc_files[index])
        assert(highway_cc_file_name[:highway_cc_file_name.find(".")] == highway_pc_file_name[:highway_pc_file_name.find(".")])
        shutil.copy(highway_cc_files[index], "/home/carl/Desktop/PhysicalCoverageDataSubSet/highway/random_tests/code_coverage/raw/{}_external_vehicles/{}".format(external_vehicles, highway_cc_file_name))
        shutil.copy(highway_pc_files[index], "/home/carl/Desktop/PhysicalCoverageDataSubSet/highway/random_tests/physical_coverage/raw/{}_external_vehicles/{}".format(external_vehicles, highway_pc_file_name))

    # Copy these files over
    for index in beamng_indices:
        # Make sure they are the same file
        beamng_cc_file_name = os.path.basename(beamng_cc_files[index])
        beamng_pc_file_name = os.path.basename(beamng_pc_files[index])
        assert(beamng_cc_file_name[:beamng_cc_file_name.find(".")] == beamng_pc_file_name[:beamng_pc_file_name.find(".")])
        shutil.copy(beamng_cc_files[index], "/home/carl/Desktop/PhysicalCoverageDataSubSet/beamng/random_tests/code_coverage/raw/{}_external_vehicles/{}".format(external_vehicles, beamng_cc_file_name))
        shutil.copy(beamng_pc_files[index], "/home/carl/Desktop/PhysicalCoverageDataSubSet/beamng/random_tests/physical_coverage/lidar/{}_external_vehicles/{}".format(external_vehicles, beamng_pc_file_name))

