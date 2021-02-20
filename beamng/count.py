import glob
raw_file_location       = "../../PhysicalCoverageData/beamng/raw/"
file_names = glob.glob(raw_file_location + "*.csv")
empty_file_count = 0
full_file_count = 0
for file_name in file_names:
    number_of_lines = len(open(file_name).readlines(  ))
    if number_of_lines <= 1:
        empty_file_count += 1
    else:
        full_file_count += 1
        print("Lidar scans in file: " +str(number_of_lines) + " - estimated duration: " + str(number_of_lines / 2.0) + "s")

print("-----------")
print("total files: " + str(len(file_names)))
print("Empty: " + str(empty_file_count))
print("Full: " + str(full_file_count))