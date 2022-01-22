def get_beam_number_from_file(files):
    beam_numbers = []
    for f in files:
        f_name = f[f.rfind("/"):]
        if "random" in f_name:
            f_name = f_name[f_name.find("_random")+7:]
        elif "generated" in f_name:
            f_name = f_name[f_name.find("_generated")+10:]
        beam = f_name[f_name.find("_b")+2:-4]
        if beam.find("_") != -1:
            beam = beam[:beam.find("_")]
        beam_numbers.append(int(beam))
    return beam_numbers

def order_files_by_beam_number(files, beam_number):
    resultant_files = []
    # Go through each beam
    for b in beam_number:
        found = False
        for f in files:
            if "_b{}_".format(b) in f or "_b{}.".format(b) in f:
                found = True
                resultant_files.append(f)
        if not found:
            resultant_files.append("")
    
    return resultant_files

def get_filename_prefix(filename):
    if filename == "car_controller.py":
        return 10000
    elif filename == "controller.py":
        return 20000
    elif filename == "ai.lua":
        return 30000
    else:
        print("Error! File not known.")
        exit()

