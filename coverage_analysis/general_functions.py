def get_beam_numbers(files):

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

def order_by_beam(files, beam_number):
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

def get_ignored_code_coverage_lines(scenario):
    if scenario == "highway":
        ignored_lines = set([1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1016, 1018, 1034, 1067, 1068, 1069, 1070, 1071, 1072, 1073, 1074, 1075, 1076, 1077, 1078, 1079, 1080, 1081, 1082, 1083, 1084, 1085, 1086, 1087, 1088, 1089, 1090, 1091, 1092, 2088])
    elif scenario == "beamng":
        ignored_lines = set([680, 681, 682, 683, 684, 685, 687, 688, 689, 690, 658, 659, 660, 661, 662, 663, 664, 665, 666, 667, 668,592,593,594,595,596,597,598,599,601,602,603,604,605,606,607,608,609,610,611,612,613,614,615,616,617,618,619,620,621,622,623,624,625,626,627,628,629,630,631,632,633,634,635,636,637,638,639,640,641,642,643,644,645,646,647,648,649,650])
    return ignored_lines
