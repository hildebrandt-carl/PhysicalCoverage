def get_beam_numbers(files):
    beam_numbers = []
    for f in files:
        f_name = f[f.rfind("/"):]
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