

import numpy as np
from collections import defaultdict


def get_dimensions(fin):
    data_all = np.loadtxt(fin, delimiter=",", skiprows=2, usecols=range(1,10))

    dimensions = defaultdict(dict)
    areas = defaultdict(dict)

    for sample in range(1, 12):
        data = data_all[sample - 1]
        dimensions["FN"] = data[0]
        dimensions["FS"] = data[1]
        dimensions["FF"] = data[2]
        dimensions["SN"] = data[3]
        dimensions["SF"] = data[4]
        dimensions["SS"] = data[5]
        dimensions["NS"] = data[6]
        dimensions["NF"] = data[7]
        dimensions["NN"] = data[8]
        
        areas["FN"] = data[1]*data[2]
        areas["FS"] = data[0]*data[2]
        areas["FF"] = data[0]*data[1]
        
        areas["SN"] = data[4]*data[5]
        areas["SF"] = data[3]*data[5]
        areas["SS"] = data[3]*data[4]

        areas["NS"] = data[7]*data[8]
        areas["NF"] = data[6]*data[8]
        areas["NN"] = data[6]*data[7]

    return dimensions, areas

def load_experimental_data_stretch(fin, width, area):
    data = np.loadtxt(fin, delimiter=",", skiprows=1)

    displacement = data[:,0] * 1E-3 # m
    force = data[:,1]               # N

    # approximately right
    width *= 1E-3             # m
    area  *= 1E-6             # m2

    stretch = displacement / width           # fraction
    load = force / area *1E-3                # kPa

    # only consider tensile displacement for now
    i = 0
    while displacement[i] < 0:
        i+=1
    
    return stretch[i:], load[i:]

def load_experimental_data_shear(fin, width, area):
    data = np.loadtxt(fin, delimiter=",", skiprows=1)

    displacement = data[:,0] * 1E-3 # m
    shear_force = data[:,1]               # N
    normal_force = data[:,2]               # N

    # approximately right
    width *= 1E-3             # m
    area  *= 1E-6             # m2

    stretch = displacement / width           # fraction
    normal_load = normal_force / area *1E-3                # kPa
    shear_load = shear_force / area *1E-3                # kPa

    # only consider tensile displacement for now
    i = 0
    while displacement[i] < 0:
        i+=1
    
    return stretch[i:], normal_load[i:], shear_load[i:]


