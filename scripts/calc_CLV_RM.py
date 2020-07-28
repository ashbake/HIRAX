##############################################################
# Take exoplanet atmospheric spectrum and stellar info and 
# output HIRAX observation
###############################################################


import numpy as np
import matplotlib.pylab as plt
from astropy import units as u

import batman

plt.ion()

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 18}

matplotlib.rc('font', **font)

os.system('mkdir output') # make folder to save all things made here

data_dir = '../data/'


