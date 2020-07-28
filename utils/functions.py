##############################################################
# Integrate PBay Spectrum with actualy Alluxa profile to get Measurements
# To make spectrum, go to Kavli research folder where PBay is stored
# Outputs: Plots saved to plots folder
# Inputs: Spectra from pbay + their log should be saved to spectra folder
###############################################################

import numpy as np
import matplotlib.pylab as plt
from scipy.interpolate import interp1d
from scipy.integrate import trapz
import matplotlib
import os,sys
from numpy import float64,array
plt.ion()
from astropy.io import fits

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 18}

matplotlib.rc('font', **font)



def integrate(x,y):
    """
    Integrate y wrt x
    """
    return trapz(y,x=x)

def gaussian(x, shift, sig):
    ' Return normalized gaussian with mean shift and var = sig^2 '
    return np.exp(-.5*((x - shift)/sig)**2)/(sig * np.sqrt(2*np.pi))

def vac_to_stand(wave_vac):
    """Convert vacuum wavelength (Ang) to standard wavelength in air since we're
    doing ground based stuff. 

	https://idlastro.gsfc.nasa.gov/ftp/pro/astro/vactoair.pro
    Equation from Prieto 2011 Apogee technical note
    and equation and parametersfrom Cidor 1996
    
    wavelength units???? must be micron??"""
    # eqn
    sigma2= (1e4/wave_vac)**2.
    fact = 1. +  5.792105e-2/(238.0185 - sigma2) + \
                            1.67917e-3/( 57.362 - sigma2)
                            
    # return l0 / n which equals lamda
    return wave_vac/fact 


def calc_signal(vels,rad,x,y,radlow=6371,specrad0 =0.2):
	"""
	Integrate over transit spectrum for velocity array, vels,
	and stellar radius, rad. 
	
	Returns signal in ppm per vel
	
	specrad0=0.2 for when the transit spec was created assuming r_star =0.2r_sun
	units of transit spectrum should be (R_p/R_s)^2
	radlow is lower radius to take (like in refraction case)
	"""
	rsun   = 695508.0
	
	onsignal_vel = np.zeros(len(vels))
	offsignal_vel = np.zeros(len(vels))

	iflatten    = np.where(np.sqrt(y*(0.2*rsun)**2) < radlow)[0]
	ytemp       = 1.0*y
	ytemp[iflatten] = radlow**2/(0.2*rsun)**2

	for i, vel in enumerate(vels):
		x_temp = x + x*(1.0*vel/300000.0)
		interpx_shift  = interp1d(x_temp,ytemp,bounds_error=False,fill_value=0)
		y_shift = interpx_shift(x) * (specrad0/rad)**2
		onsignal_vel[i] = integrate(x,(onband*(1-y_shift)))
		offsignal_vel[i]= integrate(x,(offband*(1-y_shift)))

	rat_bands = integrate(x,(onband)) / integrate(x,(offband))
	
	return (rat_bands-onsignal_vel/offsignal_vel)*1e6


def read_hitran2012_parfile(filename):
    '''
    Given a HITRAN2012-format text file, read in the parameters of the molecular absorption features.
    Parameters
    ----------
    filename : str
        The filename to read in.
    Return
    ------
    data : dict
        The dictionary of HITRAN data for the molecule.
    '''

    if not os.path.exists:
        raise ImportError('The input filename"' + filename + '" does not exist.')

    if filename.endswith('.zip'):
        import zipfile
        zip = zipfile.ZipFile(filename, 'r')
        (object_name, ext) = os.path.splitext(os.path.basename(filename))
        print(object_name, ext)
        filehandle = zip.read(object_name).splitlines()
    else:
        filehandle = open(filename, 'r')

    data = {'M':[],               ## molecule identification number
            'I':[],               ## isotope number
            'linecenter':[],      ## line center wavenumber (in cm^{-1})
            'S':[],               ## line strength, in cm^{-1} / (molecule m^{-2})
            'Acoeff':[],          ## Einstein A coefficient (in s^{-1})
            'gamma-air':[],       ## line HWHM for air-broadening
            'gamma-SD':[],        ## line HWHM for Speed Dependent Voigt Profile
            'gamma-self':[],      ## line HWHM for self-emission-broadening
            'Epp':[],             ## energy of lower transition level (in cm^{-1})
            'N':[],               ## temperature-dependent exponent for "gamma-air"
            'delta':[],           ## air-pressure shift, in cm^{-1} / atm
            'Vp':[],              ## upper-state "global" quanta index
            'Vpp':[],             ## lower-state "global" quanta index
            'Qp':[],              ## upper-state "local" quanta index
            'Qpp':[],             ## lower-state "local" quanta index
            'Ierr':[],            ## uncertainty indices
            'Iref':[],            ## reference indices
            'flag':[],            ## flag
            'gp':[],              ## statistical weight of the upper state
            'gpp':[],             ## statistical weight of the lower state
            'anuVc':[],           ## Velocity-changing frequency in cm-1 (Input). 
            'eta':[],             ## Correlation parameter, No unit 
            'Shift0':[],          ## Speed-averaged line-shift in cm-1 (Input).
            'Shift2':[],          ## Speed dependence of the line-shift in cm-1 (Input)     
            'gamD':[]}            ## Doppler HWHM in cm-1 (Input)

    print('Reading "' + filename + '" ...')

    for line in filehandle:
        if (len(line) < 160):
            raise ImportError('The imported file ("' + filename + '") does not appear to be a HITRAN2012-format data file.')

        data['M'].append(np.uint(line[0:2]))
        data['I'].append(np.uint(line[2]))
        data['linecenter'].append(float64(line[3:15]))
        data['S'].append(float64(line[15:25]))
        data['Acoeff'].append(float64(line[25:35]))
        data['gamma-air'].append(float64(line[35:40]))
        data['gamma-SD'].append(float64(line[35:40])*0.07)
        data['gamma-self'].append(float64(line[40:45]))
        data['Epp'].append(float64(line[45:55]))
        data['N'].append(float64(line[55:59]))
        data['delta'].append(float64(line[59:67]))
        data['Vp'].append(line[67:82])
        data['Vpp'].append(line[82:97])
        data['Qp'].append(line[97:112])
        data['Qpp'].append(line[112:127])
        data['Ierr'].append(line[127:133])
        data['Iref'].append(line[133:145])
        data['flag'].append(line[145])
        data['gp'].append(line[146:153])
        data['gpp'].append(line[153:160])
        # Hartmann profile parameters
        data['anuVc'].append(0*float64(line[25:35]))
        data['eta'].append(0*float64(line[25:35]))
        data['Shift0'].append(0*float64(line[25:35]))
        data['Shift2'].append(0*float64(line[25:35]))
        data['gamD'].append(0.009+0*float64(line[25:35]))

    if filename.endswith('.zip'):
        zip.close()
    else:
        filehandle.close()

    for key in data:
        data[key] = array(data[key])

    return(data)

