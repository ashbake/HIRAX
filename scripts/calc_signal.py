##############################################################
# Take exoplanet atmospheric spectrum and stellar info and 
# output HIRAX observation
###############################################################


import numpy as np
import matplotlib.pylab as plt
from astropy import units as u

import os,sys
import argparse
import matplotlib
import multiprocessing
from scipy import signal
from astropy.io import fits
from astropy.table import Table
from scipy.interpolate import interp1d
from scipy.integrate import trapz
#from joblib import Parallel, delayed

plt.ion()

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 18}

matplotlib.rc('font', **font)

os.system('mkdir output') # make folder to save all things made here

data_dir = '../data/'
sys.path.append('../utils/')
from functions import *

from objects import load_object
from load_inputs import load_inputs, reload_stellar

from functions import *

def define_lsf(v,res):
	# define gaussian in pixel elements to convolve resolved spectrum with to get rightish resolution
	dlam  = np.median(v)/res
	sigma = dlam/np.mean(np.diff(v)) # desired lambda spacing over current lambda spacing resolved to give sigma in array elements
	x = np.arange(sigma*10)
	gaussian = (1./sigma/np.sqrt(2*np.pi)) * np.exp(-0.5*( (x - 0.5*len(x))/sigma)**2 )

	return gaussian

def setup_band(x, x0=0, sig=0.3, eta=1):
	"""
	give step function

	inputs:
	------
	x0
	sig
	eta
	"""
	y = np.zeros_like(x)

	ifill = np.where((x > x0-sig/2) & (x < x0 + sig/2))[0]
	y[ifill] = eta

	return y

def resample(x,y_in,y_out,sig=0.3, dx=0, eta=1,mode='slow'):
	"""
	resample using convolution

	sig, dx in nanometers
	eta 0-1 for efficiency
	
	modes: slow, fast
	slow more accurate, fast uses fft

	slow method uses trapz so slightly more accurate, i think?
	"""
	if mode=='fast':
		dlam= np.median(np.diff(x)) # nm per pixel
		nsamp = int(sig / dlam)     # width of tophat
		temp_band   = eta * np.ones(nsamp)

		int_spec_in_oversample         = dlam * signal.fftconvolve(y_in,temp_band,mode='same') # dlam integrating factor
		int_spec_out_oversample        = dlam * signal.fftconvolve(y_out,temp_band,mode='same') # dlam integrating factor
		int_lam = x[int(nsamp/2 + dx/dlam):][::nsamp] # shift over by dx/dlam (npoints) before taking every nsamp point

		int_spec_in , int_spec_out = int_spec_in_oversample[int(nsamp/2 + dx/dlam):][::nsamp], int_spec_out_oversample[int(nsamp/2 + dx/dlam):][::nsamp]

	elif mode=='slow':
		i=0
		int_lam, int_spec_in, int_spec_out = [], [], []
		# step through and integrate each segment
		while i*sig/2 + dx< np.max(so.exo.v)-sig/2 - np.min(so.exo.v): # check
			xcent = np.min(x) + dx + i*sig/2
			temp_band   = setup_band(x, x0=xcent, sig=sig, eta=eta) # eta throughput of whole system
			int_spec_in.append(integrate(x,temp_band * y_in))
			int_spec_out.append(integrate(x,temp_band * y_out))
			int_lam.append(xcent)
			i += 1

	return int_lam, int_spec_in, int_spec_out

"""
calc flux using phoenix spectrum...reload spectrum for each type before running
then choose magnitude
""" 
def calc_flux(so,sig, eta, vel_p=0, amass1=1.0, amass2=1.0, res=4000, grism=False,sample_mode='slow'):
	"""

	"""
	# exoplanet transmission is 1-(rp/rs)^2
	transmission  = 1 - so.exo.s**2/(so.const.rsun * so.stel.rad)**2

	# shift star to user-defined velocity
	x_temp        = so.exo.v * (1 +(1.0 * so.stel.vel/300000.0))
	stel          = np.interp(so.exo.v,x_temp,so.stel.s,left=0,right=0)

	x_temp        = so.exo.v * (1 +(1.0 * (vel_p + so.stel.vel)/300000.0))
	transmission  = np.interp(so.exo.v,x_temp,transmission,left=0,right=0)

	# compute source
	source     = stel * transmission
	source_out = stel

	# multiply by telluric
	at_scope     = source     * so.tel.h2o**amass1 * so.tel.o2**amass1 * so.tel.rayleigh**amass1
	#at_scope     = source     * so.tel.h2o * so.tel.o2 * so.tel.rayleigh
	at_scope_out = source_out * so.tel.h2o**amass2 * so.tel.o2**amass2 * so.tel.rayleigh**amass2

	# Instrument
	at_ccd       = at_scope     * so.inst.exp_time   * so.inst.tel_area
	at_ccd_out   = at_scope_out * so.inst.exp_time   * so.inst.tel_area

	if grism == True:
		# convolve w/ PSF first - can make PFS variable
		lsf    = define_lsf(so.exo.v,res=res)
		at_ccd      = np.convolve(at_ccd,lsf,mode='same')
		at_ccd_out  = np.convolve(at_ccd_out,lsf,mode='same')	

	# resample data
	int_lam, int_spec_in, int_spec_out = resample(so.exo.v,at_ccd,at_ccd_out,sig=sig,dx=1,eta=1,mode=sample_mode)

	# compute errors
	int_lam, int_spec_in, int_spec_out = np.array(int_lam), np.array(int_spec_in), np.array(int_spec_out)
	err_in = np.sqrt(int_spec_in)
	err_out = np.sqrt(int_spec_out)

	in_over_out     = int_spec_in/int_spec_out
	in_over_out_err = in_over_out * np.sqrt(1/int_spec_in + 1/int_spec_out)

	# put things into so.out. - make this a thing instead of afterthought*
	so.out.s_in  = at_ccd
	so.out.s_out = at_ccd_out
	so.out.s_in_obvs = int_spec_in
	so.out.s_in_obvs = int_spec_out	

	return int_lam, in_over_out, in_over_out_err


if __name__ == '__main__':
	config_name = 'calc_signal.cfg'

	so = load_object(config_name) # load storage object and load config file
	lin = load_inputs()
	so = lin.load_all(so) # must do this if want different star type

	# calc flux for range of variable changes

	lam, frat, frat_err = calc_flux(so,sig, eta, vel_p=0, amass1=1.0, amass2=1.0, res=4000, grism=False,sample_mode='slow'):

#plot
plt.figure(-10)
plt.errorbar(int_lam,in_over_out,in_over_out_err)

plt.figure(-9)
plt.errorbar(int_lam, int_spec_in,err_in)
plt.errorbar(int_lam, int_spec_out,err_out)


def plot_signal(int_lam, in_over_out, in_over_out_err):
	"""
	plot f_in over f_out
	"""
	fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True,figsize=(10,6))

	ax1.plot(so.exo.v,source/source_out)
	ax2 = ax.twinx()     # use this for stellar contamination later


	# plot
	if plot_on:
		fig, (ax, ax3) = plt.subplots(2, 1, sharex=True,figsize=(10,6))

		ax.plot(so.exo.v,source/source_out)

		ax2 = ax.twinx() # use this for stellar contamination later
		ax2.set_zorder(ax.get_zorder()+1) # put ax in front of ax2 
		ax2.fill_between(so.exo.v,y1=onband,y2=0*so.exo.v,facecolor='m',alpha=0.5)
		ax2.fill_between(so.exo.v,y1=offband,y2=0*so.exo.v,facecolor='g',alpha=0.5)
		
		ax.set_title('Star: %s    Vel: %s'%(so.stel.type,so.stel.vel))
	
		ax2.set_ylim(0,0.9)
		ax.set_ylabel('($R_{p}/R_*)^2$')
		ax2.set_ylabel('Etalon Transmission')
	
		# Add stellar spectrum
		ax3.plot(so.exo.v,source_out,'k')
		ax3.set_xlabel('Wavelength (nm)')
		ax3.set_ylabel('Stellar Flux Density \n (phot/s/m$^2$/nm)')

		ax4 = ax3.twinx() # use this for stellar contamination later
		ax4.set_zorder(ax.get_zorder()+1) # put ax in front of ax2 
		ax4.set_ylabel('Sky Transmission')
	
		ax4.plot(so.exo.v,so.tel.o2)
		ax4.plot(so.exo.v,so.tel.rayleigh)

		# Plot OH lines
		i_oh = np.where((so.tel.x_oh > 750.0) & (so.tel.x_oh < 800.0))[0]
		ax3.scatter(so.tel.x_oh[i_oh],so.tel.y_oh[i_oh]*5000,marker='s',s=5,c='m',label='OH Emission')

		ax.set_xlim(754,800)		
		
		fig.subplots_adjust(bottom=0.1,left=0.22,hspace=0,right=0.88,top=0.9)
		plt.savefig('./plots/filter_setup_%s.png'%setup)

	return signal_ppm, noise_ppm, onsignal, offsignal, onsignal_out, offsignal_out




