##############################################################
# Take exoplanet atmospheric spectrum and stellar info and 
# output HIRAX observation
###############################################################


import numpy as np
import matplotlib.pylab as plt
from astropy import units as u

import argparse
import matplotlib
import multiprocessing
from astropy.io import fits
from astropy.table import Table
from scipy.interpolate import interp1d
from scipy.integrate import trapz
from joblib import Parallel, delayed

plt.ion()

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 18}

matplotlib.rc('font', **font)

os.system('mkdir output') # make folder to save all things made here

data_dir = '../data/'

#if len(sys.argv) > 1:
#	so.run.ispec = int(sys.argv[1]) - 1  # subtract 1 bc 0 isnt valid job id

######################### LOAD THINGS #####################
# Input: - percent planet signal
#		 - Doppler motion of planet
#		 - Barycentric velocity
#		 - width of planetary signal (?)
#        - magnitude of star
#        - HIRAX band setup (options 1- ?)

# Output:
#        - simulated HIRAX bands
#	 	 - extracted signal by some fit measure vs planet signal colored by stellar magnitude
#        - sampling of transit

# Step 2:
#        - take best result from each HIRAX setup option and show the following
#		 - telluric variability in bands - how well do need to know PWV/O2? - will have O2 band at 1.27micron , use my data to compare A, B bands over time
#        - CLV and RM effect in band - how can this be discerned at low res?
#        - comments on sampling for stellar activity/spot extraction/CLV light curve modeling
#        - comments on parvi info - on O2, H2O, RM effect, etc
"""
Use batman to simulate light curves - change planet radius and limb darkening for the different wavelengths
How does limb darkening depend on wavelength?
https://iopscience.iop.org/article/10.3847/1538-4357/aa7edf/pdf
"""
######################### Calc Signal Fxn #################

def run():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--plottype",
        type=str,
        default="pdf",
        help="Plot file suffix and file type.  E.g., 'pdf', 'png'",
    )
    parser.add_argument(
        "--plot",
        default=True,
        action="store_true",
        help="Make plot",
    )

    parser.add_argument(
        "--noplot", dest="plot", action="store_false", help="Skip making the plots."
    )

    parser.add_argument(
        "--overwrite",
        type=bool,
        default=False,
        help="Overwrite files",
    )
    parser.add_argument(
        "--datapath",
        type=str,
        default='../data/',
        help="Data path,",
    )

    args = parser.parse_args()

    if args.makefilenew:
        get_mastar_spec()  # downloading mastar files takes a few hours - there are thousands. if already downloaded then nothing happens
        make_data_files()  # this takes a few minutes

    if args.plot:
        col = load_colors(datapath=args.datapath)
        plot_error(col,suffix=args.suffix)


def calc_SNR(mag, diam=5):
	"""
	calc SNR of spectrum

	inputs:
	-------
	dist:    stellar distance
	type:    stellar type to pick phoenix model
	exptime: exposure time in seconds
	diam:    telescope diameter (defaul 5m for Hale)

	outputs:
	--------
	Return stellar spectrum in photons/lambda
	"""
	pass

def band_flux(mag, band='Vmag'):
	"""
	load rough flux counts for band interested in
	Inputs:
	------
	exp_time (sec)
	diam (m)
	band must be one of following (str): Vmag, Imag, Rmag, Jmag

	Notes:
	------
	for V band constants: http://astroweb.case.edu/ssm/ASTR620/mags.html
	1 Jy = 1.51e7 photons sec^-1 m^-2 (dlambda/lambda)^-1
	"""
	# store center wavelength (micron), delta_lambda over lambda , flux at m=0 at top of atm Jy
	mag_const = {}
	mag_const['Note'] = ['lambda_c', 'dl_l', 'flux_m0']
	mag_const['Vmag'] = [0.55, 0.16, 3640]
	mag_const['Rmag'] = [0.64, 0.23, 3080]
	mag_const['Imag'] = [0.79, 0.19, 2550]
	mag_const['Jmag'] = [1.26, 0.16, 1600]

	# define V band constants
	lambda_c, dl_l, flux_m0 = mag_const['band']

	flux_mag = 10**(-0.4*mag) * flux_m0

	return flux_mag * 1.51e7 * dl_l # phot/sec/m2

def load_phoenix(stelname,wav_start=750*u.nm,wav_end=780*u.nm):
	"""
	load fits file stelname with stellar spectrum from phoenix 
	http://phoenix.astro.physik.uni-goettingen.de/?page_id=15
	
	return subarray 
	
	wav_start, wav_end specified in nm
	
	convert s from egs/s/cm2/cm to phot/cm2/s/nm using
	https://hea-www.harvard.edu/~pgreen/figs/Conversions.pdf
	"""
	
	# conversion factor

	f = fits.open(stelname)
	spec = f[0].data / (1e8) # ergs/s/cm2/cm to ergs/s/cm2/Angstrom for conversion
	f.close()
	
	path = stelname.split('/')
	f = fits.open(path[0] + '/' + path[1] + '/' + path[2] +'/' + \
					 'WAVE_PHOENIX-ACES-AGSS-COND-2011.fits')
	lam = f[0].data # angstroms
	f.close()
	
	# Convert
	conversion_factor = 5.03*10**7 * lam
	spec *= conversion_factor # phot/cm2/s/angstrom
	
	# Take subarray requested
	isub = np.where( (lam*u.Angstrom  > wav_start) & (lam*u.Angstrom < wav_end))[0]

	# Convert 
	return lam[isub],spec[isub]

def get_stellar(x, type, mag, band='Vmag'):
	"""
	load stellar of spectral type normalized by band magnitude

	Inputs:
	------
	x: x array over which to sample spectrum
	type: stellar type e.g. 'G0'
	mag:  magntidue of star
	band: (str) either Vmag, Jmag, Imag, or Rmag

	Outputs:
	--------
	Stellar Spectrum photons/m2/s/nm
	"""
	band_waves = {}
	band_waves['Vmag'] = [400.0,600.0]  * u.nm
	band_waves['Rmag'] = [500.0, 700.0] * u.nm
	band_waves['Imag'] = [690.0, 931.0] * u.nm
	band_waves['Jmag'] = [1200.0,1800.0]* u.nm

	# load stellar model
	stelpath = glob.glob(data_dir + 'phoenix/' + '%s*' %type)
	lam,spec = load_phoenix(stelpath,wav_start=band_waves[band][0],wav_end=band_waves[band][0])


def calc_signal(rad,so,onband,offband,radlow=6371,specrad0=0.2):
	"""
	Integrate over transit spectrum for velocity array, vels,
	and stellar radius, rad. 
	
	Returns signal in ppm per vel
	
	specrad0=0.2 for when the transit spec was created assuming r_star =0.2r_sun
	units of transit spectrum should be (R_p/R_s)^2
	radlow is lower radius to take (like in refraction case)
	"""
	iflatten    = np.where(np.sqrt(so.exo.s*(specrad0*so.const.rsun)**2) < radlow)[0]
	ytemp       = 1.0*so.exo.s
	ytemp[iflatten] = radlow**2/(specrad0*so.const.rsun)**2

	onsignal_vel = integrate(so.exo.v,onband*(1-ytemp*(specrad0/rad)**2))
	offsignal_vel= integrate(so.exo.v,offband*(1-ytemp*(specrad0/rad)**2))

	rat_bands = integrate(so.exo.v,onband) / integrate(so.exo.v,offband)
	
	return (rat_bands-onsignal_vel/offsignal_vel)*1e6


def calc_flux(so,setup='narrow',radlow=6371,plot_on=False):
	"""
	calc flux using phoenix spectrum...reload spectrum for each type before running
	then choose magnitude
	""" 
	specrad0    = float(so.exo.specrad0)
	iflatten    = np.where(np.sqrt(so.exo.s*(specrad0*so.const.rsun)**2) < radlow)[0]
	ytemp       = 1.0 * so.exo.s * (specrad0/so.stel.rad)**2
	ytemp[iflatten] = radlow**2/(so.stel.rad*so.const.rsun)**2
	transmission    = 1-ytemp
	
	source     = so.stel.s * 10**(-0.4*so.stel.V_mag) * transmission
	source_out = so.stel.s * 10**(-0.4*so.stel.V_mag) 

	# Pick bandpass
	if setup=='narrow':
		onband   = so.filt.s_alluxa_on
		offband  = so.filt.s_alluxa_off
	elif setup=='wide':
		onband = so.filt.s_alluxa_wide_on
		offband  = so.filt.s_alluxa_wide_off
	

	# Shift source to user specified velocity (default is 0.0)
	if so.stel.vel != 0.0:
		x_temp = so.exo.v + so.exo.v*(1.0*so.stel.vel/300000.0)
		# shift stellar
		interpx_shift      = interp1d(x_temp,source,bounds_error=False,fill_value=0)
		interpx_shift_out  = interp1d(x_temp,source_out,bounds_error=False,fill_value=0)
		source           = interpx_shift(so.exo.v)
		source_out       = interpx_shift_out(so.exo.v)
		# Shift dualons
		interpx_onband   = interp1d(x_temp,onband,bounds_error=False,fill_value=0)
		interpx_offband  = interp1d(x_temp,offband,bounds_error=False,fill_value=0)
		onband  = interpx_onband(so.exo.v)
		offband = interpx_offband(so.exo.v)
		
	# multiply by telluric
	at_scope     = source     * so.inst.tel_area * so.tel.o2 * so.tel.rayleigh 
	at_scope_out = source_out * so.inst.tel_area * so.tel.o2 * so.tel.rayleigh

	# Instrument
	at_ccd       = at_scope     * so.inst.exp_time * so.inst.qe * so.inst.tel_reflectivity
	at_ccd_out   = at_scope_out * so.inst.exp_time * so.inst.qe * so.inst.tel_reflectivity
	
	# Integrate
	onsignal   = so.inst.mirror_reflectivity**4 * integrate(so.exo.v,onband*at_ccd)
	offsignal  = so.inst.mirror_reflectivity* integrate(so.exo.v,offband*at_ccd)
	onsignal_out   = so.inst.mirror_reflectivity**4 * integrate(so.exo.v,onband*at_ccd_out)
	offsignal_out  = so.inst.mirror_reflectivity*integrate(so.exo.v,offband*at_ccd_out)

	signal_ppm = (onsignal_out/offsignal_out - onsignal/offsignal)*1e6

	# Photon Noise
	frat_out = onsignal_out/offsignal_out
	frat_in  = onsignal/offsignal
	
	noise_out = frat_out * np.sqrt(1/onsignal_out + 1/offsignal_out)
	noise_in  = frat_in  * np.sqrt(1/onsignal     + 1/offsignal)
	
	noise_ppm = 1e6 * np.sqrt(noise_out**2 + noise_in**2)

	# plot
	if plot_on:
		fig, (ax, ax3) = plt.subplots(2, 1, sharex=True,figsize=(10,6))

		ax.plot(so.exo.v,source/source_out)

		ax2 = ax.twinx() # use this for stellar contamination later
		ax2.set_zorder(ax.get_zorder()+1) # put ax in front of ax2 
		ax2.fill_between(so.exo.v,y1=onband,y2=0*so.exo.v,facecolor='m',alpha=0.5)
		ax2.fill_between(so.exo.v,y1=offband,y2=0*so.exo.v,facecolor='g',alpha=0.5)
		
		ax.set_title('Star: %s  Type: %s   Vel: %s'%(so.stel.type,so.stel.star,so.stel.vel))
	
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


	
#####################################
# Calculate Flux - can shift exoplanet + stellar signal multiplied 
# because will need to include telluric 
#####################################


#####################################
# Calculate Flux - use vel=25 km/s 
#####################################
d_bands = np.arange(100)/10.0+1.0
s0,n0,s1,n1 = np.zeros((2,len(d_bands))), np.zeros((2,len(d_bands))), np.zeros((2,len(d_bands))), np.zeros((2,len(d_bands)))
# HD189
so.stel.star = so.const.names[0]
so = lin.stellar(so)
for i,d_band in enumerate(d_bands):
	so.filt.d_band = d_band
	so = lin.filter(so)
	s0[0,i], n0[0,i] , onsignal, offsignal, onsignal_out, offsignal_out \
				= calc_flux(so,setup='wide',radlow=0,plot_on=False)
	s0[1,i], n0[1,i] , onsignal, offsignal, onsignal_out, offsignal_out \
				= calc_flux(so,setup='narrow',radlow=0,plot_on=False)

#compute for when a band is in between 
so.filt.d_band = -1.58 #nm
so = lin.filter(so)
s189, n189, onsignal, offsignal, onsignal_out, offsignal_out \
				= calc_flux(so,setup='narrow',radlow=0,plot_on=False)

# HD209
so.stel.star = so.const.names[1]
so = lin.stellar(so)
for i,d_band in enumerate(d_bands):
	so.filt.d_band = d_band
	so = lin.filter(so)
	s1[0,i], n1[0,i] , onsignal, offsignal, onsignal_out, offsignal_out \
				= calc_flux(so,setup='wide',radlow=0,plot_on=False)
	s1[1,i], n1[1,i] , onsignal, offsignal, onsignal_out, offsignal_out \
				= calc_flux(so,setup='narrow',radlow=0,plot_on=False)

#compute for when a band is in between 
so.filt.d_band = -1.58 #nm
so = lin.filter(so)
s209, n209, onsignal, offsignal, onsignal_out, offsignal_out \
				= calc_flux(so,setup='narrow',radlow=0,plot_on=True)

# Plot
plt.figure()
plt.plot(d_bands,2*(3.0/(s0[0]/n0[0]))**2,c=so.const.colors[0], \
				ls=so.const.symbols[0],label='Wide')
plt.plot(d_bands,2*(3.0/(s0[1]/n0[1]))**2,c=so.const.colors[0], \
				ls=so.const.symbols[2],label='Narrow')

plt.plot(d_bands,2*(3.0/(s1[0]/n1[0]))**2,c=so.const.colors[3], \
				ls=so.const.symbols[0])
plt.plot(d_bands,2*(3.0/(s1[1]/n1[1]))**2,c=so.const.colors[3], \
				ls=so.const.symbols[2])


plt.axhline(2*(3.0/(s189/n189))**2,c=so.const.colors[0])
plt.axhline(2*(3.0/(s209/n209))**2,c=so.const.colors[3])

#plt.xlim(4,7.5)
#plt.ylim(200,4000)
plt.xlabel('Band Spacing (nm)')
plt.ylabel('N$_{exp}$')
plt.legend(loc='best')
plt.subplots_adjust(bottom=0.17,left=0.19)

plt.savefig('./plots/best_filter.png')

exposures = np.arange(100)



def plot_potassium_setup():
	plt.style.use('dark_background')
	fig, ax = plt.subplots(1, 1, sharex=True,figsize=(10,6))

	ax.plot(so.exo.v,so.exo.s)

	#ax2 = ax.twinx() # use this for stellar contamination later
	#ax2.set_zorder(ax.get_zorder()+1) # put ax in front of ax2 
	#ax2.fill_between(so.exo.v,y1=so.filt.s_alluxa_on,y2=0*so.exo.v,facecolor='m',alpha=0.5)
	#ax2.fill_between(so.exo.v,y1=so.filt.s_alluxa_off,y2=0*so.exo.v,facecolor='g',alpha=0.5)

	fig.subplots_adjust(bottom=0.2,left=0.25,hspace=0,right=0.88,top=0.9)
	ax.set_xlabel('Wavelength (nm)')
	ax.set_ylabel(r'$\frac{R_p^2}{R_s^2}$',rotation=0,labelpad=20,fontsize=30)



