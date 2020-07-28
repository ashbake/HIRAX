##############################################################
# Load variables into objects object
###############################################################

import numpy as np
import matplotlib.pylab as plt
from scipy.interpolate import interp1d
from scipy.integrate import trapz
from astropy.io import fits
from scipy import interpolate
import glob,sys,os

from functions import *

__all__ = ['load_inputs','load_phoenix','reload_stellar','get_line_info']

def load_phoenix(stelname,wav_start=750,wav_end=780):
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
	isub = np.where( (lam > wav_start*10.0) & (lam < wav_end*10.0))[0]

	# Convert 
	return lam[isub],spec[isub]


def load_phoenix_txt(filename,skipline=1,wav_column=0,wav_start=0.75,wav_end=0.78):
	"""
	load a high resolution file by not reading all of it
	
	no. of columns can be different
	
	wav_column assumed wavelength is located in zeroeth column
	
	also assuming wavelength is in units of wav_start/end and increasing
	"""
	f = open(filename,'r')
	lines = f.readlines()
	
	sublines = []
	nrows = len(lines[skipline+1].strip().split())
	
	for i,line in enumerate(lines[skipline:]):
		if float(line.strip().split()[wav_column]) > wav_start:
			if float(line.strip().split()[wav_column]) > wav_end:
				break
			sublines.append(line)
	
	loaded_file = np.zeros((len(sublines),nrows))
	for i in range(len(sublines)):
		saveme         = np.array(sublines[i].strip().split())
		loaded_file[i] = saveme.astype(float)
	
	return loaded_file



class load_inputs():
	""" 
	Load variables into storage object
	
	Inputs: so (storage object with user defined things loaded)
	Outputs: so (storage object with data and other stuff loaded)
	
	Edits
	-----
	Ashley - initial implementation Oct 26, 2018
	"""
	def __init__(self):
		pass
		
	def exoplanet(self,so):
		if so.run.mode == 'transmission':
			spec = np.loadtxt(so.exo.folder + so.exo.name)
			x,y   =  spec[:,0], spec[:,1] # x: micron vacuum  y: unitless Rp2/Rs2
			so.exo.v = vac_to_stand(x[::-1]*10000)/10 # needs to be in angstrom for vac to air fxn
			so.exo.s = y[::-1]
		if so.run.mode == 'albedo':
			spec = np.loadtxt(so.exo.folder + so.exo.name)
			x,y   =  spec[:,0], spec[:,4] # x: micron vacuum  y: albedo spectrum
			v = vac_to_stand(x*10000)/10 # needs to be in angstrom for vac to air fxn
			#so.exo.s = y
			# Resample bc the sampling sucks
			so.exo.v = np.arange(v[3000],v[3800],0.001)
			so.exo.s = np.interp(so.exo.v,v,y)
		if so.run.mode=='albedo_sarah':
			spec  = np.loadtxt(so.exo.folder + so.exo.name)
			x,y   =  10* 1e7/spec[:,0], spec[:,2]*0.07 # x: micron vacuum  y: albedo spectrum
			v = vac_to_stand(x)/10 # needs to be in angstrom for vac to air fxn
			#so.exo.v = np.arange(v[85000],v[77000],0.001) # 0.01 resolution
			#so.exo.v = np.arange(v[-1110],v[1111],0.001)
			so.exo.v  = v[np.where( (v < 775.0) & (v > 750.0) )[0]][::3]
			so.exo.s  = np.interp(so.exo.v,v[::-1],y[::-1])
		
		return so



	def instrument(self,so):
		"""
		Load telescope option
		"""
		if so.inst.telescope in so.const.telescopes:
			itelescope = np.where(str(so.inst.telescope) == np.array(so.const.telescopes))[0][0]
			so.inst.tel_area = so.const.tel_areas[itelescope]
			so.inst.focal_ratio = so.const.focal_ratios[itelescope]
		else:
			print 'ERROR: Did not pick a correct telescope name. Options: GMT, TMT, ELT'
			print 'loading default GMT'
			so.inst.telescope = so.const.telescopes[0]
			so.inst.tel_area = so.const.tel_areas[0]
			so.inst.focal_ratio = so.const.focal_ratios[0]
			
		return so
	
	def telluric(self,so):
		# OH 
		f = np.loadtxt(so.tel.folder + so.tel.name_oh) # vacuum wl angstroms
		so.tel.x_oh = vac_to_stand(f[:,0])/10
		so.tel.y_oh = f[:,1]

		# Load hitran
		so.tel.hitran = read_hitran2012_parfile(so.tel.name_hitran)
		istrong       = np.where(so.tel.hitran['S'] > 0.25e-24)[0]
		so.tel.hitran_cents  = 0.1*vac_to_stand(1.e8/so.tel.hitran['linecenter'][istrong][::-1]) # increased order array of strongest lines
	
		# TAPAS oxygen
		tapload        = fits.open(so.tel.folder + so.tel.name_o2)
		wavo2          = 1e8/tapload[1].data['wavenumber'][4800000:6000000]  # convert to standard*****                                                     
		o2             = tapload[1].data['transmittance'][4800000:6000000]
		so.tel.airmass = float(tapload[1].header['AIRMASS'])
		tapload.close()
		wavo2          = vac_to_stand(wavo2)/10.0
		o2[np.where(o2 < 0)[0]] = 1e-8
	
		o2_interp = interp1d(wavo2,o2,bounds_error=False,fill_value=1)
		so.tel.v   = so.exo.v
		so.tel.o2  = o2_interp(so.exo.v)

		# TAPAS water
		tapload         = fits.open(so.tel.folder + so.tel.name_h2o)
		wavh2o          = 1e8/tapload[1].data['wavenumber'][4800000:6000000]  # convert to standard*****                                                     
		h2o             = tapload[1].data['transmittance'][4800000:6000000]
		so.tel.pwv      = float(tapload[1].header['H2OCV'])
		if so.tel.airmass != float(tapload[1].header['AIRMASS']):
			print ' WARNING..assuming airmass 1 for h2o but it is a lie '
		tapload.close()
		wavh2o                    = vac_to_stand(wavh2o)/10.0
		h2o[np.where(h2o < 0)[0]] = 1e-8
	
		h2o_interp = interp1d(wavh2o,h2o,bounds_error=False,fill_value=1)
		so.tel.h2o  = h2o_interp(so.exo.v)
	
		# Tapas airmass at zenith
		rayleigh     = np.loadtxt(so.tel.folder + so.tel.name_rayleigh)	
		ray_interp   = interp1d(rayleigh[:,0],rayleigh[:,1],bounds_error=False,fill_value=1)
		so.tel.rayleigh = ray_interp(so.exo.v)

		return so

	def filter(self,so,nbands=0):
		###########
		# DUALON BAND PASS
		if nbands > 0:
			so.filt.nbands = nbands

		filt = np.loadtxt(so.filt.folder + so.filt.dualon_raw_name)
		amp_current = np.max(filt[:,1])
		filt[:,1]*= float(so.filt.amp)/amp_current
		so.filt.dualon_raw = filt

		cents = so.tel.hitran_cents
		# Red end dualon
		so.filt.band_cents_red = cents[np.where(cents > 762.0)[0]][0:so.filt.nbands]
		# Blue end dualon
		so.filt.band_cents_blue = cents[np.where(cents < 762.0)[0]][0:so.filt.nbands]

		# create array of dualons..might have to upgrade this to interpolate e'rthing
		filt_x = np.arange(750,770,np.diff(filt[:,0])[0]) # take advantage of uniform sampling of filt profile
		filt_y = np.zeros(len(filt_x))
		for cent in so.filt.band_cents_red:
			icent = np.where(np.abs(filt_x - cent) == np.min(np.abs(filt_x - cent)))[0][0]
			filt_y[icent-len(filt[:,0])/2:icent+len(filt[:,0])/2+1] = filt[:,1]

		# interp
		tck_filt           = interpolate.splrep(filt_x,filt_y, k=2, s=0)
		so.filt.s_dual_red = interpolate.splev(so.exo.v,tck_filt,der=0)
		so.filt.v_dual     = so.exo.v
		
		filt_x = np.arange(750,770,np.diff(filt[:,0])[0]) # take advantage of uniform sampling of filt profile
		filt_y = np.zeros(len(filt_x))
		# Blue end (start at stronger end)
		for cent in so.filt.band_cents_blue[::-1]:
			icent = np.where(np.abs(filt_x - cent) == np.min(np.abs(filt_x - cent)))[0][0]
			filt_y[icent-len(filt[:,0])/2:icent+len(filt[:,0])/2+1] = filt[:,1]
		# interp
		tck_filt       = interpolate.splrep(filt_x,filt_y, k=2, s=0)
		so.filt.s_dual_blue = interpolate.splev(so.exo.v,tck_filt,der=0)
		so.filt.v_dual = so.exo.v
	
		############
		# Create Oxyometer bandpasses as just tophat
		f        = np.loadtxt(so.filt.folder + so.filt.name_alluxa)
		on_lam   = 759.6 # nm
		off_lam  = 758.0 # nm
	
		on_f  = interpolate.interp1d(f[:,0]+(on_lam-607.4),f[:,1])   # x in nanometers
		off_f = interpolate.interp1d(f[:,0]+(off_lam-607.4),f[:,1])  # x in nanometers 
		so.filt.s_alluxa_on  = on_f(so.exo.v)/100.0
		so.filt.s_alluxa_off = off_f(so.exo.v)/100.0
	
		# put tophat at wider bandpass
		on_wide = np.zeros(len(so.exo.v))
		off_wide= np.zeros(len(so.exo.v))
		on_wide[np.where((so.exo.v > 759.3) & (so.exo.v < 761.8))]  = 1.0
		off_wide[np.where((so.exo.v > 756.7) & (so.exo.v < 759.2))] = 1.0
	
		so.filt.s_alluxa_wide_on  = on_wide
		so.filt.s_alluxa_wide_off = off_wide
		
		# I band filter	
		so.filt.I_raw = np.loadtxt(so.filt.folder + so.filt.name_I)
		
		return so
				
	
	def stellar(self,so):
		so.stel.I_mag     = 5.0 * np.log10(so.stel.dist/10.0) + so.const.abs_mag[np.where(so.const.types == so.stel.type)[0]]
		so.stel.rad       = so.const.radii[np.where(so.const.types == so.stel.type)[0]]

		so.stel.name = glob.glob(so.stel.folder + so.stel.type + '*.fits')[0]
		#Iband_sub    = load_phoenix_old(so.stel.name,wav_start=0.69,wav_end=0.931)
		Iband_sub    = load_phoenix(so.stel.name,wav_start=690.0,wav_end=931.0)

		## Stellar to photon units
		I_interp       = interpolate.interp1d(so.filt.I_raw[:,0],so.filt.I_raw[:,1], bounds_error=False,fill_value=0)
		I_stel         = Iband_sub[1] * I_interp(Iband_sub[0])
		# Shift I_stel to air wavelength to be super consistent
		tck_Istel    = interpolate.splrep(vac_to_stand(Iband_sub[0]),I_stel, k=2, s=0)
		I_stel_air   = interpolate.splev(Iband_sub[0],tck_Istel,der=0,ext=0)
	
		# what's the integrate i flux supposed to be in photons?
		xstel              = Iband_sub[0]  # angstroms
		so.filt.dl_l_I     = np.mean(integrate(xstel,I_interp(Iband_sub[0]))/xstel)
		nphot_expected_0   = so.filt.dl_l_I * so.filt.zp_I*(1e-23 * 1.5*10**26) # last factor to convert to phot/s/cm2
		nphot_phoenix      = integrate(xstel,I_stel_air)
		so.stel.factor_0   = nphot_expected_0/nphot_phoenix 
	
		# Save stellar spectrum info
		oxy_sub     = load_phoenix(so.stel.name,wav_start=750,wav_end=780) 
		tck_stel    = interpolate.splrep(vac_to_stand(oxy_sub[0])/10.,oxy_sub[1], k=2, s=0)
		so.stel.s   = so.stel.factor_0 * 10.0 * 100**2 * interpolate.splev(so.exo.v,tck_stel,der=0,ext=1)
		so.stel.v   = so.exo.v
		so.stel.units = 'photons/s/m2/nm'

		so.stel.notes = 'spent a lot of time making sure this is correct - expect 670000photons/s/cm2 through I band for 0 magnitude star so can check with this'\
					+ ' and at the website https://www.astro.umd.edu/~ssm/ASTR620/mags.html'\
					+ '. Multiply so.stel.s by 10**(-0.4*so.stel.Imag) to get diff magnitude'
		# stellar spec is in photons/s/m2/nm(100^2 is to convert to m2 and 10 is for nm from angstrom)
	
		#############
		#### Stellar Contamination
		#############
		if so.stel.type != 'G2':
			spotslist  = glob.glob('%s%s*rspot7*' %(so.stel.folder_contamination ,so.stel.type))
			spotslist2 = glob.glob('%s%s*rspot2*' %(so.stel.folder_contamination ,so.stel.type))
			dat   = load_phoenix_txt(spotslist[0],wav_start=0.75,wav_end=0.78) 
			dat2  = load_phoenix_txt(spotslist2[0],wav_start=0.75,wav_end=0.78) 
		
			interp_dat  = interp1d(dat[:,0],dat[:,1],bounds_error=False,fill_value=1)
			interp_dat2 = interp1d(dat2[:,0],dat2[:,1],bounds_error=False,fill_value=1)

			so.stel.contamination  = interp_dat(so.exo.v/1000)
			so.stel.contamination2 = interp_dat2(so.exo.v/1000)

		return so
	
	def load_all(self,so):
		so = self.exoplanet(so)
		so = self.instrument(so)
		so = self.telluric(so)
		so = self.filter(so)
		so = self.stellar(so)
		
		return so
	
	def reload_stellar(self,so,new_type,new_dist):
		i_type         = np.where(so.const.types == new_type)[0]
		if so.stel.type == new_type:
			so.stel.dist   = new_dist
			so.stel.I_mag  = 5.0 * np.log10(so.stel.dist/10.0) + so.const.abs_mag[i_type]
			return so
		else:
			so.stel.type   = new_type
			so.stel.rad    = so.const.radii[i_type]
			so.stel.dist   = new_dist
			so.stel.I_mag  = 5.0 * np.log10(so.stel.dist/10.0) + so.const.abs_mag[i_type]
			so= self.stellar(so)
		
		return so


def calc_noise(self,so):
	"""
	determine total noise for each band

	must do after load bands
	"""
	C = 10e-9
	#c_leak = so.out.star * C *


		
def reload_stellar(so,new_type):
	"""
	Redo stellar loading stuff here since I think this is the only thing
	have to constantly reload
	can make load thing a class...
	"""
	#############
	#### Stellar # use I band for scaling bc have those magnitudes
	#############
	so.stel.type = new_type

	so.stel.I_mag     = 5.0 * np.log10(so.stel.dist/10.0) + so.const.abs_mag[np.where(so.const.types == so.stel.type)[0]]
	so.stel.rad       = so.const.radii[np.where(so.const.types == so.stel.type)[0]]

	so.stel.name = glob.glob(so.stel.folder + so.stel.type + '*.fits')[0]
	#Iband_sub    = load_phoenix_old(so.stel.name,wav_start=0.69,wav_end=0.931)
	Iband_sub    = load_phoenix(so.stel.name,wav_start=690.0,wav_end=931.0)

	## Stellar to photon units
	I_interp       = interpolate.interp1d(so.filt.I_raw[:,0],so.filt.I_raw[:,1], bounds_error=False,fill_value=0)
	I_stel         = Iband_sub[1] * I_interp(Iband_sub[0])
	# Shift I_stel to air wavelength to be super consistent
	tck_Istel    = interpolate.splrep(vac_to_stand(Iband_sub[0]),I_stel, k=2, s=0)
	I_stel_air   = interpolate.splev(Iband_sub[0],tck_Istel,der=0,ext=0)
	
	# what's the integrate i flux supposed to be in photons?
	xstel              = Iband_sub[0]  # angstroms
	so.filt.dl_l_I     = np.mean(integrate(xstel,I_interp(Iband_sub[0]))/xstel)
	nphot_expected_0   = so.filt.dl_l_I * so.filt.zp_I*(1e-23 * 1.5*10**26) # last factor to convert to phot/s/cm2
	nphot_phoenix      = integrate(xstel,I_stel_air)
	so.stel.factor_0   = nphot_expected_0/nphot_phoenix 
	
	# Save stellar spectrum info
	oxy_sub     = load_phoenix(so.stel.name,wav_start=750,wav_end=780) 
	tck_stel    = interpolate.splrep(vac_to_stand(oxy_sub[0])/10.,oxy_sub[1], k=2, s=0)
	so.stel.s   = so.stel.factor_0 * 10.0 * 100**2 * interpolate.splev(so.exo.v,tck_stel,der=0,ext=0)
	so.stel.v   = so.exo.v
	so.stel.units = 'photons/s/m2/nm'
	so.stel.notes = 'spent a lot of time making sure this is correct - expect 670000photons/s/cm2 through I band for 0 magnitude star so can check with this'\
					+ ' and at the website https://www.astro.umd.edu/~ssm/ASTR620/mags.html'\
					+ '. Multiply so.stel.s by 10**(-0.4*so.stel.Imag) to get diff magnitude'
	# stellar spec is in photons/s/m2/nm(100^2 is to convert to m2 and 10 is for nm from angstrom)

	#############
	#### Stellar Contamination
	#############
	if so.stel.type != 'G2':
		spotslist  = glob.glob('%s%s*rspot7*' %(so.stel.folder_contamination ,so.stel.type))
		spotslist2 = glob.glob('%s%s*rspot2*' %(so.stel.folder_contamination ,so.stel.type))
		dat   = load_phoenix_txt(spotslist[0],wav_start=0.75,wav_end=0.78) 
		dat2  = load_phoenix_txt(spotslist2[0],wav_start=0.75,wav_end=0.78) 
		
		interp_dat  = interp1d(dat[:,0],dat[:,1],bounds_error=False,fill_value=1)
		interp_dat2 = interp1d(dat2[:,0],dat2[:,1],bounds_error=False,fill_value=1)

		so.stel.contamination  = interp_dat(so.exo.v/1000)
		so.stel.contamination2 = interp_dat2(so.exo.v/1000)

	return so
	
def get_line_info(so):
	q_low = []
	q_hi=[]
	branch = []
	for i in range(len(so.tel.hitran['S'])):
		temp = so.tel.hitran['Qpp'][i]
		q_low.append(int(temp[3:5]))
		q_hi.append(int(temp[7:9]))
		branch.append(temp[1:2]+temp[5:6])
	q_low = np.array(q_low)
	q_hi  = np.array(q_hi)
	branch = np.array(branch)
	
	
