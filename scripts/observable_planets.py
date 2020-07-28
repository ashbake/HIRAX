##############################################################
# Integrate PBay Spectrum with actualy Alluxa profile to get Measurements
# To make spectrum, go to Kavli research folder where PBay is stored
# Outputs: Plots saved to plots folder
# Inputs: Spectra from pbay + their log should be saved to spectra folder
# plot planets from Kepler and TESS
###############################################################

import numpy as np
import matplotlib.pylab as plt
from scipy.interpolate import interp1d
from scipy.integrate import trapz
import matplotlib,sys,os
from astropy.io import fits
from astropy.table import Table
plt.ion()

font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 18,
        'sans-serif'    : 'Oswald'}

matplotlib.rc('font', **font)
matplotlib.rc('axes',linewidth=2)

sys.path.append('./utils/')
#from functions import *
data_dir = '../data/'


######################### LOAD THINGS #####################
# load exoplanet archive hot jupiters
hj = Table.read(data_dir + 'exoplanet/exoplanet_archive_hot_jupiters2.txt',format='ascii')

# Get key things
vmag = hj['V']
kb = 1.38064852e-23 #m2 kg s-2 K-1
mu = 2.3 * 1.6605390e-27 # kg
G = 6.67408e-11 #m3 kg-1 s-2

# estimate Temp planet in kelvin
Teq = (1/4.)**(1/4.) * hj['TEFF'] * np.sqrt(0.00465047 * hj['RSTAR']/hj['A']) # 0.00465047 AU per Rsun
gravity = G * hj['MASS'] * 1.898e27 / (hj['R'] * 69911000.0)**2 #kg from m_jup

# get H -> kb * T/ mu / g
H = kb * Teq / mu / gravity # meters

# calculate A like Sing did
A = 2* hj['R']*0.10049 * H*1.4374e-9 / hj['RSTAR']**2 #0.10049 rsun/rjupiter, 1.4374e-9 rsun/meters


#####################################
# Step through code changing Vmag in range - calculate and save noise
#####################################
tot_eff = 0.25
ntransits= 1
target_snr = 7
tel_diam   = 3
#------------------
def calc_s(tot_eff,ntransits,target_snr,tel_diam,dl_l=0.0012):
	"""
	calc 
	"""
	tel_area = np.pi * (tel_diam/2.)**2
	exp_one_transit = 2.3 * 3600.
	vmags = np.arange(5,17,0.2)
	s_3sig=  np.zeros(len(vmags))
	for i in range(len(vmags)):
		signal = tot_eff * 10**(-0.4*vmags[i]) * 3640*1.51e7 *tel_area * exp_one_transit *dl_l
		noise = 2/np.sqrt(signal)
		s_3sig[i] = target_snr*noise/np.sqrt(ntransits)

	return vmags,s_3sig

vmags,hubble   = calc_s(0.1, 3, 3, 3.0,dl_l=1/100.)
_,onefivemeter = calc_s(0.8, 3, 3, 1.5)
_,mred         = calc_s(0.8, 5, 3, 0.7)
_,mmt          = calc_s(0.7, 1, 3, 6.5)
_,hale          = calc_s(0.7, 1, 3, 5.1)
_,gmt          = calc_s(0.7,  3, 5, 25.0)

#####################################
# PLOT
#####################################
fig, ax = plt.subplots(figsize=(10,8))
plt.semilogx(2.5*A*100,vmag,'ko',ms=3)
plt.ylim(15,7.2)
plt.xlim(8e-4,0.45)
plt.plot(mred*100,vmags,'r--',lw=.6)
plt.plot(hale*100,vmags,'r--',lw=.6)
plt.plot(onefivemeter*100,vmags,'r--',lw=.6)
#plt.plot(gmt*100,vmags,'r--',lw=.6)
#plt.plot(hubble*100*2.5,vmax, y, s, rotation=45gs,'r--')
plt.text(0.19, 12.3, '0.7m (5)', rotation=-56,fontsize=10,color='r')
plt.text(0.149, 12.8, '1.5m (3)', rotation=-56,fontsize=10,color='r')
plt.text(0.1, 13.8, '5.1m (1)', rotation=-56,fontsize=10,color='r')
for i in range(len(A)): # labelplanets
	if np.isnan(A[i]):
		pass
	else:
		if (vmag[i] < 15) & (2.51*A[i]*100 > 8e-4):
			plt.text(2.51*A[i]*100,vmag[i],hj['NAME'][i],fontsize=4,color='steelblue')

plt.grid(color='lightgray',linestyle='--',zorder=-1000)
ax.tick_params(axis='both',direction='in',width=1.3,which='both',length=3,pad=5)
plt.xlabel('Transit Signal (%)',fontweight='bold',fontstyle='italic')
plt.ylabel('V magnitude',fontweight='bold',fontstyle='italic')

# shade for hubble
plt.fill_between(hubble*100*2.5,vmags,y2 = vmags - 10, facecolor='g',alpha=.1,zorder=-100)
plt.text(0.12, 8.46, 'Hubble (3)', fontsize=12,color='g')
plt.savefig('plots/Sing_inspired_nsf.pdf')



