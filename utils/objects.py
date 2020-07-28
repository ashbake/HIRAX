import configparser,glob
import numpy as np
from distutils.util import strtobool

all = {'storage_object','load_object'}


class storage_object():
    """
    Main storage object for organization
    """
    def __init__(self):
        # Classes
        self.run = RUN()
        self.tel = TELLURIC()
        self.exo = EXOPLANET()
        self.stel= STELLAR()
        self.out = OUTPUT()
        self.filt = FILTERS()
        self.inst = INSTRUMENT()
        self.const = CONSTANTS()
        # non class things
        self.info = "see objects.py in tools/ for info"
        self.input_path = '../data/'
        self.output_path = './output/'
        self.plot_path   = './plot/'
        self.plot_prefix = 'default'
        
class RUN():
	""" Define which directory and files are currently loaded"""
	def __init__(self):
		# User defined:
		self.ndualons    = None  # Number of dualons to use
		self.start_lam   = None # lambda in nanometers of first dualon (blue side)
		self.nbands      = None  # Number of bands to do
		self.amplitude   = None  # amplitude of band pass
		self.overwrite   = False  # overwrite output file?
		
class EXOPLANET():
    def __init__(self):
        self.folder   = 'exoplanet/' # Where transmission spec stored
        self.name     = 'H2O_O2_Earth_spectrum_nwidth_150_750_765_res_0,008_newpar.dat' # Name of transmission spectrum to load
        self.hitran_h2o    = 'h2o_9000_20000.par'
        self.v       = None # wavelength
        self.s       = None # transmission
        self.vel     = None # velocity if shifted
        self.dist_orb= 1.496e8 # km, planet to star orbital distance
        self.radius  = 6371. # km, radius of planet
        
class FILTERS():
    def __init__(self):
        self.folder 	      = 'filters/'
        self.dualon_raw_name  = 'dualonTransmission.dat'
        # dualon passband properties
        self.amp         = None  # amplitude of transmission (same for both)
        self.nbands      = 12  # number of bands to cover (red: max 19, blue: max ?)
        self.band        = 'red' # red or blue (or can add a new one) if want to do blue end or red end
        # Output dualon passband arrays
        self.dualon_raw            = None # Wavelength of raw dualon
        self.s_dual_red            = None #  dualon transmission
        self.s_dual_blue           = None #  dualon transmission blue end
        self.v_dual                = None # dualon v same as exo
        # Oxyometer alluxa/baker 2018 bandpass for comparison
        self.name_alluxa      = 'Alluxa_f607.4_0.3.txt'
        self.s_alluxa_on         = None # Normal alluxa filter shifted to edge of o2 line
        self.s_alluxa_off        = None # Normal alluxa filter shifted to edge of o2 line
        self.s_alluxa_wide_on    = None # 2nm wide filter shifted to o2 blue edge
        self.s_alluxa_wide_off   = None # 2nm wide filter shifted to o2 blue edge

        # I band stuff
        self.name_I             = 'Generic_Bessell.I.dat'
        self.I_raw              = None # loaded I band (not interpolated)
        self.zp_I               = 2550.0 # Jansky zero point for I johnson cousins band
        self.dl_l_I             = None # d lambda over lambda
        self.v_I				= None # wavelength of r band
        self.s_I                = None # transmission out of 1
        
             
class STELLAR():
    "stellar model for fitting..start with Kurucz but then change" 
    "once make own stellar generated spec"
    def __init__(self):
        # User optional define:
        self.folder = 'phoenix/' # Path to stellar spec
        self.name   = None       # stellar spec file name
        # Stellar type picked
        self.type = None # str of M#
        self.temp = None
        self.rad  = None # radius of star
        self.dist = None # distance to star
        self.I_mag = None # I band magnitude
        self.vel   = 0 # stellar radial velocity in km/s
        self.fluxunits = None # flux units from phoenix
        # Filled in by code:
        self.v = None # wavelength like normal (should match exoplanet and be in standard wavelength)
        self.s = None #  spectrum
        self.units = None # units of stellar spectrum
        self.notes = None # Notes about stellar spectrum calc
        # Stellar contamination
        self.folder_contamination=None
        self.contamination  = None # load rackham stellar models interpolated to exoplanet v grid
        self.contamination2 = None #  "" ...need to figure out which model is which...worse one is smaller spots


class OUTPUT():
    "stellar model for fitting"
    def __init__(self):
        # User optional defined
        self.folder = './outputs/'              # Path to atlas
        self.savename = None # output name
        # Filled in by code
        self.signal = None # wavenumber like normal
        self.star   = None # Star signal at observer in phot/m2/s/nm
        self.planet = None # planet signal at observer
        self.g_phase= None # filling fraction calc. from phase and inclination inputs

class INSTRUMENT():	
    "stellar model for fitting"
    def __init__(self):
        # telescope, default GMT
        self.telescope     = 'ELT' # telescope name
        self.tel_area      = 978.0 # telescope area (m^2)
        self.focal_ratio   = 8.2 # focal ratio  
        # ccd stuff mostly
        self.qe     = 0.8  # CCD QE
        self.tel_reflectivity = 0.7 # telescope reflectivity
        self.read   = None # CCD read
        self.photon = None # photon noise
        self.dark   = None # dark noise
        self.phot_aperture = None   # photometry aperture
        self.image_size    = None   # size of star on CCD
        self.exp_time      = 15*60. # exposure time (s) default 15 min


class TELLURIC():
    def __init__(self):
        """
        TAPAS generated spectra
        When generate files, all v values should be the same sampling for
        all the species
        """
        self.load_tapas = True # if false, won't load the tapas stuff
        self.name_hitran  = 'Hitran/O2_A_band.par'
        self.hitran   = None # Loaded hitran file of o2 line positions
        self.folder   = 'telluric/'
        self.name_o2  = 'tapas_oxygen.fits'
        self.name_h2o = 'tapas_water.fits'
        self.name_rayleigh = 'tapas_rayleigh.txt'
        self.v      = None # wavenumber
        self.h2o    = None
        self.o2     = None
        self.rayleigh = None
        self.pwv    = None # Precipitable water vapor of generated water spectrum
        self.name_oh = 'OHlineList.txt'
        self.x_oh   = None # OH list
        self.y_oh   = None # Oh line list y value
        
class CONSTANTS():
    "useful constants"
    def __init__(self):
        self.pc_to_m   = 3.08567758e16 #m per 1 pc
        self.rsun = 695508.0 # Km
        self.rearth = 6371.0 # km
        self.rjup   = 69911.0 #km
        self.types             = np.array(['G2','M0','M1','M2','M3','M4','M5','M6','M7','M8','M9'])
        self.radii             = np.array([1.0,0.62,0.49,0.44,0.39,0.26,0.20,0.15,0.12,0.10,0.08])
        self.radlows           = 6371 + np.array([12.7,5.15,3.5,2.79,1.0,0.66,0,0,0,0,0])
        self.abs_mag           = np.array([4.1,7.1,7.7,8.3,8.8,10.0,11.2,12.4,13.6,13.9,14.7])
        self.transit_durations = np.array([13.1 ,5.37,3.96,3.36,2.96,2.06,1.50,1.07,0.78,0.69,0.43])
        self.yeartransits      = np.array([1.0, 5.6,8.4,11.1,13.8,21.8,36.7,59.7,89.1,108.1,191.8])
        self.colors  		   = ['gray','brown','gold','r','c','m','gold','g','darkslateblue','brown','k','darkorange']
        self.symbols 		   = ['-','-.','--',':','-','-.','-.','-','-','--','--',':']
        self.telescopes        = ['GMT','TMT','ELT','Hale']
        self.tel_areas         = [368.0 , 655.0 , 978.0, 20.0] #m2
        self.focal_ratios      = [8.2,8.2,8.2, 3.3] # input actual ones
    	


def LoadConfig(configfile, config={}):
    """
    Reads configuration file 'XXX.cfg'
    returns a dictionary with keys of the form
    <section>.<option> and the corresponding values
    """
    config = config.copy(  )
    cp = configparser.ConfigParser(  )
    cp.read(configfile)
    for sec in cp.sections(  ):
        name = str.lower(sec)
        for opt in cp.options(sec):
            config[name + "." + str.lower(opt)] = str.strip(
                cp.get(sec, opt))
    return config


def load_object(configfile):
	"""
	Loads config file as dictionary using LoadConfig function
	Then loads stoar_object and fills in user-defined
	quantities
	"""
	config = LoadConfig(configfile)
	so = storage_object()
	
	for key in config:
		s1,s2=key.split('.')
		setattr(getattr(so,s1),s2,config[key])
	
	if type(so.run.overwrite) is str:
		so.run.overwrite = strtobool(so.run.overwrite)
	
	so.filt.nbands = int(so.filt.nbands)
	so.stel.dist   = float(so.stel.dist)
	so.stel.vel    = float(so.stel.vel)
	
	return so
	



