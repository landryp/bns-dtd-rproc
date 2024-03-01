#!/usr/bin/env python3
# coding: utf-8


'CALCULATEENRICHMENTHISTORY.PY -- calculate Eu vs Fe abundance history from binary neutron star merger rate, ejecta, delay time distribution and fractional contribution relative to a second channel, using Daniel Siegel\'s one-zone r-process nucleosynthesis code'
__usage__ = 'CalculateEnrichmentHistory.py nsamp outpath dtdpath ejpath --alpha alpha_min,alpha_max --tmin tmin_min,tmin_max --xsfh xsfh_min,xsfh_max --nmarg nmarg'
__author__ = 'Philippe Landry (pgjlandry@gmail.com)'
__date__ = '09-2023'


### PRELIMINARIES


# load packages

from argparse import ArgumentParser
import numpy as np
import scipy
import h5py
import numpy.lib.recfunctions as rfn
from tqdm import tqdm

from etc.rProcessChemicalEvolution import rproc_evolution
from etc.rProcessUtils import * # import Daniel Siegel's one-zone r-process code


parser = ArgumentParser(description=__doc__)
parser.add_argument('nsamp')
parser.add_argument('outpath')
parser.add_argument('dtdpath')
parser.add_argument('ejpath')
parser.add_argument('-s','--solpath',default='etc/Arnould07_solar_rprocess.dat')
parser.add_argument('-a','--alpha',default="-3.,-0.5")
parser.add_argument('-t','--tmin',default="1e-2,2.01")
parser.add_argument('-x','--xsfh',default="1e-3,0.999")
parser.add_argument('-m','--nmarg',default=500)
args = parser.parse_args()

NUM = int(args.nsamp) # 100 # number of abundance predictions to calculate -- equals number of BNS DTD samples to draw
OUTPATH = str(args.outpath) # 'dat/EuFe_grbbnscls-100.h5' # outpath path for Eu vs Fe abundance curves, population samples
DTDPATH = str(args.dtdpath) # 'etc/label_samples.dat' # GRB-informed DTD parameter distributions # '' # uniform DTD parameter distribution
EJECTAPATH = str(args.ejpath) # 'etc/mej_gal_lcehl_nicer_numuncertainty.txt' # input samples in ejecta mass and rate
SOLARPATH = str(args.solpath) # 'etc/arnould_07_solar_r-process.txt' # solar abundances
ALPHA_BOUNDS = [float(bnd) for bnd in str(args.alpha).split(',')] # (-3.,-0.5) # bounds for uniform prior DTD power law index
TDMIN_BOUNDS = [float(bnd) for bnd in str(args.tmin).split(',')] # (1e-2,2.01) # bounds for log-uniform prior on minimum delay time in Gyr # see below for option to change to uniform prior
EU_RATIO_Z0_BOUNDS = [float(bnd) for bnd in str(args.xsfh).split(',')] # (1e-6,1.) # bounds for log-uniform prior on fractional contribution of collapsar channel to local r-process mass*rate density (i.e. m_Eu_coll*norm_coll/(m_Eu_coll*norm_coll + m_Eu_bns*norm_bns)) # see below for option to change to uniform prior
NUM_MARG = int(args.nmarg) # 100 # number of mej and rate samples per DTD sample


### SAMPLE FROM PRIORS


# load ejecta mass and rate samples

mej_dyn, mej_dsk, rate = np.loadtxt(EJECTAPATH, unpack=True)
mej = mej_dyn + mej_dsk
rate = rate*(1e9*1e-2)/1e6 # convert to Gpc^-3 yr^-1


# sample in BNS delay time distribution

ndtd = NUM # number of DTD samples to draw

if DTDPATH != '':
    
	alpha, tdmin, tdmax = np.loadtxt(DTDPATH, unpack=True, skiprows=1)
	tdmin = tdmin/1e9 # convert to Gyr
    
	idxs = np.random.choice(range(len(alpha)),ndtd,False)
	alpha = alpha[idxs]
	tdmin = tdmin[idxs]

else:
    
    alpha = np.random.uniform(*ALPHA_BOUNDS,ndtd) # uniform prior on DTD power law index
    tdmin = scipy.stats.loguniform.rvs(*TDMIN_BOUNDS,size=ndtd) # log-uniform prior on minimum delay time in Gyr # np.random.uniform(*TDMIN_BOUNDS,ndtd) # uniform prior on minimum delay time in Gyr


# sample in collapsar yield

# Eu_ratio_z0 = scipy.stats.loguniform.rvs(*EU_RATIO_Z0_BOUNDS,size=ndtd) # log-uniform prior on ratio of collapsar to total Eu abundance at z=0 
Eu_ratio_z0 = np.random.uniform(*EU_RATIO_Z0_BOUNDS,ndtd) # uniform prior on ratio of collapsar to total Eu abundance at z=0


### DO R-PROCESS ABUNDANCE CALCULATION


# settings for r-process yield calculation

key_SFR='MF17' # star formation rate

m_Mg = 0.12 # Mg yield per enrichment event in Msun
m_alpha = m_Mg # He

nppdt = 20 # number of timesteps in chemical evolution integration
t_int_min = t_z(zmax_int) # minimum cosmic time to start integration
t_sun = t_z(0.) - age_sun # cosmic time at birth of sun in Gyr
tmin = t_z(zmax_int)
tmax = t_z(0.) # age of the universe
tmin_intMW = tmin

# supernovae

m_Fe_cc = 0.074 # CCSN Fe yield per enrichment event in Msun
m_Fe_Ia = 0.7 # IaSN Fe
m_Eu_MHDSN = 1.4e-5  # MHD SN Eu

P_MHDSN = 0.3  # percentage of MHD SN among CCSN
C_Ia = 1.3e-3 # calibration for IaSN

b_Ia = 1.0 # IaSN DTD power law exponent
tmin_Ia = 0.4 # DTD tmin for IaSN in Gyr

key_PDF_vkick='exp' # kick distribution
v_mean = 180. # mean kick velocity in [km/s]

R_cc_z0 = 0.705 * 1e-4 * 1e9 # local volumetric CCSN rate
R_MHDSN_z0 = P_MHDSN * 1e-2 * R_cc_z0 # local volumetric MHD SN rate

# collapsars

Asun, Nsun = loadtxt(SOLARPATH, unpack=True) # Nsun is number of atoms, normalized to 10^6 Si atoms (i.e., N_Si = 1e6)
Asun = Asun[Nsun > 0.]
Nsun = Nsun[Nsun > 0.]

Xsun = 0.7110 * Nsun * Asun / 2.431e10 # Lodders 2003, Table 2, proto-solar abundances ("solar system abundances")
X_r_tot69 = sum(Xsun[Asun >= 69]) # 1st peak # hydrogen mass fraction: X_H0 = 0.7110

Xsun_Eu = Xsun[Asun == 151] + Xsun[Asun == 153] # solar mass fraction of all Eu isotopes
Xsun_Eu = Xsun_Eu[0]
Xsun_Eu_r69 = Xsun_Eu / X_r_tot69 # Eu mass fraction for r-process starting at A=69

m_Fe_coll = 0.0 # collapsar Fe -- keep this zero

# galactic enrichment

X_H = 0.75 # ISM H mass fraction
eta = 0.25 # outflow rate (<1) normalized by star formation rate ("o" in Hotokezaka 2018)

r_eff_MW = 6.7e3 * pc * 1e-3 # effective Milky Way radius in km
f_r_eff_MW = 2. # boundary factor for r-process enrichment of Milky Way
R_enc = 0.5 *r_eff_MW * f_r_eff_MW # radius within which events contribute to Milky Way r-process enrichment

normalize_to_observed_solar_values = 1


# match up ejecta, DTD and collapsar fraction samples to make population realizations

nsamps = NUM_MARG # number of rate, mej samples per r-process yield prediction

if NUM_MARG == 1:
    
    mejs = np.array([0.027]) # approximate mode
    rates = np.array([123.])
    
else:

    ej_idxs = np.random.choice(range(len(mej)),nsamps,False) # sample in ejecta mass and rate
    mejs = mej[ej_idxs]
    rates = rate[ej_idxs]

fNSs = np.full(nsamps,0.5) #np.random.uniform(0.,1.,nsamps) # uniform prior on BNS enrichment efficiency -- marginalize over this #np.random.choice(fNS,nsamps,False) # sample in BNS enrichment efficiency

dtd_idxs = np.arange(ndtd) #np.random.choice(range(len(alpha)),ndtd,False) # sample in delay time distribution
alphas = alpha[dtd_idxs]
tdmins = tdmin[dtd_idxs]
Eu_ratio_z0s = Eu_ratio_z0[dtd_idxs] #np.random.choice(Eu_ratio_z0,nsamps,False) # sample in collapsar Eu abundance fractions


# do r-process yield calculation, using Daniel Siegel's one-zone model

Fe_H_list = []
r_Fe_list = []
Rbns_of_t_list = []
Rcoll_of_t_list = []
X_of_t_list = []
t_list = []
bns_dat = np.zeros((ndtd*nsamps,6))

for i,(alpha,tdmin,Xcoll) in tqdm(enumerate(zip(alphas,tdmins,Eu_ratio_z0s))):

	for j,(mej,rate,f_NSgal) in enumerate(zip(mejs,rates,fNSs)):

		Fe_H, r_Fe, R_of_ts = rproc_evolution(rate,mej,-alpha,tdmin,Xcoll,f_NSgal,nppdt)
		Fe_H_list += [Fe_H]
		r_Fe_list += [r_Fe]
		Rbns_of_t_list += [R_of_ts[0]]
		Rcoll_of_t_list += [R_of_ts[1]]
		X_of_t_list += [R_of_ts[2]] # FIXME: this is X of production at z, but we want cumulative X up to z
		t_list += [R_of_ts[3]]

		bns_dat[nsamps*i+j,:] = array([mej,rate,-alpha,tdmin,f_NSgal,Xcoll])

    
### SAVE RESULTS
    
    
# save r-process abundance data and parameters for population realizations

outdat = {}
outdat['pop'] = {}
outdat['yield'] = {}
outdat['frac'] = {}

for i in range(nsamps*ndtd):

    outdat['pop'][i] = bns_dat[i]
    outdat['yield'][i] = {}
    outdat['yield'][i]['Fe_H'] = Fe_H_list[i]
    outdat['yield'][i]['Eu_Fe'] = r_Fe_list[i]
    outdat['frac'][i] = {}
    outdat['frac'][i]['Rbns'] = Rbns_of_t_list[0]
    outdat['frac'][i]['Rcoll'] = Rcoll_of_t_list[0]
    outdat['frac'][i]['X'] = X_of_t_list[0]
    outdat['frac'][i]['t'] = t_list[0]

outfile = h5py.File(OUTPATH, 'w')

pop_set = outfile.create_group('pop')
for key, value in outdat['pop'].items():
    pop_data = np.array(value)
    pop_data = rfn.unstructured_to_structured(pop_data, np.dtype([('m_ej', 'f8'), ('rate', 'f8'), ('b', 'f8'), ('tmin', 'f8'), ('f', 'f8'), ('X_coll', 'f8')]))
    pop_set.create_dataset(str(key),data=pop_data)

yield_set = outfile.create_group('yield')
for key, value in outdat['yield'].items():
    yield_data = np.column_stack((value['Fe_H'],value['Eu_Fe']))
    yield_data = rfn.unstructured_to_structured(yield_data, np.dtype([('Fe_H', 'f8'), ('Eu_Fe', 'f8')]))
    yield_set.create_dataset(str(key),data=yield_data)
    
frac_set = outfile.create_group('frac')
for key, value in outdat['frac'].items():
    frac_data = np.column_stack((value['Rbns'],value['Rcoll'],value['X'],value['t']))
    frac_data = rfn.unstructured_to_structured(frac_data, np.dtype([('Rbns', 'f8'), ('Rcoll', 'f8'), ('X', 'f8'), ('t', 'f8')]))
    frac_set.create_dataset(str(key),data=frac_data)

outfile.close()
