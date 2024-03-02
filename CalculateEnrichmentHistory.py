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
from scipy.stats import loguniform
from scipy.integrate import cumtrapz
import h5py
import numpy.lib.recfunctions as rfn
from tqdm import tqdm

from etc.rProcessChemicalEvolution import rproc_evolution, Xsun_Eu_r69
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
	alphas = alpha[idxs]
	tdmins = tdmin[idxs]

else:
    
    alphas = np.random.uniform(*ALPHA_BOUNDS,ndtd) # uniform prior on DTD power law index
    tdmins = loguniform.rvs(*TDMIN_BOUNDS,size=ndtd) # log-uniform prior on minimum delay time in Gyr # np.random.uniform(*TDMIN_BOUNDS,ndtd) # uniform prior on minimum delay time in Gyr


# sample in collapsar yield

# Eu_ratio_z0 = loguniform.rvs(*EU_RATIO_Z0_BOUNDS,size=ndtd) # log-uniform prior on ratio of collapsar to total Eu abundance at z=0 
Eu_ratio_z0s = np.random.uniform(*EU_RATIO_Z0_BOUNDS,ndtd) # uniform prior on ratio of collapsar to total Eu abundance at z=0


# sample in BNS rate and ejecta

nsamps = NUM_MARG # number of rate, mej samples per r-process yield prediction

if NUM_MARG == 1:
    
    mejs = np.array([0.027]) # approximate mode
    rates = np.array([123.])
    
else:

    ej_idxs = np.random.choice(range(len(mej)),nsamps,False) # sample in ejecta mass and rate
    mejs = mej[ej_idxs]
    rates = rate[ej_idxs]


### DO R-PROCESS ABUNDANCE CALCULATION


# do r-process yield calculation, using Daniel Siegel's one-zone model

outdat = {}
outdat['pop'] = {}
outdat['yield'] = {}
outdat['frac'] = {}

for i,(alpha,tdmin,Xcoll) in tqdm(enumerate(zip(alphas,tdmins,Eu_ratio_z0s))):
	for j,(mej,rate) in enumerate(zip(mejs,rates)):

		idx = nsamps*i+j

		Fe_H, r_Fe, rate_evolutions = rproc_evolution(rate,mej,-alpha,tdmin,Xcoll,f_NSgal=0.5,nppdt=20)
		
		Rbns, Rcoll, Xs, zs, ts = rate_evolutions
		
		outdat['pop'][idx] = np.array([mej,rate,alpha,tdmin,Xcoll])
		outdat['yield'][idx] = {}
		outdat['yield'][idx]['Fe_H'] = Fe_H
		outdat['yield'][idx]['Eu_Fe'] = r_Fe
		outdat['frac'][idx] = {}
		outdat['frac'][idx]['Rbns'] = Rbns
		outdat['frac'][idx]['Rcoll'] = Rcoll
		outdat['frac'][idx]['z'] = zs
		outdat['frac'][idx]['t'] = ts
		
		Rcoll0 = Rcoll[-1]
		mcoll = mej*Xsun_Eu_r69*rate/((1./Xcoll - 1.)*Rcoll0)
		num = cumtrapz(mej*Rbns,ts)
		denom = cumtrapz(mcoll*Rcoll,ts)
		Xts = 1./(1.+num/denom)
		
		outdat['frac'][idx]['X'] = list(Xts)+[Xts[-1]]


### SAVE RESULTS
    
    
# save r-process abundance data and popoulatio parameters

outfile = h5py.File(OUTPATH, 'w')

pop_set = outfile.create_group('pop')
for key, value in outdat['pop'].items():
    pop_data = np.array(value)
    pop_data = rfn.unstructured_to_structured(pop_data, np.dtype([('mej', 'f8'), ('rate', 'f8'), ('alpha', 'f8'), ('tmin', 'f8'), ('X0', 'f8')]))
    pop_set.create_dataset(str(key),data=pop_data)

yield_set = outfile.create_group('yield')
for key, value in outdat['yield'].items():
    yield_data = np.column_stack((value['Fe_H'],value['Eu_Fe']))
    yield_data = rfn.unstructured_to_structured(yield_data, np.dtype([('Fe_H', 'f8'), ('Eu_Fe', 'f8')]))
    yield_set.create_dataset(str(key),data=yield_data)
    
frac_set = outfile.create_group('frac')
for key, value in outdat['frac'].items():
    frac_data = np.column_stack((value['Rbns'],value['Rcoll'],value['X'],value['z'],value['t']))
    frac_data = rfn.unstructured_to_structured(frac_data, np.dtype([('Rbns', 'f8'), ('Rcoll', 'f8'), ('X', 'f8'), ('z', 'f8'), ('t', 'f8')]))
    frac_set.create_dataset(str(key),data=frac_data)

outfile.close()
