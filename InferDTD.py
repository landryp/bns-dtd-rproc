#!/usr/bin/env python3
# coding: utf-8

'INFERDTD.IPYNB -- infer BNS delay time distribution from stellar abundance data'
__usage__ = 'InferDTD.py outdir eufepath poppath --nlikes 100 --maxnum 100000 --disk False --parts 10'
__author__ = 'Philippe Landry (pgjlandry@gmail.com)'
__date__ = '09-2023'


### PRELIMINARIES


# load packages

from argparse import ArgumentParser
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import multivariate_normal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import h5py
from tqdm import tqdm
import astropy.units as u
from astropy.coordinates import Distance
from astropy.cosmology import Planck15 as cosmo
from astropy.cosmology import z_at_value

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
sns.set_palette('Set1',color_codes=True)


parser = ArgumentParser(description=__doc__)
parser.add_argument('outdir')
parser.add_argument('eufepath')
parser.add_argument('poppath')
parser.add_argument('-n','--nlikes',default=100)
parser.add_argument('-m','--maxnum',default=None)
parser.add_argument('-d','--disk',default=False)
parser.add_argument('-p','--parts',default=None)
parser.add_argument('-t','--tag',default=None)
args = parser.parse_args()

OUTDIR = str(args.outdir) # 'dat/' # output directory for plots, likelihood-weighted population samples
EUFEPATH = str(args.eufepath) # 'data/SAGA_MP.csv' # observations of stellar Eu vs Fe abundances
POPPATH = str(args.poppath) # 'dat/EuFe_bnscls-10000.part.h5' # path to population samples
NLIKES = int(args.nlikes) # number of likelihood samples to draw per observation
if args.maxnum is not None:
	NUM = int(args.maxnum) # number of abundance predictions to evaluate likelihood for
else: NUM = None
if args.disk == 'False' or args.disk is False:
	DISK = False # whether to restrict to disk stars ([Fe/H] > -1.)
else: DISK = True
if args.parts is not None:
	PARTS = int(args.parts) # number of chunks population samples are split into, assuming 'part0', 'part1', etc labeling
else: PARTS = None
if args.tag is not None:
	TAG = str(args.tag) # optional tag for output file
else: TAG = None

if not os.path.exists(OUTDIR):
	os.makedirs(OUTDIR)


### BUILD LIKELIHOOD FUNCTION FOR OBSERVATIONS


# load SAGA data

FeHs, EuFes, FeH_errs, EuFe_errs = np.loadtxt(EUFEPATH, unpack=True, delimiter=',', skiprows=1)


# examine distribution of errors and compute means

avg_FeH_err = np.mean(FeH_errs[FeH_errs>0.])
avg_EuFe_err = np.mean(EuFe_errs[EuFe_errs>0.])


# impose disk/halo star cut if desired, and use average errors for data points without error bars

if DISK: FE_CUT = -1. # cut at [Fe/H] > -1., disk stars only
else: FE_CUT = -1e16 # no cut, halo+disk

FeHs_in, EuFes_in, FeH_errs_in, EuFe_errs_in = [], [], [], [] # replace data with no errors
for FeH, EuFe, FeH_err, EuFe_err in zip(FeHs, EuFes, FeH_errs, EuFe_errs):
    
    if FeH > FE_CUT:
    
        FeHs_in += [FeH]
        EuFes_in += [EuFe]
    
        if FeH_err > 0. and EuFe_err > 0.:
            FeH_errs_in += [FeH_err]
            EuFe_errs_in += [EuFe_err]

        else:
            FeH_errs_in += [avg_FeH_err]
            EuFe_errs_in += [avg_EuFe_err]
        
FeHs, EuFes, FeH_errs, EuFe_errs = np.array(FeHs_in), np.array(EuFes_in), np.array(FeH_errs_in), np.array(EuFe_errs_in)


# make gaussian likelihood model for each SAGA datapoint

saga_dat_likes = []
saga_like_means = []
saga_like_stds = []

for fe,eu,fe_err,eu_err in zip(FeHs, EuFes, FeH_errs, EuFe_errs):

    mean = np.array([fe,eu])
    std = np.array([[fe_err,0.],[0.,eu_err]])
    
    saga_like_means += [mean]
    saga_like_stds += [std]   
    saga_dat_likes += [np.random.multivariate_normal(mean,std,NLIKES)]
    
like_funcs = [multivariate_normal(mean,std) for mean,std in zip(saga_like_means,saga_like_stds)]


### DO INFERENCE OF BNS DTD AND COLLAPSAR CONTRIBUTION


# load abundance predictions, map likelihoods to delay time distribution parameters and calculate likelihood for abundance predictions

alphas, tmins, xcolls, rates, mejs = [], [], [], [], []
log_like = []

k = 0

if PARTS is None:
    
    INPUTPATH = POPPATH

    inputdat = h5py.File(INPUTPATH, 'r')
    pop_dat_i = inputdat['pop']
    yield_dat_i = inputdat['yield']
    
    for j in tqdm(range(len(pop_dat_i))):
    
        pop_dat = pop_dat_i[str(j)]
        yield_dat = yield_dat_i[str(j)]
        
        alphas += [-pop_dat['b']]
        tmins += [pop_dat['tmin']]
        xcolls += [pop_dat['X_coll']]
        rates += [pop_dat['rate']]
        mejs += [pop_dat['m_ej']]
        
        curve = np.column_stack((yield_dat['Fe_H'],yield_dat['Eu_Fe']))
        log_likes = [np.log(np.trapz(like_func.pdf(curve),curve[:,0])) for like_func in like_funcs]
        log_like += [np.sum(log_likes)]
        k += 1
        
        if NUM is not None: 
            if k >= NUM: break
    
else:
    
    keys = list(np.arange(PARTS))

    for key in tqdm(keys):
        INPUTPATH = '.'.join(POPPATH.split('.')[:-1])+'.part{0}'.format(key)+'.'+POPPATH.split('.')[-1]

        inputdat = h5py.File(INPUTPATH, 'r')
        pop_dat_i = inputdat['pop']
        yield_dat_i = inputdat['yield']

        for j in tqdm(range(len(pop_dat_i))):

            pop_dat = pop_dat_i[str(j)]
            yield_dat = yield_dat_i[str(j)]

            alphas += [-pop_dat['b']]
            tmins += [pop_dat['tmin']]
            xcolls += [pop_dat['X_coll']]
            rates += [pop_dat['rate']]
            mejs += [pop_dat['m_ej']]

            curve = np.column_stack((yield_dat['Fe_H'],yield_dat['Eu_Fe']))
            log_likes = [np.log(np.trapz(like_func.pdf(curve),curve[:,0])) for like_func in like_funcs]
            log_like += [np.sum(log_likes)]
            k += 1
            
            if NUM is not None: 
                if k >= NUM: break

npops = k

if NUM is None: NUM = npops
    

### SAVE RESULTS

   
# map likelihoods to delay time distribution parameters

log_wts = log_like-np.max(log_like)


# save delay time distribution parameter likelihoods

if TAG is None: OUTPATH = OUTDIR+'/'+(EUFEPATH.split('/')[-1]).split('.')[0]+'_{0}.csv'.format(NUM)
else: OUTPATH = OUTDIR+'/'+(EUFEPATH.split('/')[-1]).split('.')[0]+'_{0}.{1}.csv'.format(NUM,TAG)

outdat = np.column_stack((alphas,tmins,xcolls,mejs,rates,log_like))
np.savetxt(OUTPATH,outdat,delimiter=',',comments='',header='alpha,tmin,xcoll,mej,rate,log_likelihood')
