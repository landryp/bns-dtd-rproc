#!/usr/bin/env python3
# coding: utf-8

'INFERBNSDTD.IPYNB -- infer binary neutron star delay time distribution parameters and fractional second-channel contribution from galactic r-process abundance observations and Eu vs Fe abundance histories'
__usage__ = 'InferDTD.py outdir obspath eufepath --maxnum maxnum --parts parts'
__author__ = 'Philippe Landry (pgjlandry@gmail.com)'
__date__ = '09-2023'


### PRELIMINARIES


# load packages

from argparse import ArgumentParser
import numpy as np
from scipy.stats import multivariate_normal
from scipy.integrate import cumtrapz
import os
import h5py
from tqdm import tqdm

parser = ArgumentParser(description=__doc__)
parser.add_argument('outdir')
parser.add_argument('obspath')
parser.add_argument('eufepath')
parser.add_argument('-m','--maxnum',default=None)
parser.add_argument('-p','--parts',default=None)
parser.add_argument('-t','--tag',default=None)
args = parser.parse_args()

OUTDIR = str(args.outdir) # 'dat/' # output directory for plots, likelihood-weighted population samples
OBSPATH = str(args.obspath) # 'data/SAGA_MP.csv' # observations of stellar Eu vs Fe abundances
EUFEPATH = str(args.eufepath) # 'dat/EuFe_bnscls-10000.part.h5' # path to population samples
if args.maxnum is not None:
	NUM = int(args.maxnum) # number of abundance predictions to evaluate likelihood for
else: NUM = None
if args.parts is not None:
	PARTS = int(args.parts) # number of chunks population samples are split into, assuming 'part0', 'part1', etc labeling
else: PARTS = None
if args.tag is not None:
	TAG = str(args.tag) # optional tag for output file
else: TAG = None

if not os.path.exists(OUTDIR):
	os.makedirs(OUTDIR)


### BUILD LIKELIHOOD FUNCTIONS FOR OBSERVATIONS


# load SAGA data

FeHs, EuFes, FeH_errs, EuFe_errs = np.loadtxt(OBSPATH, unpack=True, delimiter=',', skiprows=1)


# make gaussian likelihood model for each SAGA datapoint

saga_like_means = []
saga_like_stds = []

for fe,eu,fe_err,eu_err in zip(FeHs, EuFes, FeH_errs, EuFe_errs):

    mean = np.array([fe,eu])
    std = np.array([[fe_err,0.],[0.,eu_err]])
    
    saga_like_means += [mean]
    saga_like_stds += [std]   
    
like_funcs = [multivariate_normal(mean,std) for mean,std in zip(saga_like_means,saga_like_stds)]


### DO INFERENCE OF BNS DTD AND COLLAPSAR CONTRIBUTION


# load abundance predictions, map likelihoods to delay time distribution parameters and calculate likelihood for abundance predictions

alphas, tmins, xcolls, rates, mejs, Rbnss, Rcolls, ts = [], [], [], [], [], [], [], []
log_like = []

k = 0

if PARTS is None:
    
    INPUTPATH = EUFEPATH

    inputdat = h5py.File(INPUTPATH, 'r')
    pop_dat_i = inputdat['pop']
    yield_dat_i = inputdat['yield']
    
    for j in tqdm(range(len(pop_dat_i))):
    
        pop_dat = pop_dat_i[str(j)]
        yield_dat = yield_dat_i[str(j)]
        
        alphas += [pop_dat['alpha']]
        tmins += [pop_dat['tmin']]
        xcolls += [pop_dat['X0']]
        rates += [pop_dat['rate']]
        mejs += [pop_dat['mej']]
        
        curve = np.column_stack((yield_dat['Fe_H'],yield_dat['Eu_Fe']))
        curve = curve[np.where(curve[:,1] >= -5.)[0][0]:]
        curve = curve[~np.isnan(curve[:,1])]
        log_likes = [np.log(np.trapz(like_func.pdf(curve),curve[:,0])) for like_func in like_funcs]
        log_like += [np.sum(log_likes)]
        k += 1
        
        if NUM is not None: 
            if k >= NUM: break
    
else:
    
    keys = list(np.arange(PARTS))

    for key in tqdm(keys):
        INPUTPATH = '.'.join(EUFEPATH.split('.')[:-1])+'.part{0}'.format(key)+'.'+EUFEPATH.split('.')[-1]

        inputdat = h5py.File(INPUTPATH, 'r')
        pop_dat_i = inputdat['pop']
        yield_dat_i = inputdat['yield']

        for j in tqdm(range(len(pop_dat_i))):

            pop_dat = pop_dat_i[str(j)]
            yield_dat = yield_dat_i[str(j)]

            alphas += [pop_dat['alpha']]
            tmins += [pop_dat['tmin']]
            xcolls += [pop_dat['X0']]
            rates += [pop_dat['rate']]
            mejs += [pop_dat['mej']]

            curve = np.column_stack((yield_dat['Fe_H'],yield_dat['Eu_Fe']))
            curve = curve[np.where(curve[:,1] >= -5.)[0][0]:]
            curve = curve[~np.isnan(curve[:,1])]
            log_likes = [np.log(np.trapz(like_func.pdf(curve),curve[:,0])) for like_func in like_funcs]
            log_like += [np.sum(log_likes)]
            k += 1
            
            if NUM is not None: 
                if k >= NUM: break

npops = k

if NUM is None: NUM = npops
    

### SAVE RESULTS


# save delay time distribution parameter likelihoods

if TAG is None: OUTPATH = OUTDIR+'/'+(OBSPATH.split('/')[-1]).split('.')[0]+'_{0}.csv'.format(NUM)
else: OUTPATH = OUTDIR+'/'+(OBSPATH.split('/')[-1]).split('.')[0]+'_{0}.{1}.csv'.format(NUM,TAG)

outdat = np.column_stack((alphas,tmins,xcolls,mejs,rates,log_like))
np.savetxt(OUTPATH,outdat,delimiter=',',comments='',header='alpha,tmin,xcoll,mej,rate,log_likelihood')
