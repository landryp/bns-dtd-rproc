#!/usr/bin/env python3
# coding: utf-8


'PLOTSECONDCHANNELEVOLUTION.PY -- plot second-channel contribution fraction over Galactic history'
__usage__ = 'PlotSecondChannelEvolution.py'
__author__ = 'Philippe Landry (pgjlandry@gmail.com)'
__date__ = '11-2023'


### PRELIMINARIES


# load packages

import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
from tqdm import tqdm

from etc.rProcessChemicalEvolution import Xsun_Eu_r69 # bug fix!


# user input

LIKEDIR = '/home/philippe.landry/bns-dtd-rproc/dat/Battistini_grbbnscls/'
LIKEPATH = 'Battistini16_disk_5M.csv'

POPDIR = '/home/philippe.landry/bns-dtd-rproc/dat/'
POPPATH = 'EuFe_grbbnscls-100.h5'

PARTS = 100
NPOP = 100
NMARG = 500

MAXNUM = PARTS*NPOP*NMARG

color = 'firebrick'
compare_color = 'k'

tag = 'grbbnscls'

Z_MIN, Z_MAX = (0.,10.)
NZ = int((Z_MAX-Z_MIN)/0.1) # z grid spacing for confidence intervals
z_grid = np.linspace(Z_MIN,Z_MAX,NZ)

### LOADING INPUTS


# load population realizations and likelihoods

alphas, tmins, X0s, mejs, rates, loglikes = np.loadtxt(LIKEDIR+LIKEPATH, unpack=True, delimiter=',', skiprows=1, max_rows=MAXNUM)

Ys = np.array(rates)*np.array(mejs)
loglikes = loglikes - np.max(loglikes)
likes = np.exp(loglikes)
max_idx = np.argmax(loglikes)

marglikes = np.array([np.sum(likes[i*NMARG:(i+1)*NMARG]) for i in range(PARTS*NPOP)])

npops = len(alphas)
neff = np.sum(np.array(likes))**2/np.sum(np.array(likes)**2)
print(LIKEDIR+LIKEPATH)
print('number of samples: {0}\n'.format(npops),'number of effective samples: {0}'.format(neff))


# load abundance predictions

keys = [str(key) for key in range(PARTS)]
Xs, zs = [], []

for key in tqdm(keys):
    INPUTPATH = POPDIR+'.'.join(POPPATH.split('.')[:-1])+'.part{0}'.format(key)+'.'+POPPATH.split('.')[-1]

    inputdat = h5py.File(INPUTPATH, 'r')
    
    yield_dat_i = inputdat['yield']
    frac_dat_i = inputdat['frac']
    
    for j in range(int(len(yield_dat_i)/NMARG)):
    
        frac_dat = frac_dat_i[str(j*NMARG)]
        
        Xts = frac_dat['X']
        zts = frac_dat['z']
        
        Xt_of_z = interp1d(zts,Xts,bounds_error=False)
        Xs += [Xt_of_z(z_grid)]

Xs = np.array(Xs)
prior_Xs = np.array([X[0] for X in Xs])

### MAKE ABUNDANCE PREDICTION PLOT


# calculate abundance history confidence envelopes

CLS = [0.68,0.9]
Xmd, Xqs = [], []

def wtquantile(xs,qs,wts=[]):
    
    nan_idxs = np.isnan(xs)
    xs = np.array(xs[~nan_idxs])
    
    num_xs = len(xs)
    qs = np.array(qs, ndmin=1)
    if len(xs) < 1: return np.full(2*len(qs),np.nan)
    elif len(xs) == 1: return np.full(2*len(qs),xs[0])
    if len(wts) < 1: wts = np.full(num_xs, 1.)
    else: wts = np.array(wts[~nan_idxs])
    
    ps = wts/np.sum(wts)
    xs_sorted,ps_sorted = zip(*sorted(list(zip(xs,ps)),reverse=False))

    Ps = np.cumsum(ps_sorted)

    idxs_lb = np.array([np.where(Ps >= (1.-q)/2.)[0][0] for q in qs])
    idxs_ub = np.array([np.where(Ps >= 1.-(1.-q)/2.)[0][0] for q in qs])
    xs_sorted = np.array(xs_sorted)

    return list(xs_sorted[idxs_lb])+list(xs_sorted[idxs_ub])

prior_kde = gaussian_kde(list(prior_Xs)+list(-prior_Xs[prior_Xs < 0.1])+list(2.-prior_Xs[prior_Xs > 0.9]),bw_method='silverman')
inv_wts = prior_kde(prior_Xs)
wts = np.sum(inv_wts)/inv_wts

for i in range(NZ):
    
    Xqs += [wtquantile(Xs[:,i],[0.68,0.9],wts*marglikes)]
    Xmd += [wtquantile(Xs[:,i],0.,wts*marglikes)]
    
Xqs = np.array(Xqs)


# plot integrated mass contribution fractions

num_func = 1000
idxs = np.random.choice(range(len(Xs)),num_func,True)

plt.figure(figsize=(6.4,4.8))

for X in Xs[idxs]: plt.plot(z_grid,X,c='k',alpha=0.05)
plt.plot(z_grid,Xs[-1],c='k',alpha=0.05,label='prior')

plt.fill_between(z_grid,Xqs[:,1],Xqs[:,3],facecolor='firebrick',edgecolor=None,alpha=0.25, label='BNS+SFH',zorder=10) # 90% CI
plt.fill_between(z_grid,Xqs[:,0],Xqs[:,2],facecolor='firebrick',edgecolor=None,alpha=0.5,zorder=10) # 68% CI

plt.plot(z_grid,Xmd,c='firebrick',zorder=10) # median

plt.gca().invert_xaxis()
plt.xlabel('$z$')
plt.ylabel('$X_\mathrm{SFH}$')
plt.legend()
plt.savefig('plt/xsfr_evolution.pdf')
