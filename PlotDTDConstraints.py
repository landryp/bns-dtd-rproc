#!/usr/bin/env python3
# coding: utf-8


'PLOTDTD.PY -- generate plots for delay time distribution inference'
__usage__ = 'PlotDTD.py > post.txt'
__author__ = 'Philippe Landry (pgjlandry@gmail.com)'
__date__ = '11-2023'


### PRELIMINARIES


# load packages

import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import multivariate_normal, gaussian_kde
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

#plt.rcParams["text.usetex"] = True
#plt.rcParams["font.family"] = "serif"

#sns.set_palette('Set1',color_codes=True)
color = 'darkorange'
compare_color = 'gold'


# user input

DATADIR = '/home/philippe.landry/bns-dtd-rproc/etc/'
EUPATH = 'SAGA_MP.csv'
EUPATH_DISK = 'Battistini16_disk.csv'

LIKEDIR = '/home/philippe.landry/bns-dtd-rproc/dat/SAGA_bns/'
LIKEDIR_DISK = '/home/philippe.landry/bns-dtd-rproc/dat/Battistini_bns/'
LIKEPATH = 'SAGA_MP_5M.csv'
LIKEPATH_DISK = 'Battistini16_disk_5M.csv'

POPDIR = '/home/philippe.landry/bns-dtd-rproc/dat/'
POPPATH = 'EuFe_bns-10.h5'

COMPAREDIR = '/home/philippe.landry/bns-dtd-rproc/etc/'
COMPAREPATH = 'label_samples.dat'

PARTS = 1000
NPOP = 10
NMARG = 500

MAXNUM = PARTS*NPOP*NMARG

### LOADING INPUTS


# load SAGA data

FeHs, EuFes, FeH_errs, EuFe_errs = np.loadtxt(DATADIR+EUPATH, unpack=True, delimiter=',', skiprows=1)

#FE_CUT = -1. # cut at [Fe/H] > -1., disk stars only
FE_CUT = -1e16 # no cut, halo+disk

FeHs_in, EuFes_in, FeH_errs_in, EuFe_errs_in = [], [], [], [] # replace data with no errors
for FeH, EuFe, FeH_err, EuFe_err in zip(FeHs, EuFes, FeH_errs, EuFe_errs):
    
    if FeH > FE_CUT:
    
        FeHs_in += [FeH]
        EuFes_in += [EuFe]
    
        if FeH_err > 0. and EuFe_err > 0.:
            FeH_errs_in += [FeH_err]
            EuFe_errs_in += [EuFe_err]

        else:
            FeH_errs_in += [0.] #[avg_FeH_err]
            EuFe_errs_in += [0.] #[avg_EuFe_err]
        
FeHs, EuFes, FeH_errs, EuFe_errs = np.array(FeHs_in), np.array(EuFes_in), np.array(FeH_errs_in), np.array(EuFe_errs_in)


# load disk data

FeHs_disk, EuFes_disk, FeH_errs_disk, EuFe_errs_disk = np.loadtxt(DATADIR+EUPATH_DISK, unpack=True, delimiter=',', skiprows=1)

#FE_CUT = -1. # cut at [Fe/H] > -1., disk stars only
FE_CUT = -1e16 # no cut, halo+disk

FeHs_in, EuFes_in, FeH_errs_in, EuFe_errs_in = [], [], [], [] # replace data with no errors
for FeH, EuFe, FeH_err, EuFe_err in zip(FeHs_disk, EuFes_disk, FeH_errs_disk, EuFe_errs_disk):
    
    if FeH > FE_CUT:
    
        FeHs_in += [FeH]
        EuFes_in += [EuFe]
    
        if FeH_err > 0. and EuFe_err > 0.:
            FeH_errs_in += [FeH_err]
            EuFe_errs_in += [EuFe_err]

        else:
            FeH_errs_in += [0.] #[avg_FeH_err]
            EuFe_errs_in += [0.] #[avg_EuFe_err]
        
FeHs_disk, EuFes_disk, FeH_errs_disk, EuFe_errs_disk = np.array(FeHs_in), np.array(EuFes_in), np.array(FeH_errs_in), np.array(EuFe_errs_in)


# load comparison data

alphas_compare, tmins_compare, tmaxs_compare = np.loadtxt(COMPAREDIR+COMPAREPATH, unpack=True, delimiter=' ', skiprows=1)
tmins_compare = np.array(tmins_compare)/1e6 # convert to Myr


# load population realizations and likelihoods

alphas, tmins, xcolls, mejs, rates, loglikes = np.loadtxt(LIKEDIR+LIKEPATH, unpack=True, delimiter=',', skiprows=1, max_rows=MAXNUM)
ybnss = np.array(rates)*np.array(mejs)

loglikes = loglikes - np.max(loglikes)

likes = np.exp(loglikes)
max_idx = np.argmax(loglikes)

npops = len(alphas)
neff = np.sum(np.array(likes))**2/np.sum(np.array(likes)**2)
print(LIKEDIR+LIKEPATH)
print('number of samples: {0}\n'.format(npops),'number of effective samples: {0}'.format(neff))

assert len(alphas) == len(loglikes) # explicitly make sure weights and population realizations match up

# load population realizations and likelihoods (disk stars only)

alphas_disk, tmins_disk, xcolls_disk, mejs_disk, rates_disk, loglikes_disk = np.loadtxt(LIKEDIR_DISK+LIKEPATH_DISK, unpack=True, delimiter=',', skiprows=1, max_rows=MAXNUM)
ybnss_disk = np.array(rates_disk)*np.array(mejs_disk)

loglikes_disk = loglikes_disk - np.max(loglikes_disk)

likes_disk = np.exp(loglikes_disk)
max_idx_disk = np.argmax(loglikes_disk)

npops_disk = len(alphas_disk)
neff_disk = np.sum(np.array(likes_disk))**2/np.sum(np.array(likes_disk)**2)
print('\n')
print(LIKEDIR_DISK+LIKEPATH_DISK)
print('number of samples: {0}\n'.format(npops_disk),'number of effective samples: {0}'.format(neff_disk))


# load abundance predictions

keys = [str(key) for key in range(PARTS)]
eu_pts_list, rates, mejs = [], [], []

fe_grid = np.arange(-3.,0.5,0.05)

k = 0

for key in tqdm(keys):
    INPUTPATH = POPDIR+'.'.join(POPPATH.split('.')[:-1])+'.part{0}'.format(key)+'.'+POPPATH.split('.')[-1]

    inputdat = h5py.File(INPUTPATH, 'r')
    yield_dat_i = inputdat['yield']
    pop_dat_i = inputdat['pop']
    
    for j in range(len(yield_dat_i)):
    
        yield_dat = yield_dat_i[str(j)]
        pop_dat = pop_dat_i[str(j)]
        func = interp1d(yield_dat['Fe_H'],yield_dat['Eu_Fe'],bounds_error=False)
        eu_pts = func(fe_grid)
        eu_pts_list += [eu_pts]
        
        rates += [pop_dat['rate']]
        mejs += [pop_dat['m_ej']]
        
        k += 1
        
        
### MAKE ABUNDANCE PREDICTION PLOT
        
        
# calculate abundance history confidence envelopes

num_funcs = 10000
eu_pts_idxs = np.random.choice(range(len(eu_pts_list)),num_funcs,True,likes/np.sum(likes))
eu_pts_samps = np.array([eu_pts_list[idx] for idx in eu_pts_idxs]).T

eu_pts_disk_idxs = np.random.choice(range(len(eu_pts_list)),num_funcs,True,likes_disk/np.sum(likes_disk))
eu_pts_disk_samps = np.array([eu_pts_list[idx] for idx in eu_pts_disk_idxs]).T

mds = [np.nanmedian(eu_pts_samps[i]) for i in range(len(fe_grid))]
lbs = [np.nanquantile(eu_pts_samps[i],0.05) for i in range(len(fe_grid))]
ubs = [np.nanquantile(eu_pts_samps[i],0.95) for i in range(len(fe_grid))]
lbs_std = [np.nanquantile(eu_pts_samps[i],0.16) for i in range(len(fe_grid))]
ubs_std = [np.nanquantile(eu_pts_samps[i],0.84) for i in range(len(fe_grid))]

mds_disk = [np.nanmedian(eu_pts_disk_samps[i]) for i in range(len(fe_grid))]
lbs_disk = [np.nanquantile(eu_pts_disk_samps[i],0.05) for i in range(len(fe_grid))]
ubs_disk = [np.nanquantile(eu_pts_disk_samps[i],0.95) for i in range(len(fe_grid))]
lbs_disk_std = [np.nanquantile(eu_pts_disk_samps[i],0.16) for i in range(len(fe_grid))]
ubs_disk_std = [np.nanquantile(eu_pts_disk_samps[i],0.84) for i in range(len(fe_grid))]
        
        
# plot inferred abundance history on top of observations

plt.figure(figsize=(6.4,4.8))

plt.fill_between(fe_grid,lbs_disk,ubs_disk,facecolor=color,edgecolor=None,alpha=0.25, label='BNS',zorder=10) # 90% CI
plt.fill_between(fe_grid,lbs_disk_std,ubs_disk_std,facecolor=color,edgecolor=None,alpha=0.5,zorder=10) # 68% CI

plt.plot(fe_grid,mds_disk,c=color,zorder=10) # median

plt.errorbar(FeHs, EuFes, xerr=[FeH_errs,FeH_errs], yerr=[EuFe_errs,EuFe_errs], c='g', fmt=',', lw=1, label='SAGA')
#plt.scatter(FeHs, EuFes,c=sns.color_palette()[1],marker='.',s=2,zorder=1)
plt.scatter(FeHs_disk, EuFes_disk,marker='D',facecolor='dodgerblue',edgecolor='navy', s=16, lw=0.5, label='Battistini & Bensby (2016)')

#plt.plot(fe_grid,mds,c=sns.color_palette()[4],ls='--',lw=1,alpha=1,zorder=3)
#plt.plot(fe_grid,ubs,c=sns.color_palette()[4],ls='--',lw=1,alpha=1,zorder=3)
#plt.plot(fe_grid,lbs,c=sns.color_palette()[4],ls='--',lw=1,alpha=1,zorder=3)

plt.xlim(-3.,0.5)
plt.ylim(-1.,1.5)
plt.xlabel('[Fe/H]')#,fontsize=16)
plt.ylabel('[Eu/Fe]')#,fontsize=16)
plt.legend(frameon=True,loc='upper right')
plt.savefig('EuFe_bns.pdf')


### RECORD DTD PARAMETER CONSTRAINTS


# save confidence intervals

print('\n')
print('disk stars\n')
print('maxL population realization')
print('alpha: {0}\n'.format(alphas_disk[max_idx_disk]),'tmin: {0}\n'.format(tmins_disk[max_idx_disk]),'Xcoll: {0}\n'.format(xcolls_disk[max_idx_disk]),'Ybns: {0}\n'.format(ybnss_disk[max_idx_disk]),'log(maxL): {0}\n'.format(np.max(loglikes_disk)-np.mean(loglikes_disk)),'neff: {0}\n'.format(neff_disk))

print('marginal alpha posterior')
print('mean: {0}\n'.format(np.mean(alphas_disk[eu_pts_disk_idxs])),'median: {0}\n'.format(np.median(alphas_disk[eu_pts_disk_idxs])),'90lb: {0}\n'.format(np.quantile(alphas_disk[eu_pts_disk_idxs],0.05)),'90ub: {0}\n'.format(np.quantile(alphas_disk[eu_pts_disk_idxs],0.95)),'68lb: {0}\n'.format(np.quantile(alphas_disk[eu_pts_disk_idxs],0.16)),'68ub: {0}\n'.format(np.quantile(alphas_disk[eu_pts_disk_idxs],0.84)),'90os: {0}\n'.format(np.quantile(alphas_disk[eu_pts_disk_idxs],0.90)),'68os: {0}\n'.format(np.quantile(alphas_disk[eu_pts_disk_idxs],0.68)))

print('marginal tmin posterior')
print('mean: {0}\n'.format(np.mean(tmins_disk[eu_pts_disk_idxs])),'median: {0}\n'.format(np.median(tmins_disk[eu_pts_disk_idxs])),'90lb: {0}\n'.format(np.quantile(tmins_disk[eu_pts_disk_idxs],0.05)),'90ub: {0}\n'.format(np.quantile(tmins_disk[eu_pts_disk_idxs],0.95)),'68lb: {0}\n'.format(np.quantile(tmins_disk[eu_pts_disk_idxs],0.16)),'68ub: {0}\n'.format(np.quantile(tmins_disk[eu_pts_disk_idxs],0.84)),'90os: {0}\n'.format(np.quantile(tmins_disk[eu_pts_disk_idxs],0.90)),'68os: {0}\n'.format(np.quantile(tmins_disk[eu_pts_disk_idxs],0.68)))

print('marginal xcoll posterior')
print('mean: {0}\n'.format(np.mean(xcolls_disk[eu_pts_disk_idxs])),'median: {0}\n'.format(np.median(xcolls_disk[eu_pts_disk_idxs])),'90lb: {0}\n'.format(np.quantile(xcolls_disk[eu_pts_disk_idxs],0.05)),'90ub: {0}\n'.format(np.quantile(xcolls_disk[eu_pts_disk_idxs],0.95)),'68lb: {0}\n'.format(np.quantile(xcolls_disk[eu_pts_disk_idxs],0.16)),'68ub: {0}\n'.format(np.quantile(xcolls_disk[eu_pts_disk_idxs],0.84)),'90os: {0}\n'.format(np.quantile(xcolls_disk[eu_pts_disk_idxs],0.10)),'68os: {0}\n'.format(np.quantile(xcolls_disk[eu_pts_disk_idxs],0.32)),'10os: {0}\n'.format(np.quantile(xcolls_disk[eu_pts_disk_idxs],0.90)))

print('marginal ybns posterior')
print('mean: {0}\n'.format(np.mean(ybnss_disk[eu_pts_disk_idxs])),'median: {0}\n'.format(np.median(ybnss_disk[eu_pts_disk_idxs])),'90lb: {0}\n'.format(np.quantile(ybnss_disk[eu_pts_disk_idxs],0.05)),'90ub: {0}\n'.format(np.quantile(ybnss_disk[eu_pts_disk_idxs],0.95)),'68lb: {0}\n'.format(np.quantile(ybnss_disk[eu_pts_disk_idxs],0.16)),'68ub: {0}\n'.format(np.quantile(ybnss_disk[eu_pts_disk_idxs],0.84)),'90os: {0}\n'.format(np.quantile(ybnss_disk[eu_pts_disk_idxs],0.10)),'68os: {0}\n'.format(np.quantile(ybnss_disk[eu_pts_disk_idxs],0.32)),'10os: {0}\n'.format(np.quantile(ybnss_disk[eu_pts_disk_idxs],0.90)))

print('\n')
print('disk+halo stars\n')
print('maxL population realization')
print('alpha: {0}\n'.format(alphas[max_idx]),'tmin: {0}\n'.format(tmins[max_idx]),'Xcoll: {0}\n'.format(xcolls[max_idx]),'Ybns: {0}\n'.format(ybnss[max_idx]),'log(maxL): {0}\n'.format(np.max(loglikes)-np.mean(loglikes)),'neff: {0}\n'.format(neff))

print('marginal alpha posterior')
print('mean: {0}\n'.format(np.mean(alphas[eu_pts_idxs])),'median: {0}\n'.format(np.median(alphas[eu_pts_idxs])),'90lb: {0}\n'.format(np.quantile(alphas[eu_pts_idxs],0.05)),'90ub: {0}\n'.format(np.quantile(alphas[eu_pts_idxs],0.95)),'68lb: {0}\n'.format(np.quantile(alphas[eu_pts_idxs],0.16)),'68ub: {0}\n'.format(np.quantile(alphas[eu_pts_idxs],0.84)),'90os: {0}\n'.format(np.quantile(alphas[eu_pts_idxs],0.90)),'68os: {0}\n'.format(np.quantile(alphas[eu_pts_idxs],0.68)))

print('marginal tmin posterior')
print('mean: {0}\n'.format(np.mean(tmins[eu_pts_idxs])),'median: {0}\n'.format(np.median(tmins[eu_pts_idxs])),'90lb: {0}\n'.format(np.quantile(tmins[eu_pts_idxs],0.05)),'90ub: {0}\n'.format(np.quantile(tmins[eu_pts_idxs],0.95)),'68lb: {0}\n'.format(np.quantile(tmins[eu_pts_idxs],0.16)),'68ub: {0}\n'.format(np.quantile(tmins[eu_pts_idxs],0.84)),'90os: {0}\n'.format(np.quantile(tmins[eu_pts_idxs],0.90)),'68os: {0}\n'.format(np.quantile(tmins[eu_pts_idxs],0.68)))

print('marginal xcoll posterior')
print('mean: {0}\n'.format(np.mean(xcolls[eu_pts_idxs])),'median: {0}\n'.format(np.median(xcolls[eu_pts_idxs])),'90lb: {0}\n'.format(np.quantile(xcolls[eu_pts_idxs],0.05)),'90ub: {0}\n'.format(np.quantile(xcolls[eu_pts_idxs],0.95)),'68lb: {0}\n'.format(np.quantile(xcolls[eu_pts_idxs],0.16)),'68ub: {0}\n'.format(np.quantile(xcolls[eu_pts_idxs],0.84)),'90os: {0}\n'.format(np.quantile(xcolls[eu_pts_idxs],0.10)),'68os: {0}\n'.format(np.quantile(xcolls[eu_pts_idxs],0.32)))

print('marginal ybns posterior')
print('mean: {0}\n'.format(np.mean(ybnss[eu_pts_idxs])),'median: {0}\n'.format(np.median(ybnss[eu_pts_idxs])),'90lb: {0}\n'.format(np.quantile(ybnss[eu_pts_idxs],0.05)),'90ub: {0}\n'.format(np.quantile(ybnss[eu_pts_idxs],0.95)),'68lb: {0}\n'.format(np.quantile(ybnss[eu_pts_idxs],0.16)),'68ub: {0}\n'.format(np.quantile(ybnss[eu_pts_idxs],0.84)),'90os: {0}\n'.format(np.quantile(ybnss[eu_pts_idxs],0.10)),'68os: {0}\n'.format(np.quantile(ybnss[eu_pts_idxs],0.32)))


### MAKE INFERRED DTD PARAMETER PLOT


# plot BNS DTD parameter posteriors

wts = likes_disk
wts_halo = likes

tmins = np.array(tmins)*1e3 # convert to Myr
tmins_disk = np.array(tmins_disk)*1e3 # convert to Myr

alphas_halo = alphas[eu_pts_idxs]
tmins_halo = tmins[eu_pts_idxs]
xcolls_halo = xcolls[eu_pts_idxs]
ybnss_halo = ybnss[eu_pts_idxs]

alphas_disk = alphas_disk[eu_pts_disk_idxs]
tmins_disk = tmins_disk[eu_pts_disk_idxs]
xcolls_disk = xcolls_disk[eu_pts_disk_idxs]
ybnss_disk = ybnss_disk[eu_pts_disk_idxs]

prior_idxs = np.random.choice(range(len(alphas)),10000,True)

# plot BNS DTD parameter posteriors

fig=plt.figure(figsize=(6.4,2.4))

axx=plt.subplot(1, 3, 1)

tmins_disk_reflect, alphas_disk_reflect, tmins_halo_reflect, alphas_halo_reflect = [], [], [], []

for alpha,tmin,alpha_halo,tmin_halo in zip(alphas_disk,tmins_disk,alphas_halo,tmins_halo):
 
    if tmin < 0.015*1e3:
        tmins_disk_reflect += [2*(0.01*1e3)-tmin]
        alphas_disk_reflect += [alpha]
    '''      
    elif tmin > 0.9:
        tmins_disk_reflect += [2*(2.01)-tmin]
        alphas_disk_reflect += [alpha]
    '''
    if alpha < -2.9:
        tmins_disk_reflect += [tmin]
        alphas_disk_reflect += [2*(-3.)-alpha]
    '''            
    elif alpha > -0.4:
        tmins_disk_reflect += [tmin]
        alphas_disk_reflect += [2*(-0.5)-alpha]
    '''    
    if tmin_halo < 0.015*1e3:
        tmins_halo_reflect += [2*(0.01*1e3)-tmin_halo]
        alphas_halo_reflect += [alpha_halo]
    '''     
    elif tmin_halo > 0.9:
        tmins_halo_reflect += [2*(2.01)-tmin_halo]
        alphas_halo_reflect += [alpha_halo]
    '''
    if alpha_halo < -2.9:
        tmins_halo_reflect += [tmin_halo]
        alphas_halo_reflect += [2*(-3.)-alpha_halo]
    '''    
    elif alpha_halo > -0.4:
        tmins_halo_reflect += [tmin_halo]
        alphas_halo_reflect += [2*(-0.5)-alpha_halo]
    '''       
sns.kdeplot(x=alphas_compare,y=np.log10(tmins_compare),levels=[0.1,0.32],c=compare_color,axes=axx)
#axx.scatter(alphas[::NMARG],np.log10(tmins[::NMARG]),marker='.',s=0.6,c='k',alpha=0.1)
sns.kdeplot(x=list(alphas_disk)+alphas_disk_reflect,y=np.log10(list(tmins_disk)+tmins_disk_reflect),levels=[0.1,0.32],c=color,axes=axx,zorder=10)
sns.kdeplot(x=list(alphas_disk)+alphas_disk_reflect,y=np.log10(list(tmins_disk)+tmins_disk_reflect),levels=[0.32,1.],color=color,alpha=0.25,fill=True,axes=axx,zorder=10)
#sns.kdeplot(x=list(alphas_halo)+alphas_halo_reflect,y=np.log10(list(tmins_halo)+tmins_halo_reflect),levels=[0.1],c=sns.color_palette()[4],cut=3,linewidths=1,linestyles='--',axes=axx)

plt.plot([-10.,-10.],[-10.,-5.],color=compare_color,alpha=1,label='sGRB')
plt.plot([-10.,-10.],[-10.,-5.],color=color,alpha=1,label='BNS')

axx.set_xlim(-3.,-1.)
axx.set_ylim(1.,3.)
axx.set_xlabel(r'$\alpha$')#,fontsize=16)
axx.set_ylabel(r'$\log_{10}\,t_\mathrm{min}/\mathrm{Myr}$')#,fontsize=16)
plt.legend(frameon=True,loc='upper right',fontsize=6)

ax=plt.subplot(1, 3, 2)

alphas_reflect, alphas_disk_reflect, alphas_halo_reflect = [], [], []

for alpha,alpha_disk,alpha_halo in zip(alphas,alphas_disk,alphas_halo):

    if alpha < -2.9:
        alphas_reflect += [2*(-3.)-alpha]
    '''
    elif alpha > -0.4:
        alphas_reflect += [2*(-0.5)-alpha]
    '''    
    if alpha_disk < -2.9:
        alphas_disk_reflect += [2*(-3.)-alpha_disk]
    '''    
    elif alpha_disk > -0.4:
        alphas_disk_reflect += [2*(-0.5)-alpha_disk]
    '''    
    if alpha_halo < -2.9:
        alphas_halo_reflect += [2*(-3.)-alpha_halo]
    '''    
    elif alpha_halo > -0.4:
        alphas_halo_reflect += [2*(-0.5)-alpha_halo]
    '''
sns.kdeplot(x=alphas_compare,c=compare_color,bw_adjust=1.,cut=20,axes=ax)
#ax.hist(list(alphas[::NMARG])+alphas_reflect[::NMARG],density=True,bins=np.arange(-3.,-0.9,0.1),color='k',alpha=0.1)
ax.hist(list(alphas_disk)+alphas_disk_reflect,density=True,bins=np.arange(-3.,-0.9,0.1),color=color,alpha=0.25,zorder=10)

#sns.kdeplot(x=list(alphas)+alphas_reflect,color='k',axes=ax,alpha=0.7,bw_adjust=1.2)
sns.kdeplot(x=list(alphas_disk)+alphas_disk_reflect,c=color,axes=ax,cut=20,bw_adjust=1.,zorder=10)
#sns.kdeplot(x=list(alphas_halo)+alphas_halo_reflect,c=sns.color_palette()[4],lw=1,ls='--',axes=ax,bw_adjust=1.2)

plt.plot([-10.,-10.],[-10.,-5.],color='k',alpha=1,label='prior')
plt.plot([-10.,-10.],[-10.,-5.],color=color,alpha=1,label='posterior')

ax.set_xlim(-3.,-1.)
ax.set_ylim(0.,4.)
ax.set_xlabel(r'$\alpha$')#,fontsize=16)
ax.set_ylabel('Probability density')#,fontsize=16)
ax.set_yticks([0.,1.,2.,3.,4.])
ax.set_yticklabels([])

ax=plt.subplot(1, 3, 3)

tmins_reflect, tmins_disk_reflect, tmins_halo_reflect = [], [], []

for tmin, tmin_disk, tmin_halo in zip(tmins,tmins_disk,tmins_halo): # reflect across hard prior bounds
    
    if tmin < 0.015*1e3:
        tmins_reflect += [2*(0.01*1e3)-tmin]
    '''      
    elif tmin > 0.9:
        tmins_reflect += [2*(2.01)-tmin]
    '''
    if tmin_disk < 0.015*1e3:
        tmins_disk_reflect += [2*(0.01*1e3)-tmin_disk]
    '''   
    elif tmin_disk > 0.9:
        tmins_disk_reflect += [2*(2.01)-tmin]
    '''
    if tmin_halo < 0.015*1e3:
        tmins_halo_reflect += [2*(0.01*1e3)-tmin_halo]
    '''      
    elif tmin_halo > 0.9:
        tmins_halo_reflect += [2*(2.01)-tmin]
    '''    
sns.kdeplot(x=np.log10(tmins_compare),c=compare_color,bw_adjust=1.,cut=20,axes=ax)
#ax.hist(list(np.log10(tmins[::NMARG]))+list(np.log10(tmins_reflect[::NMARG])),density=True,bins=np.arange(-2.,0.1,0.1),color='k',alpha=0.1)
ax.hist(list(np.log10(tmins_disk))+list(np.log10(tmins_disk_reflect)),density=True,bins=np.arange(1.,3.1,0.1),color=color,alpha=0.25,zorder=10)

#sns.kdeplot(x=np.log10(list(tmins)+tmins_reflect),color='k',axes=ax,alpha=0.7,bw_adjust=1.2)
sns.kdeplot(x=np.log10(list(tmins_disk)+tmins_disk_reflect),c=color,axes=ax,cut=20,bw_adjust=1.,zorder=10)
#sns.kdeplot(x=np.log10(list(tmins_halo)+tmins_halo_reflect),c=sns.color_palette()[4],lw=1,ls='--',axes=ax,bw_adjust=1.2)

ax.set_xlim(1.,3.)
ax.set_ylim(0.,4.)
ax.set_xlabel(r'$\log_{10}\,t_\mathrm{min}/\mathrm{Myr}$')#,fontsize=16)
ax.set_yticks([0.,1.,2.,3.,4.])
ax.set_ylabel('')
ax.set_yticklabels([])

fig.subplots_adjust(wspace=0.5)
#plt.tight_layout()
plt.subplots_adjust(bottom=0.2)
plt.savefig('dtd_bns.pdf')


### MAKE INFERRED YBNS PLOT


# plot rate*mej parameter posteriors

ybnss = np.array(rates)*np.array(mejs)

fig=plt.figure(figsize=(6.4,2.4))

ax=plt.subplot(1, 3, 1)

ybnss_disk_reflect, ybnss_halo_reflect, ybnss_reflect = [], [], []

for ybns_disk,ybns_halo,ybns in zip(ybnss_disk,ybnss_halo,ybnss): # reflect across hard prior bounds
 
    if ybns_disk < 20.:
        ybnss_disk_reflect += [2.*(0.)-ybns_disk]
    '''            
    elif ybns_disk > 180.:
        ybnss_disk_reflect += [2.*(1.)-ybns_disk]
    '''
    
    if ybns_halo < 20.:
        ybnss_halo_reflect += [2.*(0.)-ybns_halo]
    '''            
    elif ybns_halo > 180.:
        ybnss_halo_reflect += [2.*(1.)-ybns_halo]
    '''
    
    if ybns < 20.:
        ybnss_reflect += [2.*(0.)-ybns]
    '''            
    elif ybns > 180.:
        ybnss_reflect += [2.*(1.)-ybns]
    '''

ax.hist(list(ybnss[:500])+ybnss_reflect[:500],density=True,bins=np.arange(0.,31.,1.),color='k',alpha=0.1)
ax.hist(list(ybnss_disk)+ybnss_disk_reflect,density=True,bins=np.arange(0.,31.,1.),color=color,alpha=0.25)

sns.kdeplot(x=list(ybnss)+ybnss_reflect,color='k',axes=ax,alpha=0.7,cut=20)
sns.kdeplot(x=list(ybnss_disk)+ybnss_disk_reflect,c=color,axes=ax,cut=20)
#sns.kdeplot(x=list(ybnss_halo)+ybnss_halo_reflect,c=color,lw=1,ls='--',axes=ax,bw_adjust=1.2)

plt.plot([-10.,-10.],[-10.,-5.],color='k',alpha=1,label='prior')
plt.plot([-10.,-10.],[-10.,-5.],color=color,alpha=1,label='BNS')

ax.set_xlim(0.,30.)
ax.set_ylim(0.,0.3)
ax.set_xlabel(r'$m_\mathrm{ej} R_\mathrm{MW}\;[M_\odot/\mathrm{Myr}]$')#,fontsize=16)
ax.set_ylabel('Probability density')#,fontsize=16)
ax.set_yticks([0.,0.075,0.15,0.225,0.3])
ax.set_yticklabels([])
plt.legend(frameon=True,loc='upper right',fontsize=6)

axx=plt.subplot(1, 3, 2)

alphas_disk_reflect, alphas_halo_reflect, ybnss_disk_reflect, ybnss_halo_reflect = [], [], [], []

for alpha,alpha_halo,ybns,ybns_halo in zip(alphas_disk,alphas_halo,ybnss_disk,ybnss_halo): # reflect across hard prior bounds
 
    if ybns < 10.:
        alphas_disk_reflect += [alpha]
        ybnss_disk_reflect += [2.*(0.)-ybns]
    '''            
    elif ybns > 190.:
        alphas_disk_reflect += [alpha]
        ybnss_disk_reflect += [2.*(1.)-ybns]
    '''
    if alpha < -2.9:
        alphas_disk_reflect += [2*(-3.)-alpha]
        ybnss_disk_reflect += [ybns]
    '''    
    elif alpha > -0.4:
        alphas_disk_reflect += [2*(-0.5)-alpha]
        ybnss_disk_reflect += [ybns]
    '''
    if ybns_halo < 10.:
        alphas_halo_reflect += [alpha_halo]
        ybnss_halo_reflect += [2.*(0.)-ybns_halo]
    '''            
    elif ybns_halo > 190.:
        alphas_halo_reflect += [alpha_halo]
        ybnss_halo_reflect += [2.*(1.)-ybns_halo]
    '''
    if alpha_halo < -2.9:
        alphas_halo_reflect += [2*(-3.)-alpha_halo]
        ybnss_halo_reflect += [ybns_halo]
    ''' 
    elif alpha_halo > -0.4:
        alphas_halo_reflect += [2*(-0.5)-alpha_halo]
        ybnss_halo_reflect += [ybns_halo]
    '''
axx.scatter(alphas[prior_idxs],ybnss[prior_idxs],marker='.',s=0.6,c='k',alpha=0.05)
sns.kdeplot(x=list(alphas_disk)+alphas_disk_reflect,y=list(ybnss_disk)+ybnss_disk_reflect,levels=[0.1,0.32],c=color,cut=3,axes=axx)
#sns.kdeplot(x=list(alphas_halo)+alphas_halo_reflect,y=list(ybnss_halo)+ybnss_halo_reflect,levels=[0.1],c=color,cut=3,linewidths=1,linestyles='--',axes=axx)

axx.set_xlim(-3.,-1.)
axx.set_ylim(0.,30.)
axx.set_xlabel(r'$\alpha$')#,fontsize=16)
axx.set_ylabel(r'$m_\mathrm{ej} R_\mathrm{MW}\;[M_\odot/\mathrm{Myr}]$')#,fontsize=16)

axx=plt.subplot(1, 3, 3)

tmins_disk_reflect, tmins_halo_reflect, ybnss_disk_reflect, ybnss_halo_reflect = [], [], [], []

for tmin,tmin_halo,ybns,ybns_halo in zip(tmins_disk,tmins_halo,ybnss_disk,ybnss_halo): # reflect across hard prior bounds
 
    if ybns < 10.:
        tmins_disk_reflect += [tmin]
        ybnss_disk_reflect += [2.*(0.)-ybns]
    '''            
    elif ybns > 190.:
        tmins_disk_reflect += [tmin]
        ybnss_disk_reflect += [2.*(1.)-ybns]
    '''
    if tmin < 0.015*1e3:
        tmins_disk_reflect += [2*(0.01*1e3)-tmin]
        ybnss_disk_reflect += [ybns]
    '''
    elif tmin > 0.9:
        tmins_disk_reflect += [2*(2.01)-tmin]
        ybnss_disk_reflect += [ybns]
    '''
    
    if ybns_halo < 10.:
        tmins_halo_reflect += [tmin_halo]
        ybnss_halo_reflect += [2.*(0.)-ybns_halo]
    '''            
    elif ybns_halo > 190.:
        tmins_halo_reflect += [tmin_halo]
        ybnss_halo_reflect += [2.*(1.)-ybns_halo]
    '''
    if tmin_halo < 0.015*1e3:
        tmins_halo_reflect += [2*(0.01*1e3)-tmin]
        ybnss_halo_reflect += [ybns_halo]
    ''' 
    elif tmin_halo > 0.9:
        tmins_halo_reflect += [2*(2.01)-tmin]
        ybnss_halo_reflect += [ybns_halo]
    '''

axx.scatter(np.log10(tmins[prior_idxs]),ybnss[prior_idxs],marker='.',s=0.6,c='k',alpha=0.05)
sns.kdeplot(x=np.log10(list(tmins_disk)+tmins_disk_reflect),y=list(ybnss_disk)+ybnss_disk_reflect,levels=[0.1,0.32],c=color,cut=3,axes=axx)
#sns.kdeplot(x=np.log10(list(tmins_halo)+tmins_halo_reflect),y=list(ybnss_halo)+ybnss_halo_reflect,levels=[0.1],c=color,cut=3,linewidths=1,linestyles='--',axes=axx)

axx.set_xlim(1.,3.)
axx.set_ylim(0.,30.)
axx.set_xlabel(r'$\log_{10}\,t_\mathrm{min}/\mathrm{Myr}$')#,fontsize=16)
axx.set_ylabel('')
axx.set_yticklabels([])

fig.subplots_adjust(wspace=0.5)
#plt.tight_layout()
plt.subplots_adjust(bottom=0.2)
plt.savefig('ybns_bns.pdf')
