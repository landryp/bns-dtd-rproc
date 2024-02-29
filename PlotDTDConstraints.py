#!/usr/bin/env python3
# coding: utf-8


'PLOTDTDCONSTRAINTS.PY -- plot posterior samples in binary neutron star delay time distribution parameters'
__usage__ = 'PlotDTDConstraints.py'
__author__ = 'Philippe Landry (pgjlandry@gmail.com)'
__date__ = '11-2023'


### PRELIMINARIES


# load packages

import numpy as np
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
from tqdm import tqdm


# user input

DATADIR = '/home/philippe.landry/bns-dtd-rproc/etc/'
EUPATH = 'SAGA_MP.csv'
EUPATH_DISK = 'Battistini16_disk.csv'

LIKEDIR_DISK = '/home/philippe.landry/bns-dtd-rproc/dat/Battistini_bns/'
LIKEPATH_DISK = 'Battistini16_disk_5M.csv'

POPDIR = '/home/philippe.landry/bns-dtd-rproc/dat/'
POPPATH = 'EuFe_bns-10.h5'

COMPAREDIR = '/home/philippe.landry/bns-dtd-rproc/etc/'
COMPAREPATH = 'label_samples.dat'

PARTS = 1000
NPOP = 10
NMARG = 500

MAXNUM = PARTS*NPOP*NMARG

color = 'darkorange'
compare_color = 'gold'

tag = 'bns'


### LOADING INPUTS


# load SAGA data

FeHs, EuFes, FeH_errs, EuFe_errs = np.loadtxt(DATADIR+EUPATH, unpack=True, delimiter=',', skiprows=1)


# load disk data

FeHs_disk, EuFes_disk, FeH_errs_disk, EuFe_errs_disk = np.loadtxt(DATADIR+EUPATH_DISK, unpack=True, delimiter=',', skiprows=1)


# load comparison data

alphas_compare, tmins_compare, tmaxs_compare = np.loadtxt(COMPAREDIR+COMPAREPATH, unpack=True, delimiter=' ', skiprows=1)
tmins_compare = np.array(tmins_compare)/1e6 # convert to Myr


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
eu_pts_disk_idxs = np.random.choice(range(len(eu_pts_list)),num_funcs,True,likes_disk/np.sum(likes_disk))
eu_pts_disk_samps = np.array([eu_pts_list[idx] for idx in eu_pts_disk_idxs]).T

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
plt.scatter(FeHs_disk, EuFes_disk,marker='D',facecolor='dodgerblue',edgecolor='navy', s=16, lw=0.5, label='Battistini & Bensby (2016)')


plt.xlim(-3.,0.5)
plt.ylim(-1.,1.5)
plt.xlabel('[Fe/H]')#,fontsize=16)
plt.ylabel('[Eu/Fe]')#,fontsize=16)
plt.legend(frameon=True,loc='upper right')
plt.savefig('EuFe_'+tag+'.pdf')


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


### MAKE INFERRED DTD PARAMETER PLOT


# sample in BNS DTD parameter posteriors

wts = likes_disk
tmins_disk = np.array(tmins_disk)*1e3 # convert to Myr
alphas_disk = alphas_disk[eu_pts_disk_idxs]
tmins_disk = tmins_disk[eu_pts_disk_idxs]
xcolls_disk = xcolls_disk[eu_pts_disk_idxs]
ybnss_disk = ybnss_disk[eu_pts_disk_idxs]

prior_idxs = np.random.choice(range(len(alphas_disk)),10000,True)


# plot BNS DTD parameter posteriors

fig=plt.figure(figsize=(6.4,2.4))

axx=plt.subplot(1, 3, 1)

tmins_disk_reflect, alphas_disk_reflect = [], []

for alpha,tmin in zip(alphas_disk,tmins_disk):
 
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
sns.kdeplot(x=alphas_compare,y=np.log10(tmins_compare),levels=[0.1,0.32],c=compare_color,axes=axx)
sns.kdeplot(x=list(alphas_disk)+alphas_disk_reflect,y=np.log10(list(tmins_disk)+tmins_disk_reflect),levels=[0.1,0.32],c=color,axes=axx,zorder=10)
sns.kdeplot(x=list(alphas_disk)+alphas_disk_reflect,y=np.log10(list(tmins_disk)+tmins_disk_reflect),levels=[0.32,1.],color=color,alpha=0.25,fill=True,axes=axx,zorder=10)

plt.plot([-10.,-10.],[-10.,-5.],color=compare_color,alpha=1,label='sGRB')
plt.plot([-10.,-10.],[-10.,-5.],color=color,alpha=1,label='BNS')

axx.set_xlim(-3.,-1.)
axx.set_ylim(1.,3.)
axx.set_xlabel(r'$\alpha$')#,fontsize=16)
axx.set_ylabel(r'$\log_{10}\,t_\mathrm{min}/\mathrm{Myr}$')#,fontsize=16)
plt.legend(frameon=True,loc='upper right',fontsize=6)

ax=plt.subplot(1, 3, 2)

alphas_reflect, alphas_disk_reflect = [], []

for alpha,alpha_disk in zip(alphas,alphas_disk):

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
sns.kdeplot(x=alphas_compare,c=compare_color,bw_adjust=1.,cut=20,axes=ax)
ax.hist(list(alphas_disk)+alphas_disk_reflect,density=True,bins=np.arange(-3.,-0.9,0.1),color=color,alpha=0.25,zorder=10)

sns.kdeplot(x=list(alphas_disk)+alphas_disk_reflect,c=color,axes=ax,cut=20,bw_adjust=1.,zorder=10)

plt.plot([-10.,-10.],[-10.,-5.],color='k',alpha=1,label='prior')
plt.plot([-10.,-10.],[-10.,-5.],color=color,alpha=1,label='posterior')

ax.set_xlim(-3.,-1.)
ax.set_ylim(0.,4.)
ax.set_xlabel(r'$\alpha$')#,fontsize=16)
ax.set_ylabel('Probability density')#,fontsize=16)
ax.set_yticks([0.,1.,2.,3.,4.])
ax.set_yticklabels([])

ax=plt.subplot(1, 3, 3)

tmins_reflect, tmins_disk_reflect = [], []

for tmin, tmin_disk in zip(tmins,tmins_disk): # reflect across hard prior bounds
    
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
sns.kdeplot(x=np.log10(tmins_compare),c=compare_color,bw_adjust=1.,cut=20,axes=ax)
ax.hist(list(np.log10(tmins_disk))+list(np.log10(tmins_disk_reflect)),density=True,bins=np.arange(1.,3.1,0.1),color=color,alpha=0.25,zorder=10)

sns.kdeplot(x=np.log10(list(tmins_disk)+tmins_disk_reflect),c=color,axes=ax,cut=20,bw_adjust=1.,zorder=10)

ax.set_xlim(1.,3.)
ax.set_ylim(0.,4.)
ax.set_xlabel(r'$\log_{10}\,t_\mathrm{min}/\mathrm{Myr}$')#,fontsize=16)
ax.set_yticks([0.,1.,2.,3.,4.])
ax.set_ylabel('')
ax.set_yticklabels([])

fig.subplots_adjust(wspace=0.5)
plt.subplots_adjust(bottom=0.2)
plt.savefig('dtd_'+tag+'.pdf')


### MAKE INFERRED YBNS PLOT


# plot rate*mej parameter posteriors

ybnss = np.array(rates)*np.array(mejs)

fig=plt.figure(figsize=(6.4,2.4))

ax=plt.subplot(1, 3, 1)

ybnss_disk_reflect, ybnss_reflect = [], [], []

for ybns_disk,ybns in zip(ybnss_disk,ybnss): # reflect across hard prior bounds
 
    if ybns_disk < 20.:
        ybnss_disk_reflect += [2.*(0.)-ybns_disk]
    '''            
    elif ybns_disk > 180.:
        ybnss_disk_reflect += [2.*(1.)-ybns_disk]
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

alphas_disk_reflect, ybnss_disk_reflect = [], []

for alpha,alpha_halo,ybns,ybns_halo in zip(alphas_disk,ybnss_disk): # reflect across hard prior bounds
 
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
axx.scatter(alphas[prior_idxs],ybnss[prior_idxs],marker='.',s=0.6,c='k',alpha=0.05)
sns.kdeplot(x=list(alphas_disk)+alphas_disk_reflect,y=list(ybnss_disk)+ybnss_disk_reflect,levels=[0.1,0.32],c=color,cut=3,axes=axx)

axx.set_xlim(-3.,-1.)
axx.set_ylim(0.,30.)
axx.set_xlabel(r'$\alpha$')#,fontsize=16)
axx.set_ylabel(r'$m_\mathrm{ej} R_\mathrm{MW}\;[M_\odot/\mathrm{Myr}]$')#,fontsize=16)

axx=plt.subplot(1, 3, 3)

tmins_disk_reflect, ybnss_disk_reflect = [], []

for tmin,ybns in zip(tmins_disk,ybnss_disk): # reflect across hard prior bounds
 
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
axx.scatter(np.log10(tmins[prior_idxs]),ybnss[prior_idxs],marker='.',s=0.6,c='k',alpha=0.05)
sns.kdeplot(x=np.log10(list(tmins_disk)+tmins_disk_reflect),y=list(ybnss_disk)+ybnss_disk_reflect,levels=[0.1,0.32],c=color,cut=3,axes=axx)

axx.set_xlim(1.,3.)
axx.set_ylim(0.,30.)
axx.set_xlabel(r'$\log_{10}\,t_\mathrm{min}/\mathrm{Myr}$')#,fontsize=16)
axx.set_ylabel('')
axx.set_yticklabels([])

fig.subplots_adjust(wspace=0.5)
plt.subplots_adjust(bottom=0.2)
plt.savefig('ybns_'+tag+'.pdf')
