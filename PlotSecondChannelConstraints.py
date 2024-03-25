#!/usr/bin/env python3
# coding: utf-8


'PLOTSECONDCHANNELCONSTRAINTS.PY -- plot posterior samples in binary neutron star rate-ejecta product and second-channel contribution fraction'
__usage__ = 'PlotSecondChannelConstraints.py'
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

DATADIR = '/home/philippe.landry/bns-dtd-rproc/etc/'
OBSPATH = 'Battistini16_disk.csv'

DATADIR2 = '/home/philippe.landry/bns-dtd-rproc/etc/'
OBSPATH2 = 'SAGA_MP.csv'

LIKEDIR = '/home/philippe.landry/bns-dtd-rproc/dat/Battistini_grbbnscls/'
LIKEPATH = 'Battistini16_disk_5M.csv'

LIKEDIR2 = '/home/philippe.landry/bns-dtd-rproc/dat/Battistini_bnscls/'
LIKEPATH2 = 'Battistini16_disk_5M.csv'

POPDIR = '/home/philippe.landry/bns-dtd-rproc/dat/'
POPPATH = 'EuFe_grbbnscls-100.h5'

PARTS = 100
NPOP = 100
NMARG = 500

MAXNUM = PARTS*NPOP*NMARG

color = 'firebrick'
compare_color = 'k'

tag = 'grbbnscls'

FEH_MIN, FEH_MAX = (-3.,0.5)
NFEH = int((FEH_MAX-FEH_MIN)/0.05) # Fe grid spacing for confidence intervals
FeH_grid = np.linspace(FEH_MIN,FEH_MAX,NFEH)


### LOADING INPUTS


# load disk and disk+halo star observations for plotting

FeHs, EuFes, FeH_errs, EuFe_errs = np.loadtxt(DATADIR+OBSPATH, unpack=True, delimiter=',', skiprows=1)
FeHs2, EuFes2, FeH_errs2, EuFe_errs2 = np.loadtxt(DATADIR2+OBSPATH2, unpack=True, delimiter=',', skiprows=1)


# load population realizations and likelihoods

alphas, tmins, X0s, mejs, rates, loglikes = np.loadtxt(LIKEDIR+LIKEPATH, unpack=True, delimiter=',', skiprows=1, max_rows=MAXNUM)

Ys = np.array(rates)*np.array(mejs)
loglikes = loglikes - np.max(loglikes)
likes = np.exp(loglikes)
max_idx = np.argmax(loglikes)

npops = len(alphas)
neff = np.sum(np.array(likes))**2/np.sum(np.array(likes)**2)
print(LIKEDIR+LIKEPATH)
print('number of samples: {0}\n'.format(npops),'number of effective samples: {0}'.format(neff))


# load abundance predictions

keys = [str(key) for key in range(PARTS)]
EuFe_pts, Xs = [], []

for key in tqdm(keys):
    INPUTPATH = POPDIR+'.'.join(POPPATH.split('.')[:-1])+'.part{0}'.format(key)+'.'+POPPATH.split('.')[-1]

    inputdat = h5py.File(INPUTPATH, 'r')
    
    yield_dat_i = inputdat['yield']
    frac_dat_i = inputdat['frac']
    
    for j in range(len(yield_dat_i)):
    
        yield_dat = yield_dat_i[str(j)]
        frac_dat = frac_dat_i[str(j)]
        '''
        func = interp1d(yield_dat['Fe_H'],yield_dat['Eu_Fe'],bounds_error=False)
        eu_pts = func(FeH_grid)
        EuFe_pts += [eu_pts]
        '''
        
        Xts = frac_dat['X']
        zts = frac_dat['z']
        
        Xt_of_z = interp1d(zts,Xts,bounds_error=False)
        Xs += [Xt_of_z(0.)]

EuFe_pts = np.array(EuFe_pts)
Xs = np.array(Xs)


### MAKE ABUNDANCE PREDICTION PLOT


# calculate abundance history confidence envelopes

CLS = [0.68,0.9]
md, qs = [], []

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

for i in range(NFEH):
    
    EuFe_pts_i = np.array([EuFe_pt[i] for EuFe_pt in EuFe_pts])
    
    qs += [wtquantile(EuFe_pts_i,[0.68,0.9],likes)]
    md += [wtquantile(EuFe_pts_i,0.,likes)]
    
qs = np.array(qs)


# plot EuFe vs FeH tracks conditioned on stellar observations

plt.figure(figsize=(6.4,4.8))

plt.fill_between(FeH_grid,qs[:,1],qs[:,3],facecolor=color,edgecolor=None,alpha=0.25, label='BNS+SFH',zorder=10) # 90% CI
plt.fill_between(FeH_grid,qs[:,0],qs[:,2],facecolor=color,edgecolor=None,alpha=0.5,zorder=10) # 68% CI

plt.plot(FeH_grid,md,c=color,zorder=10) # median

plt.errorbar(FeHs2, EuFes2, xerr=[FeH_errs2,FeH_errs2], yerr=[EuFe_errs2,EuFe_errs2], c='g', fmt=',', lw=1, label='SAGA')
plt.scatter(FeHs, EuFes, marker='D',facecolor='dodgerblue',edgecolor='navy', s=16, lw=0.5, label='Battistini & Bensby (2016)')

plt.xlim(-3.,0.5)
plt.ylim(-1.,1.5)
plt.xlabel('[Fe/H]')
plt.ylabel('[Eu/Fe]')
plt.legend(frameon=True,loc='upper right')
plt.savefig('plt/EuFe_'+tag+'.pdf')


### RECORD DTD PARAMETER CONSTRAINTS


# save confidence intervals

print('\n')
print('disk stars\n')
print('maxL population realization')
print('alpha: {0}\n'.format(alphas[max_idx]),'tmin: {0}\n'.format(tmins[max_idx]),'Xcoll: {0}\n'.format(Xs[max_idx]),'Ybns: {0}\n'.format(Ys[max_idx]),'log(maxL): {0}\n'.format(np.max(loglikes)-np.mean(loglikes)),'neff: {0}\n'.format(neff))

print('marginal alpha posterior')
print('mean: {0}\n'.format(np.average(alphas,weights=np.exp(loglikes))),'median: {0}\n'.format(wtquantile(alphas,0.,np.exp(loglikes))),'90: {0}\n'.format(wtquantile(alphas,0.9,np.exp(loglikes))),'68: {0}\n'.format(wtquantile(alphas,0.68,np.exp(loglikes))),'68lb: {0}\n'.format(wtquantile(alphas,0.16,np.exp(loglikes))),'90os: {0}\n'.format(wtquantile(alphas,0.8,np.exp(loglikes))[-1]),'68os: {0}\n'.format(wtquantile(alphas,0.36,np.exp(loglikes)))[-1])

print('marginal tmin posterior')
print('mean: {0}\n'.format(np.average(tmins,weights=np.exp(loglikes))),'median: {0}\n'.format(wtquantile(tmins,0.,np.exp(loglikes))),'90: {0}\n'.format(wtquantile(tmins,0.9,np.exp(loglikes))),'68: {0}\n'.format(wtquantile(tmins,0.68,np.exp(loglikes))),'68lb: {0}\n'.format(wtquantile(tmins,0.16,np.exp(loglikes))),'90os: {0}\n'.format(wtquantile(tmins,0.8,np.exp(loglikes))[-1]),'68os: {0}\n'.format(wtquantile(tmins,0.36,np.exp(loglikes)))[-1])

prior_kde = gaussian_kde(list(Xs[::500])+list(-(Xs[::500])[Xs[::500] < 0.1])+list(2.-(Xs[::500])[Xs[::500] > 0.9]),bw_method='silverman')
inv_wts = prior_kde(Xs)
wts = np.sum(inv_wts)/inv_wts

print('marginal X posterior')
print('mean: {0}\n'.format(np.average(Xs,weights=wts*np.exp(loglikes))),'median: {0}\n'.format(wtquantile(Xs,0.,wts*np.exp(loglikes))),'90: {0}\n'.format(wtquantile(Xs,0.9,wts*np.exp(loglikes))),'68: {0}\n'.format(wtquantile(Xs,0.68,wts*np.exp(loglikes))),'68lb: {0}\n'.format(wtquantile(Xs,0.16,wts*np.exp(loglikes))),'90os: {0}\n'.format(wtquantile(Xs,0.8,wts*np.exp(loglikes))[-1]),'68os: {0}\n'.format(wtquantile(Xs,0.36,wts*np.exp(loglikes)))[-1])

print('marginal rate*mej posterior')
print('mean: {0}\n'.format(np.average(Ys,weights=np.exp(loglikes))),'median: {0}\n'.format(wtquantile(Ys,0.,np.exp(loglikes))),'90: {0}\n'.format(wtquantile(Ys,0.9,np.exp(loglikes))),'68: {0}\n'.format(wtquantile(Ys,0.68,np.exp(loglikes))),'68lb: {0}\n'.format(wtquantile(Ys,0.16,np.exp(loglikes))),'90os: {0}\n'.format(wtquantile(Ys,0.8,np.exp(loglikes))[-1]),'68os: {0}\n'.format(wtquantile(Ys,0.36,np.exp(loglikes)))[-1])


### MAKE INFERRED DTD PARAMETER PLOT


# downsample in BNS DTD parameter posteriors for plotting

num_funcs = 10000

prior_idxs = np.random.choice(range(len(alphas)),num_funcs,True)
alphas_compare = alphas[prior_idxs]
tmins_compare = np.array(tmins[prior_idxs])*1e3 # convert to Myr
Xs_compare = Xs[prior_idxs]
Ys_compare = Ys[prior_idxs]

post_idxs = np.random.choice(range(len(alphas)),num_funcs,True,likes/np.sum(likes))
alphas_disk = alphas[post_idxs]
tmins_disk = np.array(tmins[post_idxs])*1e3 # convert to Myr
Xs_disk = Xs[post_idxs]
Ys_disk = Ys[post_idxs]


# plot BNS DTD parameter posteriors

fig=plt.figure(figsize=(6.4,2.4))

axx=plt.subplot(1, 3, 1)

tmins_disk_reflect, alphas_disk_reflect = [], []
'''
for alpha,tmin in zip(alphas_disk,tmins_disk):
 
    if tmin < 0.015*1e3:
        tmins_disk_reflect += [2*(0.01*1e3)-tmin]
        alphas_disk_reflect += [alpha]
      
    elif tmin > 0.9:
        tmins_disk_reflect += [2*(2.01)-tmin]
        alphas_disk_reflect += [alpha]

    if alpha < -2.9:
        tmins_disk_reflect += [tmin]
        alphas_disk_reflect += [2*(-3.)-alpha]
            
    elif alpha > -0.4:
        tmins_disk_reflect += [tmin]
        alphas_disk_reflect += [2*(-0.5)-alpha]
'''       
sns.kdeplot(x=alphas_compare,y=np.log10(tmins_compare),levels=[0.1,0.32],c=compare_color,axes=axx)
sns.kdeplot(x=list(alphas_disk)+alphas_disk_reflect,y=np.log10(list(tmins_disk)+tmins_disk_reflect),levels=[0.1,0.32],c=color,axes=axx,zorder=10)
sns.kdeplot(x=list(alphas_disk)+alphas_disk_reflect,y=np.log10(list(tmins_disk)+tmins_disk_reflect),levels=[0.32,1.],color=color,alpha=0.25,fill=True,axes=axx,zorder=10)

plt.plot([-10.,-10.],[-10.,-5.],color=compare_color,alpha=1,label='prior')
plt.plot([-10.,-10.],[-10.,-5.],color=color,alpha=1,label='BNS+SFH')

axx.set_xlim(-3.,-1.)
axx.set_ylim(1.,3.)
axx.set_xlabel(r'$\alpha$')#,fontsize=16)
axx.set_ylabel(r'$\log_{10}\,t_\mathrm{min}/\mathrm{Myr}$')#,fontsize=16)
plt.legend(frameon=True,loc='upper right',fontsize=6)

ax=plt.subplot(1, 3, 2)

alphas_reflect, alphas_disk_reflect = [], []
'''
for alpha,alpha_disk in zip(alphas_compare,alphas_disk):

    if alpha < -2.9:
        alphas_reflect += [2*(-3.)-alpha]

    elif alpha > -0.4:
        alphas_reflect += [2*(-0.5)-alpha]
   
    if alpha_disk < -2.9:
        alphas_disk_reflect += [2*(-3.)-alpha_disk]
   
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
'''
for tmin, tmin_disk in zip(tmins_compare,tmins_disk): # reflect across hard prior bounds
    
    if tmin < 0.015*1e3:
        tmins_reflect += [2*(0.01*1e3)-tmin]
      
    elif tmin > 0.9:
        tmins_reflect += [2*(2.01)-tmin]

    if tmin_disk < 0.015*1e3:
        tmins_disk_reflect += [2*(0.01*1e3)-tmin_disk]
   
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
plt.savefig('plt/dtd_'+tag+'.pdf')


### MAKE INFERRED YBNS PLOT


# plot rate*mej parameter posteriors

fig=plt.figure(figsize=(6.4,2.4))

ax=plt.subplot(1, 3, 1)

Ys_disk_reflect, Ys_reflect = [], []

for Y_disk,Y in zip(Ys_disk,Ys_compare): # reflect across hard prior bounds
 
    if Y_disk < 2.:
        Ys_disk_reflect += [2.*(0.)-Y_disk]
    '''            
    elif Ys_disk > 18.:
        Ys_disk_reflect += [2.*(1.)-Y_disk]
    '''
    
    if Y < 2.:
        Ys_reflect += [2.*(0.)-Y]
    '''            
    elif Y > 18.:
        Ys_reflect += [2.*(1.)-Y]
    '''

ax.hist(list(Ys_compare)+Ys_reflect,density=True,bins=np.arange(0.,31.,1.),color=compare_color,alpha=0.1)
ax.hist(list(Ys_disk)+Ys_disk_reflect,density=True,bins=np.arange(0.,31.,1.),color=color,alpha=0.25)

sns.kdeplot(x=list(Ys)+Ys_reflect,color=compare_color,axes=ax,alpha=0.7,cut=20)
sns.kdeplot(x=list(Ys_disk)+Ys_disk_reflect,c=color,axes=ax,cut=20)

plt.plot([-10.,-10.],[-10.,-5.],color=compare_color,alpha=1,label='prior')
plt.plot([-10.,-10.],[-10.,-5.],color=color,alpha=1,label='BNS+SFH')

ax.set_xlim(0.,30.)
ax.set_ylim(0.,0.3)
ax.set_xlabel(r'$m_\mathrm{ej} R_\mathrm{MW}\;[M_\odot/\mathrm{Myr}]$')#,fontsize=16)
ax.set_ylabel('Probability density')#,fontsize=16)
ax.set_yticks([0.,0.075,0.15,0.225,0.3])
ax.set_yticklabels([])
plt.legend(frameon=True,loc='upper right',fontsize=6)

axx=plt.subplot(1, 3, 2)

alphas_disk_reflect, Ys_disk_reflect = [], []

for alpha,Ys in zip(alphas_disk,Ys_disk): # reflect across hard prior bounds
 
    if Ys < 2.:
        alphas_disk_reflect += [alpha]
        Ys_disk_reflect += [2.*(0.)-Ys]
    '''            
    elif Ys > 18.:
        alphas_disk_reflect += [alpha]
        Ys_disk_reflect += [2.*(1.)-Ys]

    if alpha < -2.9:
        alphas_disk_reflect += [2*(-3.)-alpha]
        Ys_disk_reflect += [Ys]
    
    elif alpha > -0.4:
        alphas_disk_reflect += [2*(-0.5)-alpha]
        Ys_disk_reflect += [Ys]
    '''
axx.scatter(alphas_compare,Ys_compare,marker='.',s=0.6,c=compare_color,alpha=0.05)
sns.kdeplot(x=list(alphas_disk)+alphas_disk_reflect,y=list(Ys_disk)+Ys_disk_reflect,levels=[0.1,0.32],c=color,cut=3,axes=axx)

axx.set_xlim(-3.,-1.)
axx.set_ylim(0.,30.)
axx.set_xlabel(r'$\alpha$')#,fontsize=16)
axx.set_ylabel(r'$m_\mathrm{ej} R_\mathrm{MW}\;[M_\odot/\mathrm{Myr}]$')#,fontsize=16)

axx=plt.subplot(1, 3, 3)

tmins_disk_reflect, Ys_disk_reflect = [], []

for tmin,Ys in zip(tmins_disk,Ys_disk): # reflect across hard prior bounds
 
    if Ys < 2.:
        tmins_disk_reflect += [tmin]
        Ys_disk_reflect += [2.*(0.)-Ys]
    '''            
    elif Ys > 18.:
        tmins_disk_reflect += [tmin]
        Ys_disk_reflect += [2.*(1.)-Ys]

    if tmin < 0.015*1e3:
        tmins_disk_reflect += [2*(0.01*1e3)-tmin]
        Ys_disk_reflect += [Ys]
        
    elif tmin > 0.9:
        tmins_disk_reflect += [2*(2.01)-tmin]
        Ys_disk_reflect += [Ys]
    '''
axx.scatter(np.log10(tmins_compare),Ys_compare,marker='.',s=0.6,c=compare_color,alpha=0.05)
sns.kdeplot(x=np.log10(list(tmins_disk)+tmins_disk_reflect),y=list(Ys_disk)+Ys_disk_reflect,levels=[0.1,0.32],c=color,cut=3,axes=axx)

axx.set_xlim(1.,3.)
axx.set_ylim(0.,30.)
axx.set_xlabel(r'$\log_{10}\,t_\mathrm{min}/\mathrm{Myr}$')#,fontsize=16)
axx.set_ylabel('')
axx.set_yticklabels([])

fig.subplots_adjust(wspace=0.5)
plt.subplots_adjust(bottom=0.2)
plt.savefig('plt/ybns_'+tag+'.pdf')


### MAKE INFERRED XSFR PLOT


# plot Xsfr parameter posteriors

fig=plt.figure(figsize=(6.4,2.4))

ax=plt.subplot(1, 3, 1)

Xs_disk_reflect, Xs_compare_reflect = [], []

for X,X_compare in zip(Xs_disk,Xs_compare): # reflect across hard prior bounds

    if X < 0.1:
        Xs_disk_reflect += [2.*(0.)-X]
     
    if X > 0.9:
        Xs_disk_reflect += [2.*(1.)-X]
        
    if X_compare < 0.1:
        Xs_compare_reflect += [2.*(0.)-X_compare]
            
    elif X_compare > 0.9:
        Xs_compare_reflect += [2.*(1.)-X_compare]

prior_kde = gaussian_kde(list(Xs_disk[::500])+list(-(Xs_disk[::500])[Xs_disk[::500] < 0.1])+list(2.-(Xs_disk[::500])[Xs_disk[::500] > 0.9]),bw_method='silverman')
inv_wts = prior_kde(list(Xs_disk)+Xs_disk_reflect)
wts = np.sum(inv_wts)/inv_wts

prior_kde2 = gaussian_kde(list(Xs_compare[::500])+list(-(Xs_compare[::500])[Xs_compare[::500] < 0.1])+list(2.-(Xs_compare[::500])[Xs_compare[::500] > 0.9]),bw_method='silverman')
inv_wts2 = prior_kde2(list(Xs_compare)+Xs_compare_reflect)
wts2 = np.sum(inv_wts2)/inv_wts2
        
ax.hist(list(Xs_compare)+Xs_compare_reflect,density=True,bins=np.arange(0.0,1.05,0.05),color=compare_color,alpha=0.1,weights=wts2)
ax.hist(list(Xs_disk)+Xs_disk_reflect,density=True,bins=np.arange(0.0,1.05,0.05),color=color,alpha=0.25,weights=wts)

sns.kdeplot(x=list(Xs_compare)+Xs_compare_reflect,color=compare_color,axes=ax,cut=20,alpha=0.7,weights=wts2)
sns.kdeplot(x=list(Xs_disk)+Xs_disk_reflect,c=color,axes=ax,cut=20,weights=wts)

plt.plot([-10.,-10.],[-10.,-5.],color='k',alpha=1,label='prior')
plt.plot([-10.,-10.],[-10.,-5.],color=color,alpha=1,label='BNS+SFH')

ax.set_xlim(0.,1.)
ax.set_ylim(0.,4.)
ax.set_xlabel(r'$X_\mathrm{SFH}$')#,fontsize=16)
ax.set_ylabel('Probability density')#,fontsize=16)
ax.set_yticks([0.,1.,2.,3.,4.])
ax.set_yticklabels([])
plt.legend(frameon=True,loc='upper left',fontsize=6)

axx=plt.subplot(1, 3, 2)

alphas_disk_reflect, alphas_compare_reflect, Xs_disk_reflect, Xs_compare_reflect = [], [], [], []

for alpha,alpha_compare,X,X_compare in zip(alphas_disk,alphas_compare,Xs_disk,Xs_compare):

    if X < 0.1:
        alphas_disk_reflect += [alpha]
        Xs_disk_reflect += [2.*(0.)-X]
     
    if X > 0.9:
        alphas_disk_reflect += [alpha]
        Xs_disk_reflect += [2.*(1.)-X]
        
    if X_compare < 0.1:
        alphas_compare_reflect += [alpha_compare]
        Xs_compare_reflect += [2.*(0.)-X_compare]
            
    elif X_compare > 0.9:
        alphas_compare_reflect += [alpha_compare]
        Xs_compare_reflect += [2.*(1.)-X_compare]

prior_kde = gaussian_kde(list(Xs_disk[::500])+list(-(Xs_disk[::500])[Xs_disk[::500] < 0.1])+list(2.-(Xs_disk[::500])[Xs_disk[::500] > 0.9]),bw_method='silverman')
inv_wts = prior_kde(list(Xs_disk)+Xs_disk_reflect)
wts = np.sum(inv_wts)/inv_wts
        
axx.scatter(alphas_compare,Xs_compare,marker='.',s=0.6,c='k',alpha=0.1)
sns.kdeplot(x=list(alphas_disk)+alphas_disk_reflect,y=list(Xs_disk)+Xs_disk_reflect,levels=[0.1,0.32],c=color,cut=20,axes=axx,weights=wts)

axx.set_xlim(-3.,-1.)
axx.set_ylim(0.,1.)
axx.set_xlabel(r'$\alpha$')#,fontsize=16)
axx.set_ylabel(r'$X_\mathrm{SFH}$')#,fontsize=16)

axx=plt.subplot(1, 3, 3)

tmins_disk_reflect, tmins_compare_reflect, Xs_disk_reflect, Xs_compare_reflect = [], [], [], []

for tmin,tmin_compare,X,X_compare in zip(tmins_disk,tmins_compare,Xs_disk,Xs_compare):

    if X < 0.1:
        tmins_disk_reflect += [tmin]
        Xs_disk_reflect += [2.*(0.)-X]
     
    if X > 0.9:
        tmins_disk_reflect += [tmin]
        Xs_disk_reflect += [2.*(1.)-X]
        
    if X_compare < 0.1:
        tmins_compare_reflect += [tmin_compare]
        Xs_compare_reflect += [2.*(0.)-X_compare]
            
    elif X_compare > 0.9:
        tmins_compare_reflect += [tmin_compare]
        Xs_compare_reflect += [2.*(1.)-X_compare]

prior_kde = gaussian_kde(list(Xs_disk[::500])+list(-(Xs_disk[::500])[Xs_disk[::500] < 0.1])+list(2.-(Xs_disk[::500])[Xs_disk[::500] > 0.9]),bw_method='silverman')
inv_wts = prior_kde(list(Xs_disk)+Xs_disk_reflect)
wts = np.sum(inv_wts)/inv_wts
        
axx.scatter(np.log10(tmins_compare),Xs_compare,marker='.',s=0.6,c='k',alpha=0.1)
sns.kdeplot(x=np.log10(list(tmins_disk)+tmins_disk_reflect),y=list(Xs_disk)+Xs_disk_reflect,levels=[0.1,0.32],c=color,cut=20,axes=axx,weights=wts)

axx.set_xlim(1.,3.)
axx.set_ylim(0.,1.)
axx.set_xlabel(r'$\log_{10}\,t_\mathrm{min}/\mathrm{Myr}$')#,fontsize=16)
axx.set_ylabel('')
axx.set_yticklabels([])

plt.subplots_adjust(wspace=0.5)
plt.subplots_adjust(bottom=0.2)
plt.savefig('plt/xsfr_'+tag+'.pdf')


### MAKE INFERRED RATE-MEJ PLOT


# plot rate-mej parameter posteriors

fig=plt.figure(figsize=(6.4,2.4))

axx=plt.subplot(1, 3, 1)

Ys_disk_reflect, Ys_compare_reflect, Xs_disk_reflect, Xs_compare_reflect, = [], [], [], []

for Y,X,Y_compare,X_compare in zip(Ys_disk,Xs_disk,Ys_compare,Xs_compare):

    if X < 0.1:
        Ys_disk_reflect += [Y]
        Xs_disk_reflect += [2.*(0.)-X]
     
    if X > 0.9:
        Ys_disk_reflect += [Y]
        Xs_disk_reflect += [2.*(1.)-X]
        
    if Y < 2.:
        Xs_disk_reflect += [X]
        Ys_disk_reflect += [2.*(0.)-Y]
    '''            
    elif Y > 18.:
        Xs_disk_reflect += [X]
        Ys_disk_reflect += [2.*(200.)-Ys]
    '''
    if X_compare < 0.1:
        Ys_compare_reflect += [Ys_compare]
        Xs_compare_reflect += [2.*(0.)-X_compare]
            
    elif X_compare > 0.9:
        Ys_compare_reflect += [Ys_compare]
        Xs_compare_reflect += [2.*(1.)-X_compare]
        
    if Y_compare < 2.:
        Xs_compare_reflect += [X_compare]
        Ys_compare_reflect += [2.*(0.)-Y_compare]
    '''            
    elif Y_compare > 18.:
        Xs_compare_reflect += [X_compare]
        Ys_compare_reflect += [2.*(200.)-Y_compare]
    '''

prior_kde = gaussian_kde(list(Xs_disk[::500])+list(-(Xs_disk[::500])[Xs_disk[::500] < 0.1])+list(2.-(Xs_disk[::500])[Xs_disk[::500] > 0.9]),bw_method='silverman')
inv_wts = prior_kde(list(Xs_disk)+Xs_disk_reflect)
wts = np.sum(inv_wts)/inv_wts

prior_kde2 = gaussian_kde(list(Xs_compare[::500])+list(-(Xs_compare[::500])[Xs_compare[::500] < 0.1])+list(2.-(Xs_compare[::500])[Xs_compare[::500] > 0.9]),bw_method='silverman')
inv_wts2 = prior_kde2(list(Xs_compare)+Xs_compare_reflect)
wts2 = np.sum(inv_wts2)/inv_wts2
    
axx.scatter(Ys_compare,Xs_compare,marker='.',s=0.6,c='k',alpha=0.05)
sns.kdeplot(x=list(Ys_disk)+Ys_disk_reflect,y=list(Xs_disk)+Xs_disk_reflect,levels=[0.1,0.32],c=color,cut=20,axes=axx,weights=wts)
sns.kdeplot(x=list(Ys_disk)+Ys_disk_reflect,y=list(Xs_disk)+Xs_disk_reflect,levels=[0.32,1.],color=color,alpha=0.25,fill=True,cut=20,axes=axx,weights=wts)

plt.plot([-10.,-10.],[-10.,-5.],color='k',alpha=1,label='prior')
plt.plot([-10.,-10.],[-10.,-5.],color=color,alpha=1,label='BNS+SFH')

axx.set_xlim(0.,30.)
axx.set_ylim(0.,1.)
axx.set_xlabel(r'$m_\mathrm{ej} R_\mathrm{MW}\;[M_\odot/\mathrm{Myr}]$')#,fontsize=16)
axx.set_ylabel(r'$X_\mathrm{SFH}$')#,fontsize=16)
plt.legend(frameon=True,loc='upper right',fontsize=6)

ax=plt.subplot(1, 3, 2)

Ys_disk_reflect, Ys_compare_reflect = [], []

for Y,Y_compare in zip(Ys_disk,Ys_compare):
        
    if Y < 2.:
        Ys_disk_reflect += [2.*(0.)-Y]
    '''            
    elif Y > 18.:
        Ys_reflect += [2.*(200.)-Y]
    '''
    if Y_compare < 2.:
        Ys_compare_reflect += [2.*(0.)-Y_compare]
    '''            
    elif Y_compare > 18.:
        Ys_compare_reflect += [2.*(200.)-Y_compare]
    '''
   
ax.hist(list(Ys_compare)+Ys_compare_reflect,density=True,bins=np.arange(0.,31.,1.),color=compare_color,alpha=0.1)
ax.hist(list(Ys_disk)+Ys_disk_reflect,density=True,bins=np.arange(0.,31.,1.),color=color,alpha=0.25)

sns.kdeplot(x=list(Ys_compare)+Ys_compare_reflect,color=compare_color,axes=ax,alpha=0.7,cut=20)
sns.kdeplot(x=list(Ys_disk)+Ys_disk_reflect,c=color,axes=ax,cut=20)

ax.set_xlim(0.,30.)
ax.set_ylim(0.,0.3)
ax.set_xlabel(r'$m_\mathrm{ej} R_\mathrm{MW}\;[M_\odot/\mathrm{Myr}]$')#,fontsize=16)
ax.set_ylabel('Probability density')#,fontsize=16)
ax.set_yticks([0.,0.075,0.15,0.225,0.3])
ax.set_yticklabels([])

ax=plt.subplot(1, 3, 3)

Xs_disk_reflect, Xs_compare_reflect = [], []

for X,X_compare in zip(Xs_disk,Xs_compare):

    if X < 0.1:
        Xs_disk_reflect += [2.*(0.)-X]
       
    if X > 0.9:
        Xs_disk_reflect += [2.*(1.)-X]
        
    if X_compare < 0.1:
        Xs_compare_reflect += [2.*(0.)-X_compare]
            
    elif X_compare > 0.9:
        Xs_compare_reflect += [2.*(1.)-X_compare]

prior_kde = gaussian_kde(list(Xs_disk[::500])+list(-(Xs_disk[::500])[Xs_disk[::500] < 0.1])+list(2.-(Xs_disk[::500])[Xs_disk[::500] > 0.9]),bw_method='silverman')
inv_wts = prior_kde(list(Xs_disk)+Xs_disk_reflect)
wts = np.sum(inv_wts)/inv_wts

prior_kde2 = gaussian_kde(list(Xs_compare[::500])+list(-(Xs_compare[::500])[Xs_compare[::500] < 0.1])+list(2.-(Xs_compare[::500])[Xs_compare[::500] > 0.9]),bw_method='silverman')
inv_wts2 = prior_kde2(list(Xs_compare)+Xs_compare_reflect)
wts2 = np.sum(inv_wts2)/inv_wts2
        
ax.hist(list(Xs_compare)+Xs_compare_reflect,density=True,bins=np.arange(0.0,1.05,0.05),color=compare_color,alpha=0.1,weights=wts2)
ax.hist(list(Xs_disk)+Xs_disk_reflect,density=True,bins=np.arange(0.0,1.05,0.05),color=color,alpha=0.25,weights=wts)

sns.kdeplot(x=list(Xs_compare)+Xs_compare_reflect,color=compare_color,axes=ax,alpha=0.7,cut=20,weights=wts2)
sns.kdeplot(x=list(Xs_disk)+Xs_disk_reflect,c=color,axes=ax,cut=20,weights=wts)

ax.set_xlim(0.,1.)
ax.set_ylim(0.,4.)
ax.set_xlabel(r'$X_\mathrm{SFH}$')#,fontsize=16)
ax.set_yticks([0.,1.,2.,3.,4.])
ax.set_ylabel('')
ax.set_yticklabels([])

fig.subplots_adjust(wspace=0.5)
plt.subplots_adjust(bottom=0.2)
plt.savefig('plt/ratemej_'+tag+'.pdf')


### SAVAGE-DICKEY DENSITY RATIOS

Xs_reflect, likes_reflect = [], []

for X,like in zip(Xs,likes):
    if X < 0.1: 
        Xs_reflect += [2.*(0.)-X]
        likes_reflect += [like]

kde = gaussian_kde(list(Xs)+Xs_reflect,bw_method='silverman',weights=(list(likes)+likes_reflect)/np.sum(list(likes)+likes_reflect))
prior_kde = gaussian_kde(list(Xs)+Xs_reflect,bw_method='silverman')
bf = float(kde(0.)/prior_kde(0.))

print('Bayes factor for one channel vs two, given sGRBs')
print(bf)
