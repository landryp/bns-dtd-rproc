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
from scipy.interpolate import interp1d
from scipy.stats import multivariate_normal, gaussian_kde, loguniform
from scipy.integrate import cumtrapz
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import emcee
import time
import os
import h5py
from tqdm import tqdm

from etc.rProcessUtils import rho_MW
from etc.rProcessChemicalEvolution import rproc_evolution


parser = ArgumentParser(description=__doc__)
parser.add_argument('outdir')
parser.add_argument('obspath')
parser.add_argument('dtdpath')
parser.add_argument('ejpath')
parser.add_argument('-s','--solpath',default='etc/Arnould07_solar_rprocess.dat')
parser.add_argument('-a','--alpha',default="-3.,-0.5")
parser.add_argument('-t','--tmin',default="1e-2,2.01")
parser.add_argument('-x','--xsfh',default="1e-3,0.999")
parser.add_argument('-r','--ratemej',default="0.,15.")
parser.add_argument('--tag',default=None)
parser.add_argument('--multi',default=False)
parser.add_argument('-n','--npost',default=10000)
parser.add_argument('-w','--nwalk',default=21)
parser.add_argument('-b','--nburn',default=100)
parser.add_argument('-m','--maxobs',default=None)
parser.add_argument('-k','--nkde',default=None)


args = parser.parse_args()

OUTDIR = str(args.outdir) # 'dat/' # output directory for plots, likelihood-weighted population samples
OBSPATH = str(args.obspath) # 'data/SAGA_MP.csv' # observations of stellar Eu vs Fe abundances
DTDPATH = str(args.dtdpath) # 'etc/label_samples.dat' # GRB-informed DTD parameter distributions # '' # uniform DTD parameter distribution
EJECTAPATH = str(args.ejpath) # 'etc/mej_gal_lcehl_nicer_numuncertainty.txt' # input samples in ejecta mass and rate
SOLARPATH = str(args.solpath) # 'etc/arnould_07_solar_r-process.txt' # solar abundances
ALPHA_BOUNDS = [float(bnd) for bnd in str(args.alpha).split(',')] # (-3.,-0.5) # bounds for uniform prior on DTD power law index
TDMIN_BOUNDS = [float(bnd) for bnd in str(args.tmin).split(',')] # (1e-2,2.01) # bounds for log-uniform prior on minimum delay time in Gyr # see below for option to change to uniform prior
EU_RATIO_Z0_BOUNDS = [float(bnd) for bnd in str(args.xsfh).split(',')] # (1e-6,1.) # bounds for uniform prior on fractional contribution of collapsar channel to local r-process mass*rate density (i.e. m_Eu_coll*norm_coll/(m_Eu_coll*norm_coll + m_Eu_bns*norm_bns))
RATEMEJ_BOUNDS = [float(bnd) for bnd in str(args.ratemej).split(',')] # (0.,15.) # bounds for uniform prior on rate*mej
if args.tag is not None:
	TAG = str(args.tag) # optional tag for output file
else: TAG = None
NPOST = int(args.npost) # number of abundance predictions to calculate -- equals number of BNS DTD posterior samples to target with mcmc
NWALK = int(args.nwalk) # number of mcmc walkers to evolve
NBURN = int(args.nburn) # number of mcmc burn-in samples to discard
if args.maxobs is not None: MAXOBS = int(args.maxobs) # number of observations to consider
else: MAXOBS = None
if args.nkde is not None: NUM = int(args.nkde) # number of prior samples to draw for building kdes
else: NUM = None
if args.multi: 
    from multiprocessing import Pool
    import multiprocessing

FEH_MIN, FEH_MAX = (-3.,0.5)
NFEH = int((FEH_MAX-FEH_MIN)/0.05) # Fe grid spacing for chemical evolution tracks

NSTP = 20 # number of subdivisions per time step in chemical evolution integration

Z_MIN, Z_MAX = (0.,10.)
NZ = int((Z_MAX-Z_MIN)/0.1) # z grid spacing for abundance ratio evolution

KDENORMPTS = 100 # resolution to use when computing kde norms

PARAMS = ['alpha','log10tmin','ratemej','X0']

if not os.path.exists(OUTDIR):
	os.makedirs(OUTDIR)


### SAMPLE FROM PRIORS


# load ejecta mass and rate samples, and build kde

mej_dyn, mej_dsk, rate = np.loadtxt(EJECTAPATH, unpack=True)
mej = mej_dyn + mej_dsk
rate = rate*(1e9*1e-2)/1e6 # convert to Gpc^-3 yr^-1
ratemej = rate*mej

if NUM is None: NUM = len(ratemej)

idxs = np.random.choice(range(len(ratemej)),NUM,False)
ratemej = ratemej[idxs]

prior_ratemej_nonorm = gaussian_kde(ratemej)
ratemej_grid = np.linspace(*RATEMEJ_BOUNDS,KDENORMPTS)
ratemej_kde_norm = np.trapz(prior_ratemej_nonorm(ratemej_grid), ratemej_grid)
prior_ratemej = lambda ratemej: prior_ratemej_nonorm(ratemej)/ratemej_kde_norm


# sample in BNS delay time distribution, and build kde

if DTDPATH != '':
    
	alpha, tdmin, tdmax = np.loadtxt(DTDPATH, unpack=True, skiprows=1)
	tdmin = tdmin/1e9 # convert to Gyr
    
	if args.nkde is None: NUM = len(alpha)
    
	idxs = np.random.choice(range(len(alpha)),NUM,False)
	alphas = alpha[idxs]
	tdmins = tdmin[idxs]
    
	prior_alphalog10tmin_nonorm = gaussian_kde(np.row_stack((alphas,np.log10(tdmins))))
	alpha_grid = np.linspace(*ALPHA_BOUNDS,KDENORMPTS)
	log10tmin_grid = np.linspace(np.log10(TDMIN_BOUNDS[0]),np.log10(TDMIN_BOUNDS[1]),KDENORMPTS)
	x,y = np.meshgrid(alpha_grid,log10tmin_grid)
	alphalog10tmin_kde_norm = np.trapz(np.trapz(prior_alphalog10tmin_nonorm(np.vstack([x.ravel(),y.ravel()])).reshape(len(x),len(y)), alpha_grid, axis=0),log10tmin_grid, axis=0)
	def prior_alphalog10tmin(alpha,log10tmin):
		return prior_alphalog10tmin_nonorm((alpha,log10tmin))/alphalog10tmin_kde_norm

else:
    
	alphas = np.random.uniform(*ALPHA_BOUNDS,NUM) # uniform prior on DTD power law index
	tdmins = loguniform.rvs(*TDMIN_BOUNDS,size=NUM) # log-uniform prior on minimum delay time in Gyr
    
	prior_alphalog10tmin = lambda xy: 1./((ALPHA_BOUNDS[1]-ALPHA_BOUNDS[0])*(np.log10(TDMIN_BOUNDS[1])-np.log10(TDMIN_BOUNDS[0])))


# sample in collapsar yield

if args.nkde is None: NUM = min(len(alpha),len(ratemej))

Eu_ratio_z0s = np.random.uniform(*EU_RATIO_Z0_BOUNDS,NUM) # uniform prior on ratio of collapsar to total Eu abundance at z=0, what we call X0 below

prior_X0 = lambda X0: 1./(EU_RATIO_Z0_BOUNDS[1]-EU_RATIO_Z0_BOUNDS[0])


# save prior bounds to dict and determine which mcmc parameters to sample

params_dict = {}
params_dict['samples'] = {}
params_dict['bounds'] = {}
params_dict['infer'] = {}

params_dict['samples']['alpha'] = alphas[(alphas >= ALPHA_BOUNDS[0]) & (alphas <= ALPHA_BOUNDS[1])]
params_dict['samples']['log10tmin'] = np.log10(tdmins[(tdmins >= TDMIN_BOUNDS[0]) & (tdmins <= TDMIN_BOUNDS[1])])
params_dict['samples']['ratemej'] = ratemej[(ratemej >= RATEMEJ_BOUNDS[0]) & (ratemej <= RATEMEJ_BOUNDS[1])]
params_dict['samples']['X0'] = Eu_ratio_z0s[(Eu_ratio_z0s >= EU_RATIO_Z0_BOUNDS[0]) & (Eu_ratio_z0s <= EU_RATIO_Z0_BOUNDS[1])]

params_dict['bounds']['alpha'] = ALPHA_BOUNDS
params_dict['bounds']['log10tmin'] = np.log10(TDMIN_BOUNDS)
params_dict['bounds']['ratemej'] = RATEMEJ_BOUNDS
params_dict['bounds']['X0'] = EU_RATIO_Z0_BOUNDS

NDIM = 0

for i,param in enumerate([alpha,np.log10(tdmin),ratemej,Eu_ratio_z0s]):
    if len(list(set(param))) > 1:
        NDIM += 1
        params_dict['infer'][PARAMS[i]] = True
    else: params_dict['infer'][PARAMS[i]] = False


### BUILD LIKELIHOOD FUNCTIONS FOR OBSERVATIONS


# load stellar spectrum data

FeHs, EuFes, FeH_errs, EuFe_errs = np.loadtxt(OBSPATH, unpack=True, delimiter=',', skiprows=1, max_rows=MAXOBS)


# make gaussian likelihood model for each spectrum datapoint

like_means = []
like_stds = []

for fe,eu,fe_err,eu_err in zip(FeHs, EuFes, FeH_errs, EuFe_errs):

    mean = np.array([fe,eu])
    std = np.array([[fe_err,0.],[0.,eu_err]])
    
    like_means += [mean]
    like_stds += [std]   


### DO INFERENCE OF BNS DTD AND COLLAPSAR CONTRIBUTION


# define prior, likelihood and posterior for mcmc

def log_prior(params_array):
    
    alpha,log10tmin,ratemej,X0 = params_array
    
    if alpha > ALPHA_BOUNDS[1] or alpha < ALPHA_BOUNDS[0]: return -np.inf
    if log10tmin > np.log10(TDMIN_BOUNDS[1]) or log10tmin < np.log10(TDMIN_BOUNDS[0]): return -np.inf
    if ratemej > RATEMEJ_BOUNDS[1] or ratemej < RATEMEJ_BOUNDS[0]: return -np.inf
    if X0 > EU_RATIO_Z0_BOUNDS[1] or X0 < EU_RATIO_Z0_BOUNDS[0]: return -np.inf
    
    return np.log(prior_ratemej(ratemej)*prior_alphalog10tmin(alpha,log10tmin)*prior_X0(X0))

def log_likelihood(params_array, like_means, like_stds):
    
    alpha,log10tmin,ratemej,X0 = params_array
    
    b_NS = -alpha
    tmin_NS = 1e-3*10.**log10tmin
    
    FeH_track, EuFe_track, rate_evs = rproc_evolution(1.,ratemej/1.,b_NS,tmin_NS,X0,0.5,NSTP)

    track = np.column_stack((FeH_track,EuFe_track))
    track = track[np.where(track[:,1] >= -5.)[0][0]:]
    track = track[~np.isnan(track[:,1])]
    if len(track) < 1: return -np.inf
    
    loglikes = [np.log(np.trapz(multivariate_normal(mean,std).pdf(track),track[:,0])) for mean,std in zip(like_means,like_stds)]
    
    return np.sum(loglikes)

def log_posterior(theta, like_means, like_stds, PARAMS, params_dict):
    
    i = 0
    params_array = []
    for param in PARAMS:
        if params_dict['infer'][param]:
            params_array += [theta[i]]
            i += 1
        else: params_array += [params_dict['samples'][param][0]]
    params_array = np.array(params_array)
    
    logprior = log_prior(params_array)
    
    if not np.isfinite(logprior): return -np.inf
    
    return logprior + log_likelihood(params_array, like_means, like_stds)


# do mcmc sampling with emcee

i = 0
init = np.empty((NWALK, NDIM))
for param in PARAMS:
    if params_dict['infer'][param]:
        init[:,i] = np.random.choice(params_dict['samples'][param],NWALK,False)
        i += 1

if args.multi:        
    with multiprocessing.Pool(NWALK) as pool:
        print('running')
        sampler = emcee.EnsembleSampler(NWALK, NDIM, log_posterior, args=(like_means, like_stds, PARAMS, params_dict), pool=pool)
        start = time.time()
        sampler.run_mcmc(init, NPOST, progress=True)
        end = time.time()
        elapsed = end - start
        print("multiprocessing took {0:.1f} min".format(elapsed/60.))
else:
    print('running')
    sampler = emcee.EnsembleSampler(NWALK, NDIM, log_posterior, args=(like_means, like_stds, PARAMS, params_dict))
    start = time.time()
    sampler.run_mcmc(init, NPOST, progress=True)
    end = time.time()
    elapsed = end - start
    print("emcee took {0:.1f} min".format(elapsed/60.))

chains = sampler.get_chain().reshape(NWALK,NPOST,NDIM)
acls = sampler.get_autocorr_time(quiet=True)
if np.any(np.isnan(acls)): acls = [1.]
flat_samples = sampler.get_chain(flat=True)[NBURN::int(max(acls))]
log_prob = sampler.get_log_prob()[NBURN::int(max(acls))]


# compute predicted galactic r-process enrichment histories

FeH_grid = np.linspace(FEH_MIN,FEH_MAX,NFEH)
z_grid = np.linspace(Z_MIN,Z_MAX,NZ)
EuFe_pts, Xts = [], []

outdat = {}
outdat['pop'] = {}
outdat['yield'] = {}
outdat['frac'] = {}

for i,(alpha,log10tmin,ratemej,X0) in tqdm(enumerate(flat_samples)):
    
    b_NS = -alpha
    tmin_NS = 1e-3*10.**log10tmin
    
    FeH_track, EuFe_track, rate_evs = rproc_evolution(1.,ratemej/1.,b_NS,tmin_NS,X0,0.5,NSTP)
    Rbns, Rcoll, Xs, zs, ts = rate_evs
    
    track = np.column_stack((FeH_track,EuFe_track))
    track = track[np.where(track[:,1] >= -5.)[0][0]:]
    track = track[~np.isnan(track[:,1])]
    
    EuFe_of_FeH = interp1d(FeH_track,EuFe_track,bounds_error=False)
    EuFe_pts += [EuFe_of_FeH(FeH_grid)]
    
    Rcoll0 = Rcoll[-1]
    mcoll = ratemej/((1./X0 - 1.)*Rcoll0)
    num = cumtrapz(ratemej*Rbns/1.,t)
    denom = cumtrapz(mcoll*Rcoll,t)
    Xt = 1./(1.+num/denom)
    
    Xt_of_z = interp1d(zs[1:],Xt,bounds_error=False)
    Xts += [Xt_of_z(z_grid)]
    
    outdat['pop'][i] = np.array([alpha,log10tmin,ratemej,X0])
    outdat['yield'][i] = {}
    outdat['yield'][i]['Fe_H'] = FeH_track
    outdat['yield'][i]['Eu_Fe'] = EuFe_track
    outdat['frac'][i] = {}
    outdat['frac'][i]['Rbns'] = Rbns
    outdat['frac'][i]['Rcoll'] = Rcoll
    outdat['frac'][i]['z'] = zs
    outdat['frac'][i]['t'] = ts
    
    outdat['frac'][i]['X'] = list(Xts)+[Xts[-1]]
    
EuFe_pts = np.array(EuFe_pts)
Xts = np.array(Xts)
    

### SAVE RESULTS


# save r-process abundance data and popoulation parameters

if TAG is None: OUTPATH = OUTDIR+'/'+(OBSPATH.split('/')[-1]).split('.')[0]+'_{0}.h5'.format(NUM)
else: OUTPATH = OUTDIR+'/'+(OBSPATH.split('/')[-1]).split('.')[0]+'_{0}.{1}.h5'.format(NUM,TAG)

outfile = h5py.File(OUTPATH, 'w')

pop_set = outfile.create_group('pop')
for key, value in outdat['pop'].items():
    pop_data = np.array(value)
    pop_data = rfn.unstructured_to_structured(pop_data, np.dtype([('alpha', 'f8'), ('log10tmin', 'f8'), ('ratemej', 'f8'), ('X0', 'f8')]))
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


# plot mcmc samples and traces

df = pd.DataFrame(flat_samples,columns=PARAMS)
g = sns.pairplot(df,corner=True,diag_kind='kde',plot_kws={'size': 2, 'alpha': 0.1})
g.map_offdiag(sns.kdeplot,levels=[0.1,0.5])
plt.savefig('.'.join(OUTPATH.split('.')[:-1])+'_corner.png')

fig = plt.figure()
gs = gridspec.GridSpec(NDIM, 1)
axs = [plt.subplot(gs[i]) for i in range(NDIM)]
plt.subplots_adjust(hspace=0.05)

for i,ax in enumerate(axs):
    for chain in chains: ax.plot(chain[:,i], c=sns.color_palette()[0], alpha=0.2)
    ax.set_ylabel(samples.columns[i])
    ax.axvline(NBURN,lw=1,ls='--',c='k')
    if i < len(axs)-1: ax.tick_params(labelbottom=False)

axs[-1].set_xlabel('steps')
plt.savefig('.'.join(OUTPATH.split('.')[:-1])+'_traces.png')