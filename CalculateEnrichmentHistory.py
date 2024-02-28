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

from etc.SiegelOneZone import * # import Daniel Siegel's one-zone r-process code


parser = ArgumentParser(description=__doc__)
parser.add_argument('nsamp')
parser.add_argument('outpath')
parser.add_argument('dtdpath')
parser.add_argument('ejpath')
parser.add_argument('-s','--solpath',default='etc/arnould_07_solar_r-process.txt')
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
NUM_MARG = int(args.marg) # 100 # number of mej and rate samples per DTD sample


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
    
    alpha = np.random.choice(alpha,ndtd,False)
    tdmin = np.random.choice(tdmin,ndtd,False)

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
bns_dat = np.zeros((ndtd*nsamps,6))

for i,(alpha,tdmin,Xcoll) in tqdm(enumerate(zip(alphas,tdmins,Eu_ratio_z0s))):
    
    for j,(mej,rate,f_NSgal) in enumerate(zip(mejs,rates,fNSs)):
    
        m_r_NS = mej
        b_NS = [-alpha]
        tmin_NS = tdmin

        dt_max = min(tmin_NS,tmin_Ia,tmin) / nppdt # max step size
        n_dt_max = int(round((tmax - tmin) / dt_max)) # number of grid points
        ts = linspace(tmin,tmax,n_dt_max)
        zs_ts = z_t(ts,t_int_min)

        ts_tab = ts
        psi_t_tab = psi_t(ts_tab, t_int_min,SFR = key_SFR) # SFR [Msun Mpc^-3 Gyr^-1]  
        psi_t_tab_MW = psi_t_tab / rho_MW # SFR per Milky Way equivalent galaxy [Msun/Gyr/MWEG]

        R_coll_z0 = psi_t_tab[-1] # local volumetric collapsar rate, set to match SFR -- infer this in combination with m_r_coll

        R_NSNS_z0 = rate*1e9 / (1e3)**3  #1540. *1e9 / (1e3)**3 # LIGO rate of NSNS mergers (1540 Gpc^-3 yr^-1) in Mpc^-3 Gyr^-1
        m_Eu_NS = m_r_NS * Xsun_Eu_r69

        norms_DTD_NS = []
        for nb_NS, b_NS_ in enumerate(b_NS):
            norms_DTD_NS.append(R_NSNS_z0 / int_NS(tmax,ts,tmin_NS,1.0,b_NS_,psi_t_tab,tmin_intMW,v_mean,R_enc,key_PDF_vkick,cutoff=False))

        DIa = D_t(tmax-ts,1.0,b_Ia,tmin_Ia)
        norm_DTD_Ia = C_Ia / integrate(DIa, ts, tmin_intMW, tmax, method = 'auto')
        DIacheck = D_t(tmax-ts,norm_DTD_Ia,b_Ia,tmin_Ia)

        norm_cc = R_cc_z0 / psi_t_tab[-1]
        norm_MHDSN = R_MHDSN_z0 / psi_t_tab[-1]
        norm_coll = R_coll_z0 / psi_t_tab[-1] # collapsars enters into abundance calculation via m_Eu_coll*norm_coll

        m_Eu_coll = m_Eu_NS*R_NSNS_z0/((1./Xcoll - 1.)*R_coll_z0) # collapsar Eu yield
        #m_r_coll = m_Eu_coll/Xsun_Eu_r69 # collapsar r-process -- infer this in combination with R_coll_z0

        Rates_NS = []
        Rates_NS_r = []
        Ns_NS = []

        Rate_CC = norm_cc*psi_z(zs_ts, SFR=key_SFR)
        Rate_MHDSN = norm_MHDSN*psi_z(zs_ts, SFR=key_SFR)

        NS_cutoff=False
        for nb_NS, b_NS_ in enumerate(b_NS):
            Rates_NS.append(array([int_NS(t,ts,tmin_NS,norms_DTD_NS[nb_NS],b_NS_,psi_t_tab,tmin_intMW,v_mean,R_enc,key_PDF_vkick,cutoff=False) for t in ts]) )
            Rates_NS_r.append(array([int_NS(t,ts,tmin_NS,norms_DTD_NS[nb_NS],b_NS_,psi_t_tab,tmin_intMW,v_mean,R_enc,key_PDF_vkick,cutoff=NS_cutoff) for t in ts]) )

        Rate_Ia = array([int_Ia(t,ts,tmin_Ia,norm_DTD_Ia,b_Ia,psi_t_tab,tmin_intMW) for t in ts])

        N_CC = array([integrate(Rate_CC, ts, tmin_intMW, t, method = 'auto') for t in ts])
        N_MHDSN = array([integrate(Rate_MHDSN, ts, tmin_intMW, t, method = 'auto') for t in ts])
        for nb_NS, b_NS_ in enumerate(b_NS):
            N_NS = array([integrate(Rates_NS[nb_NS], ts, tmin_intMW, t, method = 'auto') for t in ts])
            Ns_NS.append(N_NS)

        N_Ia = array([integrate(Rate_Ia, ts, tmin_intMW, t, method = 'auto') for t in ts])

        Rate_CC_av = N_CC[-1]/(tmax-tmin_intMW)
        Rate_MHDSN_av = N_MHDSN[-1]/(tmax-tmin_intMW)
        Rate_NS_av = N_NS[-1]/(tmax-tmin_intMW)

        nnss = len(b_NS)
        arr_sols_alpha = zeros((nnss,len(ts)))
        arr_sols_r = zeros((nnss,len(ts)))
        arr_sols_Fe = zeros((nnss,len(ts)))
        arr_sols_H = zeros((nnss,len(ts)))
        arr_sols_fZs = zeros((nnss,len(ts)))

        NS_only=False # turn collapsars on
        GRB_cutoff=False
        GRB_FeH_thr=-0.312,0.058
        add_MHD_SNe=False
        for nb_NS, b_NS_ in enumerate(b_NS):  
            sola, solr, solFe, solH, f_Z = integrate_chemical_evolution(ts,tmin_NS,[tmin_Ia],m_alpha,m_Eu_NS,m_Eu_coll,m_Eu_MHDSN,m_Fe_cc,m_Fe_Ia,m_Fe_coll,X_H,
               norm_cc,norm_MHDSN,norm_coll,[norm_DTD_Ia],norms_DTD_NS[nb_NS],f_NSgal,b_Ia,b_NS_,eta,psi_t_tab_MW,tmin_intMW,NS_only,v_mean,R_enc,key_PDF_vkick,NS_cutoff,GRB_cutoff,GRB_FeH_thr,m_Fe_u,t_sun,add_MHD_SNe)

            arr_sols_alpha[nb_NS,:] = sola
            arr_sols_r[nb_NS,:] = solr[0]
            arr_sols_Fe[nb_NS,:] = solFe[0]
            arr_sols_H[nb_NS,:] = solH
            arr_sols_fZs[nb_NS,:] = f_Z[0]

        for nb_NS, b_NS_ in enumerate(b_NS):
          NFe_NH = (arr_sols_Fe[nb_NS][1:])/arr_sols_H[nb_NS][1:] * (1./m_Fe_u)
          NFe_NH_sun = NFe_NH[ts[1:] >= t_sun][0]

          Nr_NFe = (arr_sols_r[nb_NS][1:]/arr_sols_Fe[nb_NS][1:]) * (m_Fe_u / m_Eu_u)
          Nr_NFe_sun = Nr_NFe[ts[1:] >= t_sun][0]

          if (normalize_to_observed_solar_values):
            Fe_H = log10(NFe_NH) - logNFe_NH_sun
            r_Fe = log10(Nr_NFe) - logNEu_NFe_sun
          else:
            Fe_H = log10(NFe_NH/NFe_NH_sun)
            r_Fe = log10(Nr_NFe/Nr_NFe_sun)    

        Fe_H_list.append(Fe_H)
        r_Fe_list.append(r_Fe)
        bns_dat[nsamps*i+j,:] = array([mej,rate,-alpha,tmin_NS,f_NSgal,Xcoll])

    
### SAVE RESULTS
    
    
# save r-process abundance data and parameters for population realizations

outdat = {}
outdat['pop'] = {}
outdat['yield'] = {}

for i in range(nsamps*ndtd):

    outdat['pop'][i] = bns_dat[i]
    outdat['yield'][i] = {}
    outdat['yield'][i]['Fe_H'] = Fe_H_list[i]
    outdat['yield'][i]['Eu_Fe'] = r_Fe_list[i]
    
outfile = h5py.File(OUTPATH, 'w')

pop_set = outfile.create_group('pop')
for key, value in outdat['pop'].items():
    pop_data = np.array(value)
    pop_data = rfn.unstructured_to_structured(pop_data,
    np.dtype([('m_ej', 'f8'), ('rate', 'f8'), ('b', 'f8'), ('tmin', 'f8'), ('f', 'f8'), ('X_coll', 'f8')]))
    pop_set.create_dataset(str(key),data=pop_data)

yield_set = outfile.create_group('yield')
for key, value in outdat['yield'].items():
    yield_data = np.column_stack((value['Fe_H'],value['Eu_Fe']))
    yield_data = rfn.unstructured_to_structured(yield_data,
    np.dtype([('Fe_H', 'f8'), ('Eu_Fe', 'f8')]))
    yield_set.create_dataset(str(key),data=yield_data)

outfile.close()
