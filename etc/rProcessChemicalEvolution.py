import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import multivariate_normal
import numpy.lib.recfunctions as rfn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
from tqdm import tqdm

from etc.rProcessUtils import *

key_SFR='const' # star formation rate
SOLARPATH = 'etc/Arnould07_solar_rprocess.dat'

m_Mg = 0.12 # Mg yield per enrichment event in Msun
m_alpha = m_Mg # He

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

def rproc_evolution(R_NSNS_z0,m_r_NS,b_NS,tmin_NS,Xcoll=0.,f_NSgal=0.5,nppdt=20):

	b_NS = [b_NS]

	dt_max = min(tmin_NS,tmin_Ia,tmin) / nppdt # max step size
	n_dt_max = int(round((tmax - tmin) / dt_max)) # number of grid points
	ts = linspace(tmin,tmax,n_dt_max)
	zs_ts = z_t(ts,t_int_min)

	ts_tab = ts
	psi_t_tab = psi_t(ts_tab, t_int_min,SFR = key_SFR) # SFR [Msun Mpc^-3 Gyr^-1]  
	psi_t_tab = psi_t_tab * integrate(psi_t(ts_tab, t_int_min,SFR = 'MF17'), ts_tab, t_int_min, tmax)/integrate(psi_t(ts_tab, t_int_min,SFR = key_SFR), ts_tab, t_int_min, tmax)
	psi_t_tab_MW = psi_t_tab / rho_MW # SFR per Milky Way equivalent galaxy [Msun/Gyr/MWEG]

	R_coll_z0 = psi_t_tab[-1] # local volumetric collapsar rate, set to match SFR, infer this in combination with m_r_coll

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

	if Xcoll > 0.:
		m_Eu_coll = m_Eu_NS*R_NSNS_z0/((1./Xcoll - 1.)*R_coll_z0) # collapsar Eu yield
	else: m_Eu_coll = 0.

	Rates_NS = []
	#Rates_NS_r = []
	Ns_NS = []

	Rate_CC = norm_cc*psi_z(zs_ts, SFR=key_SFR)
	Rate_MHDSN = norm_MHDSN*psi_z(zs_ts, SFR=key_SFR)
	Rate_coll = norm_coll*psi_z(zs_ts, SFR=key_SFR)

	NS_cutoff=False
	for nb_NS, b_NS_ in enumerate(b_NS):
		Rates_NS.append(array([int_NS(t,ts,tmin_NS,norms_DTD_NS[nb_NS],b_NS_,psi_t_tab,tmin_intMW,v_mean,R_enc,key_PDF_vkick,cutoff=False) for t in ts]) )
		#Rates_NS_r.append(array([int_NS(t,ts,tmin_NS,norms_DTD_NS[nb_NS],b_NS_,psi_t_tab,tmin_intMW,v_mean,R_enc,key_PDF_vkick,cutoff=NS_cutoff) for t in ts]) )

	Rate_Ia = array([int_Ia(t,ts,tmin_Ia,norm_DTD_Ia,b_Ia,psi_t_tab,tmin_intMW) for t in ts])

	N_CC = array([integrate(Rate_CC, ts, tmin_intMW, t, method = 'auto') for t in ts])
	N_MHDSN = array([integrate(Rate_MHDSN, ts, tmin_intMW, t, method = 'auto') for t in ts])
	N_coll = array([integrate(Rate_coll, ts, tmin_intMW, t, method = 'auto') for t in ts])
	for nb_NS, b_NS_ in enumerate(b_NS):
		N_NS = array([integrate(Rates_NS[nb_NS], ts, tmin_intMW, t, method = 'auto') for t in ts])
		Ns_NS.append(N_NS)

	N_Ia = array([integrate(Rate_Ia, ts, tmin_intMW, t, method = 'auto') for t in ts])

	print(N_Ia, N_CC)

	Rate_CC_av = N_CC[-1]/(tmax-tmin_intMW)
	Rate_MHDSN_av = N_MHDSN[-1]/(tmax-tmin_intMW)
	Rate_NS_av = N_NS[-1]/(tmax-tmin_intMW)
	Rate_coll_av = N_coll[-1]/(tmax-tmin_intMW)

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

	Rate_NS = np.array(Rates_NS[0])
	Rate_coll = np.array(Rate_coll)
	if Xcoll > 0.: Rate_X = m_Eu_coll*Rate_coll/(m_Eu_coll*Rate_coll+Rate_NS*m_Eu_NS)
	else: Rate_X = np.full(len(Rate_NS),0.)

	return Fe_H, r_Fe, (Rate_NS, Rate_coll, Rate_X, zs_ts, ts)
