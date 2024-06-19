import matplotlib
import matplotlib.pyplot as plt
from pylab import *
from numpy import *
from scipy.optimize import fsolve as fsolve
from scipy.integrate import simps
from astropy import constants as consts
from astropy import units as units

# some global constants

# Note: time is in Gyr for all purposes
pc = consts.pc.si.value # 3.085677581491367e+16 m, parsec in meters
km_Mpc = 1e3 / (pc*1e6)
Gyr = (1.0e9*units.yr).cgs.value # Gyr in seconds
Gyr2s = Gyr
m_u = consts.u.cgs.value # atomic mass unit in cgs
M_sun = consts.M_sun.cgs.value # solar mass in cgs
m_Eu_u = 151.96 # mean atomic weight of Eu in m_u
m_Fe_u = 55.845 # mean atomic weight of Fe in m_u
m_Mg_u = 24.305 # mean atomic weight of Mg in m_u
logNFe_NH_sun = 7.5 - 12. # solar log10(N_Fe/N_H) from Asplund et al. 2009 (for comparison with chemical evolution model)
logNEu_NH_sun = 0.52 - 12. # solar log10(N_Eu/N_H) from Asplund et al. 2009 (for comparison with chemical evolution model)
logNEu_NFe_sun = logNEu_NH_sun - logNFe_NH_sun


# cosmic concordance cosmology
# Madau & Dickinson, Annu. Rev. Astron. Astrophys. 2014, 52:415, p.4
# Madau & Fragos 2017 ApJ 840, 39
Omega_M = 0.3
Omega_Lambda = 0.7
Omega_b = 0.046
h0 = 0.7
H_0 = h0*100. * km_Mpc * Gyr # Hubble constant in 1/Gyr # ~ h_0*100 km/s/Mpc

zmax_int = 10. # maximum redshift for integration
age_sun = 4.568  # age of sun in Gyr
t_MW = 10.  # age of the Milky Way in Gyr
rho_MW = 0.01  # density of MW like galaxies in Mpc^-3 (using fiducial 0.01 Mpc^-3 as in Abadie+ 2010) 
Mstars_MW = 6.4e10  # stellar mass of Milky Way (McMillan 2011)


def std_wpanel_llspace():
  fig_width = 6.0
  fig_height = 0.6*fig_width #/ 1.2 #1.618
  params = {
    'figure.figsize' : [fig_width,fig_height],
    'figure.subplot.left' : 0.145,
    'figure.subplot.right' : 0.99, 
    'figure.subplot.top' : 0.96,
    'figure.subplot.bottom' : 0.15,
    'lines.markersize': 6,
    'axes.labelsize': 15,
    'text.usetex':True,
    'font.size': 14,
    'legend.fontsize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'axes.formatter.limits': [-2,3],
    'savefig.dpi': 320,
    'lines.antialiased': True
  }
  matplotlib.rcParams.update(params)
#

def std_wpanel():
  fig_width = 6.0
  fig_height = 0.6*fig_width #/ 1.2 #1.618
  params = {
    'figure.figsize' : [fig_width,fig_height],
    'figure.subplot.left' : 0.105,
    'figure.subplot.right' : 0.99,
    'figure.subplot.top' : 0.96,
    'figure.subplot.bottom' : 0.15,
    'lines.markersize': 6,
    'axes.labelsize': 15,
    'text.usetex':True,
    'font.size': 14,
    'legend.fontsize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'axes.formatter.limits': [-2,3],
    'savefig.dpi': 320,
    'lines.antialiased': True
  }
  matplotlib.rcParams.update(params)
#

def Nature_1col_ar06():
  fig_width = 3.5 # 89mm one-column = 3.504inches 
  fig_height = 0.6*fig_width #/ 1.2 #1.618
  params = {
    'figure.figsize' : [fig_width,fig_height],
    'figure.subplot.left' : 0.135,
    'figure.subplot.right' : 0.99, 
    'figure.subplot.top' : 0.97,
    'figure.subplot.bottom' : 0.205,
    'lines.markersize': 4,
    'axes.labelsize': 11,
    'font.size': 9,
    'legend.fontsize': 9,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.formatter.limits': [-2,3],
    'savefig.dpi': 320,
    'lines.antialiased': True,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'text.usetex': False
  }
  matplotlib.rcParams.update(params)
#

def Nature_1col_ar06_lspace():
  fig_width = 3.5 # 89mm one-column = 3.504inches 
  fig_height = 0.6*fig_width #/ 1.2 #1.618
  params = {
    'figure.figsize' : [fig_width,fig_height],
    'figure.subplot.left' : 0.17,
    'figure.subplot.right' : 0.99, 
    'figure.subplot.top' : 0.97,
    'figure.subplot.bottom' : 0.205,
    'lines.markersize': 4,
    'axes.labelsize': 11,
    'font.size': 9,
    'legend.fontsize': 9,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.formatter.limits': [-2,3],
    'savefig.dpi': 320,
    'lines.antialiased': True,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'text.usetex': False
  }
  matplotlib.rcParams.update(params)
#

def std_wpanel_lspace_Nature15():
  fig_width = 5.12 # Nature 1.5 column ~130mm = 5.12 inch 
  fig_height = 0.6*fig_width #/ 1.2 #1.618
  params = {
    'figure.figsize' : [fig_width,fig_height],
    'figure.subplot.left' : 0.13,
    'figure.subplot.right' : 0.99, 
    'figure.subplot.top' : 0.97,
    'figure.subplot.bottom' : 0.16,
    'lines.markersize': 4,
    'axes.labelsize': 13,
    'font.size': 10,
    'legend.fontsize': 11,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'axes.formatter.limits': [-2,3],
    'savefig.dpi': 320,
    'lines.antialiased': True,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'text.usetex': False
  }
  matplotlib.rcParams.update(params)
#

def multi_savefig(name, formats, fig=None):
  print( name, formats )
  for fmt in formats.split(','):
    fname = "%s.%s" % (name, fmt)
    print( fname )
    if (fig == None):
      plt.savefig(fname,dpi=300)
    else:
      fig.savefig(fname,dpi=300)
    #
  #
#

def interp_lin(q,x,x0):
  xlower = x[x < x0]
  xhigher = x[x > x0]
  q_lower = q[x < x0]
  q_higher = q[x > x0]

  if (len(xlower) == 0): 
    q_int = q[0]
  elif (len(xhigher) == 0):
    q_int = q[-1]
  else:
    q_l = q_lower[-1]
    q_r = q_higher[0]
    x_l = xlower[-1]
    x_r = xhigher[0]

    q_int = ((q_r - q_l)/(x_r - x_l))*(x0 - x_l) + q_l
  #
  return q_int
#

def integrate(q, radii, radii_min, radii_max, method = 'trapezoidal'):
  
  if ((radii_max <= radii_min) or (radii_min >= radii_max)):
    integral = 0.
  else:
    radii_int = radii[(radii <= radii_max) & (radii >= radii_min)]
    q_int = q[(radii <= radii_max) & (radii >= radii_min)]
    del_r_int = radii_int[1:]-radii_int[0:-1]

    if (method == 'left'):
      # manual rectangular integration taking q_int at the left boundary of the interval
      integral = sum(q_int[0:-1]*del_r_int)
    elif (method == 'right'):
      # manual rectangular integration taking q_int at the right end of the interval
      integral = sum(q_int[1:]*del_r_int)
    elif (method == 'trapezoidal'):
      integral = sum(0.5 * (q_int[0:-1] + q_int[1:]) * del_r_int)
    elif (method == 'auto'):
      # more precise integration
      integral = simps(q_int,radii_int)
    #
  #
  return integral
#

def compute_eta_smooth(x, xl, xr):
  # this is a smoothing function used to superimpose two terms (smoothly switch between two regimes) in the region [xl,xr]
  eta = cos( (2.*pi / (4.*abs(xr - xl))) * (x - xl) )**2
  eta[x <= xl] = 1.
  eta[x >= xr] = 0.
 
  return eta
#

def psi_z(z, SFR = 'MF17'):
  # returns cosmic star formation rate as a function of redshift z, in Msun/Gyr/Mpc^3,
  
  if (SFR == 'MD14'):
    # best-fitting function form Madau & Dickinson, Annu. Rev. Astron. Astrophys. 2014, 52:415, Eq.(15)
    psi = 1e9 * 0.015 * (1. + z)**(2.7) / ( 1. + ((1.+z)/2.9)**5.6)

  elif (SFR == 'MF17'):  
    # best-fitting function form Madau & Dickinson, Annu. Rev. Astron. Astrophys. 2014, 52:415, Eq.(15)
    # but updated coefficients from recent measurements at z>4, see Madau & Fragos 2017 ApJ 840, 39
    # and Maoz & Graur 2017 ApJ 848 25
    psi = 1e9 * 0.01 * (1. + z)**2.6 / ( 1. + ((1.+z)/3.2)**6.2)
  elif (SFR == 'const'):
    #psi = 1e9 * 0.01 * ones_like(z) / ( 1. + (1./3.2)**6.2)
    psi = 1e9 * 0.01 * ones_like(z)
  else:
    exit('No valid SFR specified! Aborting.')
  #

  return psi
#

def psi_t(t,tint_min, SFR = 'cosmic'):
  z = z_t(t,tint_min)
  return psi_z(z, SFR=SFR)
#

def t_z(z):
  # returns cosmic time (time unit is that of H_0^-1) as a function of redshift for a flat cosmology
  # dt = dz / (H(z) * (1+z)), where H(z) = H_0 * sqrt(Omega_M(1+z)**3 + Omega_Lambda)
  theta = arctan(sqrt(Omega_M/Omega_Lambda)*(1.+z)**1.5)

  return (2. / (3.*H_0*sqrt(Omega_Lambda))) * log((1.+ cos(theta))/sin(theta))
#

def t_root(z,t):
  return t - t_z(z)
#  

def z_t(t,tint_min):
  # return redshift as a function of cosmic time

  # assuming flat cosmology, use rough redshift grid to identify initial guess for t
  zs = linspace(0.,zmax_int,int(1e5))
  ts = t_z(zs)
  
  if (size(t) > 1):
    z_t = zeros_like(t)
    for ntloc,tloc in enumerate(t):
      # initial guess
      if (tloc > tint_min):
        if (size(zs[ts <= tloc]) > 1):
          z_guess = zs[ts <= tloc][0]
        else:
          z_guess = ts[0]
        #    
        # find t via root finding
        sol = fsolve(t_root,z_guess,args=tloc,xtol=1e-14)
        # enforce z >= 0
        z_t[ntloc] = max(sol[0],0.0)
      else:
        z_t[ntloc] = zmax_int
      #
    #
  else:
    # initial guess
    z_guess = zs[ts <= t][0]
  
    # find t via root finding
    sol = fsolve(t_root,z_guess,args=t,xtol=1e-14)

    # enforce z >= 0
    z_t = max(sol[0],0.0)
  #

  return z_t  
#  

def M_ISM_t(t,ttab,Psitab):

  # total mass of ISM, use Kennicutt-Schmidt relation
  # assume M_ISM(z=0) = 1e10 Msun
  # M_ISM ~ Psi^(1./1.4)
  # M_ISM(t) = ( Psi(t)^(1./1.4) / Psi(t=t(z=0))^(1./1.4) ) * M_ISM(z=0)
  M_ISM_z0 = 1e10 #0.1*Mstars_MW  #1e10 # in Msun
  Psi0 = Psitab[-1] #psi_z(0.)
  psit = interp(t,ttab,Psitab,right=0.,left=0.)
  return M_ISM_z0 * (psit / Psi0)**(1./1.4)
#

def M_ISM(Psitab):

  # total mass of ISM, use Kennicutt-Schmidt relation
  # assume M_ISM(z=0) = 1e10 Msun
  # M_ISM ~ Psi^(1./1.4)
  # M_ISM(t) = ( Psi(t)^(1./1.4) / Psi(t=t(z=0))^(1./1.4) ) * M_ISM(z=0)
  M_ISM_z0 = 1e10 #0.1*Mstars_MW #1e10 # in Msun
  Psi0 = Psitab[-1] #psi_z(0.)
  return M_ISM_z0 * (Psitab / Psi0)**(1./1.4)
#

def f_t(t,eta,ttab,Psitab):
  psit = interp(t,ttab,Psitab,right=0.,left=0.)
  M_ISM = M_ISM_t(t,ttab,Psitab)
  #M_ISM_z0 = 1e10 # in Msun
  #Psi0 = Psitab[-1] #psi_z(0.)
  #result = (1. + eta) * psit**(0.2/0.7) * Psi0**(1./1.4) / (M_ISM_z0)
  result = (1. + eta) * psit / M_ISM

  #if ((psit > 0.) & (M_ISM > 0.)):
  #  result = (1. + eta) * psit / M_ISM
  #else:
  #  result = 0.
  #
  return result
#

def D_t_NS(t,C,b,tmin,vmean,Renc,key_PDF,cutoff=False):
  if (cutoff):
    #result = C * heaviside(t-tmin,1.) * heaviside(cutoff-t,1.)/ t**b
    result = C * heaviside(t-tmin,1.) * P_r_gtr_Renc(t,vmean,Renc,key_PDF) / t**b
  else:
    result = C * heaviside(t-tmin,1.) / t**b
  #
  result[t==0.] = 0.
  
  return result
#

def D_t(t,C,b,tmin):
  result = C * heaviside(t-tmin,1.) / t**b
  result[t==0.] = 0.
  return result
#

def D_t_test(t,C,b,tmin):
  
  result = C / t**b
  result[t-tmin <= 0.] = 0.
  
  return result
#

def int_Ia(t,ts,tmin,CIa,bIa,psit,tmin_int):
  DIa = D_t(t-ts,CIa,bIa,tmin)
  return integrate(DIa*psit, ts, tmin_int, t, method = 'auto')
#

def int_NS(t,ts,tmin,CNS,bNS,psit,tmin_int,vmean,Renc,key_PDF,cutoff=False):
  DNS = D_t_NS(t-ts,CNS,bNS,tmin,vmean,Renc,key_PDF,cutoff=cutoff)
  return integrate(DNS*psit, ts, tmin_int, t, method = 'auto')
#

def F_alpha_RHS(y,t,eta,malpha,Ccc,ttab,Psitab):
  psit = interp(t,ttab,Psitab,right=0.,left=0.)
  return malpha*Ccc*psit - y * f_t(t,eta,ttab,Psitab)
#  

def F_Fe_RHS(y,t,ts,eta,tmin,mFecc,Ccc,mFeIa,CIa,bIa,mFecoll,Ccoll,Psitab,tmin_int,fZ):
  psit = interp(t,ts,Psitab,right=0.,left=0.)
  #psit = Psitab
  return mFecc*Ccc*psit + mFecoll*Ccoll*fZ*psit + mFeIa*int_Ia(t,ts,tmin,CIa,bIa,Psitab,tmin_int) - y*f_t(t,eta,ts,Psitab)
#

def F_r_RHS(y,t,ts,eta,tmin,mr_NS,mr_coll,mr_MHDSN,CNS,Ccoll,CMHDSN,bns,Psitab,tmin_int,key_NS_only,vmean,Renc,key_PDF,rNS_cutoff,fZ,addMHDSNe):
  if (key_NS_only):
    result = mr_NS*int_NS(t,ts,tmin,CNS,bns,Psitab,tmin_int,vmean,Renc,key_PDF,cutoff=rNS_cutoff) - y*f_t(t,eta,ts,Psitab)
  else:
    psit = interp(t,ts,Psitab,right=0.,left=0.)
    if (addMHDSNe):
      result = mr_coll*Ccoll*fZ*psit + mr_MHDSN*CMHDSN*psit + mr_NS*int_NS(t,ts,tmin,CNS,bns,Psitab,tmin_int,vmean,Renc,key_PDF,cutoff=rNS_cutoff) - y*f_t(t,eta,ts,Psitab)
    else:  
      result = mr_coll*Ccoll*fZ*psit + mr_NS*int_NS(t,ts,tmin,CNS,bns,Psitab,tmin_int,vmean,Renc,key_PDF,cutoff=rNS_cutoff) - y*f_t(t,eta,ts,Psitab)
    #
  #
  return result
#

# def dF_alpha_RHS(Malpha,t,eta,ttab,Psitab):
#   return -f_t(t,eta,ttab,Psitab)
# #

# def dF_Fe_RHS(MFe,t,eta,tmin,ttab,Psitab):
#   return -f_t(t,eta,ttab,Psitab)
# #

# def dF_r_RHS(Mr,t,eta,tmin,ttab,Psitab):
#   return -f_t(t,eta,ttab,Psitab)
# #  

def integrate_chemical_evolution(times,tminNS,tminsIa,malpha,mEu_NS,mEu_coll,mEu_MHDSN,mFe_cc,mFe_Ia,mFe_coll,XH,
       normcc,normMHDSN,normcoll,normsDTD_Ia,normDTD_NS,fNSgal,bIa,bNS,etafac,psi_t_tabMW,tmin_intMW,key_NS_only,vmean,Renc,key_PDF,NScutoff,GRBcutoff,GRB_FeHthr,mFe_u,tsun,add_MHDSNe):

  sols_Fe = []
  sol_alpha = zeros_like(times)
  sols_r = []

  # time steps delta_t
  dts = times[1:] - times[:-1]

  # Hydrogen abundances
  #sol_H = (rho_MW/0.01) * XH*M_ISM(psi_t_tabMW)
  sol_H = XH*M_ISM(psi_t_tabMW)

  # used to implement GRB-collapsar cutoff
  f_Zs = []
  
  #print( 'Integrating Fe and r-process...' )
  #print( 'tmin_NS [Gyr] =', tminNS )
  for ntmin,tmin_Ia in enumerate(tminsIa):
    #print( 'tmin_Ia [Gyr] =', tmin_Ia )
    
    sol_Fe = zeros_like(times)

    MFe = 0.
    for nt,t in enumerate(times[1:]):
      fZloc = 1.
      #!!!!!!!!!!!!!!!!!!!!
      ## need to fix FZloc here if collapsar Fe co-production is taken into account
      ## currently this is not fully self-consistent (a minor effect, however)
      #!!!!!!!!!!!!!!!!!!!!
      dMFe = dts[nt] * F_Fe_RHS(MFe,t,times,etafac,tmin_Ia,mFe_cc,normcc,mFe_Ia,normsDTD_Ia[ntmin],bIa,mFe_coll,normcoll,psi_t_tabMW,tmin_intMW,fZloc)
      MFe += dMFe
      sol_Fe[nt+1] = max(MFe, 1e-50)
    #
    sols_Fe.append(sol_Fe)

    # set GRB-collapsar cutoff
    if (GRBcutoff):
      # compute [Fe/H] as a function of cosmic time t
      NFe_NH = sol_Fe/sol_H * (1./mFe_u)
      NFe_NH_sun = NFe_NH[times >= tsun][0]
      Fe_H = log10(NFe_NH/NFe_NH_sun)
      Fe_H[0] = -50.
      # set GRB cut-off profile 
      f_Z = compute_eta_smooth(Fe_H, GRB_FeHthr[0], GRB_FeHthr[1])
    else:
      f_Z = ones_like(times)
    #
    f_Zs.append(f_Z)

    # integrate r-process
    sol_r = zeros_like(times)
    Mr = 0.
    for nt,t in enumerate(times[1:]):
      dMr = dts[nt] * F_r_RHS(Mr,t,times,etafac,tminNS,mEu_NS,mEu_coll,mEu_MHDSN,normDTD_NS*fNSgal,normcoll,normMHDSN,bNS,psi_t_tabMW,tmin_intMW,key_NS_only,vmean,Renc,key_PDF,NScutoff,f_Z[1:][nt],add_MHDSNe)
      Mr += dMr
      sol_r[nt+1] = max(Mr, 1e-50)
    #
    sols_r.append(sol_r)
  #


  #print( 'Integrating alpha...' )
  Ma = 0.
  for nt,t in enumerate(times[1:]):
    dMa = dts[nt] * F_alpha_RHS(Ma,t,etafac,malpha,normcc,times,psi_t_tabMW)
    #dMr = dts[nt] * F_r_RHS(Mr,t,times,etafac,tminNS,mEu_NS,mEu_coll,normDTD_NS*fNSgal,normcoll,bNS,psi_t_tabMW,key_NS_only,vmean,Renc,key_PDF,NScutoff)
    Ma += dMa
    #Mr += dMr
    sol_alpha[nt+1] = max(Ma, 1e-50)
    #sol_r[nt+1] = max(Mr, 0.)
  #

  return sol_alpha, sols_r, sols_Fe, sol_H, f_Zs
#

class nf2(float):
  def __repr__(self):
    def format_e(f):
      a = '%1.1e' % f
      p1 = a.split('e')[0]
      p2 = a.split('e')[-1]
      p2s = a.split('e')[-1][0]
      p2e = a.split('e')[-1][1:]
      if (p1[-1] == '0'):
        p1 = p1.rstrip('0').rstrip('.')
      #
      if (p2s == '+'):
        p2s = ''
      #
      if (p2e[0] == '0'):
        p2e = p2e.lstrip('0')
      # 
      return p1 + 'e' + p2s + p2e
    #
    str = '%.1f' % (self.__float__(),)
    if str[-1] == '0':
      return format_e(self.__float__())
      #return '%1.1e' % self.__float__()
    else:
      return '%.1f' % self.__float__()          
    #
#
  
# def PDF_v_kick(v,v_mean,key_PDF = 'exp'):
#   if (key_PDF == 'exp'):
#     PDF = exp(-v/v_mean)
#   else:
#     exit('PDF not known')
#   #
#   return PDF
# #

def P_vkick_0_vcrit(vcrit,vmean,key_PDF = 'exp'):
  # returns P(v<vcrit), probability for v < vcrit
  if (key_PDF == 'exp'):
    # underlying PDF = (1/v_mean)*exp(-v/v_mean)
    P = 1.-exp(-vcrit/vmean)
  elif (key_PDF == 'delta'):
    # underlying PDF = delta(v-v_mean), where delta is the delta function
    P = heaviside(vcrit-vmean,1.0)
  else:
    exit('PDF not known')
  #
  return P
#

def P_r_gtr_Renc(ts,vmean,Renc,key_PDF):
  # returns P(r(t) < R_enc) = P(v < vcrit), where vcrit = R_enc/t
  vs_crit = Renc / ts / Gyr2s
  vs_crit[ts <= 0.] = 0.
  result = P_vkick_0_vcrit(vs_crit,vmean,key_PDF=key_PDF)
  return result
  
#
