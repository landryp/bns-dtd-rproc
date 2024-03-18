##### bns-dtd-rproc
### bayesian inference of the binary neutron star delay time distribution from galactic r-process abundance observations, and hypothesis testing on one vs two-channel models for galactic r-process enrichment history
philippe landry (pgjlandry@gmail.com) 02/2024

*where do the universe's heavy elements, like gold and platinum, originate? their large neutron-to-proton asymmetries imply that they must be synthesized through rapid neutron capture in environments where the capture rate exceeds the beta-decay rate. binary neutron star mergers are one candidate site for such r-process nucleosynthesis, because the dense matter ejected during the merger is extremely neutron-rich. the observation of absorption lines associated with r-process elements in the spectrum of the kilonova counterpart to the binary neutron star merger gw170817 support this view. but are binary neutron star mergers the sole site for r-process nucleosynthesis, or merely one site among many?*

*gravitational-wave surveys have begun to constrain the binary neutron star merger rate. numerical simulations of binary neutron star mergers inform expectations for the amount of matter ejected in each merger, and its r-process element content. starting from the galactic star formation history, and given a distribution of delay times between the birth of the binary neutron star progenitor system and its eventual merger, one can reconstruct the history of r-process enrichment events over cosmic time.*

*this repository contains python scripts for constraining a model for the binary neutron star delay time distribution based on r-process abundance measurements in galactic stellar spectra. the model can be conditioned on gravitational-wave and short gamma-ray burst osbervations, which constrain the binary neutron star merger rate and delay times, respectively. the model can also account for a second r-process nucleosynthesis channel whose rate evolution tracks the galactic star formation history. comparison of model evidences allows for hypothesis testing on the one- vs two-channel r-process nucleosynthesis scenarios.*

for more, please see Chen,Landry,Read+Siegel arXiv:2402.03696

### Notebooks

*these notebooks offer a demonstration of the analysis and showcase the key results*

BNS-DTD-from-rprocess.ipynb *# constrain binary neutron star delay times using galactic r-process abundance observations*

BNS-vs-two-channel-rprocess.ipynb *# perform hypothesis testing on one- vs two-channel galactic chemical evolution models, conditioned on galactic r-process abundance observations*

### Data

*the etc/ directory contains the data and utilities needed for the analysis*

Arnould07_solar_rprocess.dat *# solar r-process abundances from Arnould,Goriely+Takahashi PhysRep 2007* \
Battistini16_disk.csv *# r-process abundance measurements in galactic disk stars from Battistini+Bensby A&A 2016* \
mej_rate_lcehl_psr+gw+nicer.dat *# binary neutron star merger rate and ejecta samples from Abbott+ PRX 2023, conditioned on neutron star equation of state information from Legred+ PRD 2021, courtesy of HY Chen* \
mej_rate_lcehl_psr+gw.dat *# same as above, but conditioned on neutron star equation of state information from Legred+ PRD 2021, excluding x-ray pulse profile modeling by NICER, courtesy of HY Chen* \
SAGA_MP.csv *# r-process abundance measurements in galactic disk and halo stars from the SAGA database* \
Zevin22_sgrb_dtd.dat *# delay time distribution parameter samples inferred from short gamma-ray burst observations in Zevin+ ApJL 2022* \

rProcessChemicalEvolution.py *# one-zone r-process chemical evolution code, adapted from D Siegel's*
rProcessUtils.py *# utilities for one-zone r-process chemical evolution code, courtesy of D Siegel*

### Scripts

*these scripts are used for production analyses*

CalculateEnrichmentHistory.py nsamp outpath dtdpath ejpath --alpha alpha_min,alpha_max --tmin tmin_min,tmin_max --xsfh xsfh_min,xsfh_max --nmarg nmarg *# calculate Eu vs Fe abundance history from binary neutron star merger rate, ejecta, delay time distribution and fractional contribution relative to a second channel, using Daniel Siegel's one-zone r-process nucleosynthesis code*

InferBNSDTD.py outdir obspath eufepath --maxnum maxnum --parts parts *# infer binary neutron star delay time distribution parameters and fractional second-channel contribution from galactic r-process abundance observations and Eu vs Fe abundance histories*

PlotDTDConstraints.py *# plot posterior samples in binary neutron star delay time distribution parameters*

PlotSecondChannelConstraints.py *# plot posterior samples in binary neutron star rate-ejecta product and second-channel contribution fraction*

### Batch

*the scripts in batch/ allow for the submission of batch analyses via condor*

bash batch/CalculateEnrichmentHistory_DAG.sh $(cat batch/CalculateEnrichmentHistory_DAG.in) \
condor_submit_dag calculate_batch/CalculateEnrichmentHistory.dag

bash batch/InferBNSDTD_DAG.sh $(cat batch/InferBNSDTD_DAG.in) \
condor_submit_dag infer_batch/InferBNSDTD.dag \
bash InferBNSDTD_merge.sh
