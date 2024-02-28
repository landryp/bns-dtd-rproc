##### bns-dtd-rproc
### *bayesian inference of the binary neutron star delay time distribution from galactic r-process abundance observations, and hypothesis testing on one vs two-channel models for galactic r-process enrichment history*

where do the universe's heavy elements, like gold and platinum, originate? their large neutron-to-proton asymmetries imply that they must be synthesized through rapid neutron capture in environments where the capture rate exceeds the beta-decay rate. binary neutron star mergers are one candidate site for such r-process nucleosynthesis, because the dense matter ejected during the merger is extremely neutron-rich. the observation of absorption lines associated with r-process elements in the spectrum of the kilonova counterpart to the binary neutron star merger gw170817 support this view. but are binary neutron star mergers the sole site for r-process nucleosynthesis, or merely one site among many?

gravitational-wave surveys have begun to constrain the binary neutron star merger rate. numerical simulations of binary neutron star mergers inform expectations for the amount of matter ejected in each merger, and its r-process element content. starting from the galactic star formation history, and given a distribution of delay times between the birth of the binary neutron star progenitor system and its eventual merger, one can reconstruct the history of r-process enrichment events over cosmic time.

this repository contains python scripts for constraining a model for the binary neutron star delay time distribution based on r-process abundance measurements in galactic stellar spectra. the model can be conditioned on gravitational-wave and short gamma-ray burst osbervations, which constrain the binary neutron star merger rate and delay times, respectively. the model can also account for a second r-process nucleosynthesis channel whose rate evolution tracks the galactic star formation history. comparison of model evidences allows for hypothesis testing on the one- vs two-channel r-process nucleosynthesis scenarios.

##### philippe landry (pgjlandry@gmail.com) 02/2024

### Notebooks ###

BNS-DTD-from-rprocess.ipynb

BNS-vs-two-channel-rprocess.ipynb

### Scripts ###

CalcAbundances.py num outpath dtdpath ejpath --alpha alpha_min,alpha_max --tmin tmin_min,tmin_max --xcoll xcoll_min,xcoll_max --marg nmarg # calculate Eu vs Fe abundance history given models for binary neutron star merger ejecta, delay time distribution and fractional contribution relative to collapsars

InferDTD.py outdir eufepath poppath --maxnum 100000 --disk False --parts 10 > maxL.out # infer binary neutron star delay time distribution parameters and fractional collapsar contribution from stellar r-process abundance observations and Eu vs Fe abundance history predictions

PlotPost.py > stats.txt

### Batch ###

bash CalcAbundances_DAG.sh $(cat CalcAbundances_DAG.in)
condor_submit_dag batch/CalcAbundances.dag

bash InferDTD_DAG.sh $(cat InferDTD_DAG.in)
condor_submit_dag batch/InferDTD.dag

cp Battistini16_disk_50000.part0.csv Battistini16_disk_5M.csv
for i in {1..99}; do awk '(FNR > 1)' Battistini16_disk_50000.part${i}.csv >> Battistini16_disk_5M.csv; done
