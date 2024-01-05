# bns-dtd-proc
Code repository for inference of the binary neutron star delay time distribution from stellar r-process abundance observations.

Scripts:

CalcAbundances.py num outpath dtdpath ejpath --alpha alpha_min,alpha_max --tmin tmin_min,tmin_max --xcoll xcoll_min,xcoll_max --marg nmarg # calculate Eu vs Fe abundance history given models for binary neutron star merger ejecta, delay time distribution and fractional contribution relative to collapsars

InferDTD.py outdir eufepath poppath --nlikes 100 --maxnum 100000 --disk False --parts 10 > maxL.out # infer binary neutron star delay time distribution parameters and fractional collapsar contribution from stellar r-process abundance observations and Eu vs Fe abundance history predictions

Batch:

bash CalcAbundances_DAG.sh $(cat CalcAbundances_DAG.in)
condor_submit_dag batch/CalcAbundances.dag
