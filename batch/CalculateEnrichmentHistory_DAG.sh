#!/bin/bash

# Set run parameters

nums=$1 # 100,100,100,100,100 -- list of number of alpha,tmin samples per job x number of jobs
outpathtmp=$2 # dat/EuFe -- appends .h5 and appropriate identifying tags
dtdpath=$3 # 'etc/Zevin22_sgrb_dtd.dat' -- or '', which defaults to uniform DTD parameter prior
ejpath=$4 # 'etc/mej_rate_lcehl_psr+gw+nicer.txt'

batchdir='calculate_batch' # name of output directory for log files
tag='grbbnscls' # tag for output path

# Create output files and directories

mkdir -p "$PWD/$batchdir"
dagfile="$PWD/$batchdir/CalculateEnrichmentHistory.dag"
configfile="$PWD/$batchdir/CalculateEnrichmentHistory.config"

echo "${nums}" > $configfile
echo "${outpathtmp}" >> $configfile
echo "${dtdpath}" >> $configfile
echo "${ejpath}" >> $configfile

IFS=',' read -r -a nums <<< "$nums"

# Print sub files

binfile="CalculateEnrichmentHistory.sh"
subfile="$PWD/$batchdir/${binfile}.sub"
args="arguments = \"\$(num) \$(outpath) \$(dtdpath) \$(ejpath)\""

echo "universe = vanilla" > $subfile
echo "executable = $PWD/$binfile" >> $subfile
echo $args >> $subfile
echo "output = $PWD/$batchdir/$binfile.out" >> $subfile
echo "error = $PWD/$batchdir/$binfile.err" >> $subfile
echo "log = $PWD/$batchdir/$binfile.log" >> $subfile
echo "getenv = True" >> $subfile
echo "accounting_group = ligo.dev.o4.cbc.extremematter.bilby" >> $subfile
echo "accounting_group_user = philippe.landry" >> $subfile
echo "request_disk = 256MB" >> $subfile
echo "queue 1" >> $subfile

# Print dag file

echo "### Run $binfile batch jobs ###" > $dagfile

job=0

for i in $(seq 0 $((${#nums[@]}-1)))
do
    
    outtag="${tag}"
    
    outpath="${outpathtmp}_${outtag}-${nums[$i]}.part${i}.h5"

    echo "JOB $job $subfile" >> $dagfile
    echo "VARS $job num=\"${nums[$i]}\" outpath=\"${outpath}\" dtdpath=\"${dtdpath}\" ejpath=\"${ejpath}\"" >> $dagfile
    echo "RETRY $job 1" >> $dagfile

    job=$(($job+1))

done
