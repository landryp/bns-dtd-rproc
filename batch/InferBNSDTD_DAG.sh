#!/bin/bash

# Set run parameters

outdir=$1 # dat/SAGA_grbbns/
obspath=$2 # etc/SAGA_MP.csv
eufepathtmp=$3 # dat/EuFe_grbbns-100 -- appends appropriate part tag and .h5
parts=$4 # 100

batchdir='infer_batch' # name of output directory for log files

# Create output files and directories

mkdir -p "$PWD/$batchdir"
dagfile="$PWD/$batchdir/InferBNSDTD.dag"
configfile="$PWD/$batchdir/InferBNSDTD.config"

echo "${outdir}" > $configfile
echo "${obspath}" >> $configfile
echo "${eufepathtmp}" >> $configfile
echo "${parts}" >> $configfile

# Print sub files

binfile="InferBNSDTD.sh"
subfile="$PWD/$batchdir/${binfile}.sub"
args="arguments = \"\$(outdir) \$(obspath) \$(eufepath) \$(tag)\""

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

for i in $(seq 0 $((${parts}-1)))
do

    eufepath="${eufepathtmp}.part${i}.h5"
    tag="part${i}"

    echo "JOB $job $subfile" >> $dagfile
    echo "VARS $job outdir=\"${outdir}\" obspath=\"${obspath}\" eufepath=\"${eufepath}\" tag=\"${tag}\"" >> $dagfile
    echo "RETRY $job 1" >> $dagfile

    job=$(($job+1))

done
