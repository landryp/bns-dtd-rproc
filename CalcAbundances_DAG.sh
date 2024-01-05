#!/bin/bash

# Set run parameters

nums=$1 # 100,100,100,100,100 -- list of number of alpha,tmin samples per job x number of jobs
outpathtmp=$2 # dat/EuFe -- appends .h5 and appropriate identifying tags
dtdpaths=$3 # 'etc/label_samples.dat' -- or '', which defaults to uniform DTD parameter prior
ejpath=$4 # 'etc/mej_gal_lcehl_nicer_numuncertainty.txt'

# Create output files and directories

mkdir -p "$PWD/batch"
dagfile="$PWD/batch/CalcAbundances.dag"
configfile="$PWD/batch/CalcAbundances.config"

echo "${nums}" > $configfile
echo "${outpathtmp}" >> $configfile
echo "${dtdpaths}" >> $configfile
echo "${ejpath}" >> $configfile

IFS=',' read -r -a nums <<< "$nums"
IFS=',' read -r -a dtdpaths <<< "$dtdpaths"

# Print sub files

binfile="CalcAbundances.sh"
subfile="$PWD/batch/${binfile}.sub"
args="arguments = \"\$(num) \$(outpath) \$(dtdpath) \$(ejpath)\""

echo "universe = vanilla" > $subfile
echo "executable = $PWD/$binfile" >> $subfile
echo $args >> $subfile
echo "output = $PWD/batch/$binfile.out" >> $subfile
echo "error = $PWD/batch/$binfile.err" >> $subfile
echo "log = $PWD/batch/$binfile.log" >> $subfile
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
    for j in $(seq 0 $((${#dtdpaths[@]}-1)))
    do
    
        if [[ $j == 1 ]]; then
            outtag="bns"
        else
            outtag="grbbns"
        fi
        
        outpath="${outpathtmp}_${outtag}-${nums[$i]}.part${i}.h5"
    
        echo "JOB $job $subfile" >> $dagfile
        echo "VARS $job num=\"${nums[$i]}\" outpath=\"${outpath}\" dtdpath=\"${dtdpaths[$j]}\" ejpath=\"${ejpath}\"" >> $dagfile
        echo "RETRY $job 1" >> $dagfile

        job=$(($job+1))

    done
done
