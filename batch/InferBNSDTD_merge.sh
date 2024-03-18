#!/bin/bash

# Merge csv parts that result from InferBNSDTD batch job into one file

cp dat/Battistini_grbbnscls/Battistini16_disk_50000.part0.csv dat/Battistini_grbbnscls/Battistini16_disk_5M.csv

for i in {1..99}; do awk '(FNR > 1)' dat/Battistini_grbbnscls/Battistini16_disk_50000.part${i}.csv >> dat/Battistini_grbbnscls/Battistini16_disk_5M.csv; done
