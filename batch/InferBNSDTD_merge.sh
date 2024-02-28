#!/bin/bash

# Merge csv parts that result from InferBNSDTD batch job into one file

cp Battistini16_disk_50000.part0.csv Battistini16_disk_5M.csv

for i in {1..99}; do awk '(FNR > 1)' Battistini16_disk_50000.part${i}.csv >> Battistini16_disk_5M.csv; done
