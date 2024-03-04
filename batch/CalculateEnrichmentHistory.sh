#!/bin/bash

python3 CalculateEnrichmentHistory.py "$1" "$2" "$3" "$4" --alpha="-3.,-0.5" --tmin="1e-2,2.01" --xsfh="1e-3,0.999" --nmarg=500
