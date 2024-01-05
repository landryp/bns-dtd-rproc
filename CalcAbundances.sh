#!/bin/bash

python3 CalcAbundances.py "$1" "$2" "$3" "$4" --alpha="-3.,-0.5" --tmin="1e-2,2.01" --xcoll="0.,0." --marg=500

#python3 CalcAbundances.py "$1" "$2" "$3" "$4" --alpha="-3.,-0.5" --tmin="1e-2,2.01" --xcoll="1e-3,0.999" --marg=500