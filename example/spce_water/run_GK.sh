#!/bin/bash
for i in {1..5}
do
cd run$i
python3 ../../../src/gk_viscosity_fft.py > viscosity_fft.log
wait
cd -
done
python3 ../../src/viscosity_avg.py > viscosity_avg.log

