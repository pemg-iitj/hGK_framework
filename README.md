# Hybrid Green-Kubo Viscosity Framework (hGK)

This repository implements a hybrid Green-Kubo framework for
efficient and convergence-controlled viscosity calculations
from equilibrium molecular dynamics simulations.

The framework combines:

• FFT-based SACF calculation
• Multi-trajectory averaging
• Window convergence diagnostics
• Tail-extrapolated viscosity estimation
-------------------------------------------------------------
## File Structure
src/
-gk_viscosity_fft.py	# FFT-based SACF computation
-viscosity_avg.py	# Multi-run averaging
-hGK_scan.py		# Convergence window scan
-hGK_final.py		# Final viscosity estimation

example/spce_water/
-run_GK.sh
-run_hGK_scan.sh
-run_hGK_final.sh
-run1–run5/
-------------------------------------------------------------
## Example Workflow

```bash
cd example/spce_water
./run_GK.sh
./run_hGK_scan.sh
./run_hGK_final.sh
-------------------------------------------------------------
If you use this framework, please cite:

Meel, A. K.; Mogurampelly, S.
"A hybrid Green-Kubo (hGK) framework for calculating 
viscosity from short MD simulations"
arXiv:2512.04546
https://doi.org/10.48550/arXiv.2512.04546
