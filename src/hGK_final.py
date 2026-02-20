"""
==============================================================
Final Green-Kubo Viscosity Estimator
==============================================================

Computes final viscosity using user-defined SACF fitting window
and tail extension length.

INPUT FILES
-----------
avgpoft.dat
avgetaoft.dat
in.viscosity

USER PARAMETERS
---------------
tau_low  : Fit start time (ps)
tau_up   : Fit end time (ps)
tail_cut : Time to extend SACF tail (ps)

OUTPUT FILES
------------
final_viscosity.dat
fit_tail_sacf.dat

AUTHOR
------
Akash Kumar Meel

VERSION
-------
1.0
"""

import numpy as np
from scipy.optimize import curve_fit

##############################################################
# User-defined parameters
##############################################################

base_path = "."

##############################################################
# Load Data
##############################################################

print("Loading averaged viscosity data...")

data = np.loadtxt(f"{base_path}/avgpoft.dat", skiprows=1)
eta = np.loadtxt(f"{base_path}/avgetaoft.dat", skiprows=1)
in_params = np.loadtxt("in.hGK_final", usecols=0)

##############################################################
# Extract Simulation Parameters
##############################################################

volume = in_params[0]        # nm^3
timeperframe = in_params[1]  # ps
temperature = in_params[2]   # K
C0 = in_params[3]            # bar^2
tau_low = in_params[4]       # ps
tau_up = in_params[5]        # ps
tau_cut = in_params[6]       # ps
kb = 1.380649e-23
dt = timeperframe * 1e-12
prefactor = dt * 1e-14 * volume / (kb * temperature)

##############################################################
# Prepare SACF Data
##############################################################

tau = data[:, 0]
sacf = data[:, 1]

if tau_low >= tau_up:
    raise ValueError("tau_low must be smaller than tau_up")

if tau_up >= tau[-1]:
    raise ValueError("tau_up exceeds SACF time range")

tau_low_id = np.searchsorted(tau, tau_low)
tau_up_id = np.searchsorted(tau, tau_up)

##############################################################
# Stretched Exponential Fit Function
##############################################################

def stretched_exp(x, a0, a1, a2):
    return a0 * np.exp(-(x / a1) ** a2)

##############################################################
# Perform Tail Fit
##############################################################

print("Fitting SACF tail...")

try:
    popt, _ = curve_fit(
        stretched_exp,
        tau[tau_low_id:tau_up_id],
        sacf[tau_low_id:tau_up_id],
        bounds=(0, np.inf),
        maxfev=100000
    )
except RuntimeError:
    raise RuntimeError("Tail fit failed. Adjust tau_low or tau_up.")

##############################################################
# Reconstruct SACF Tail
##############################################################

t_tail = np.arange(tau_low, tau_cut, 0.001)
sacf_tail = stretched_exp(t_tail, *popt)

##############################################################
# Integrate Tail Contribution
##############################################################

etan = C0 * prefactor * np.cumsum(sacf_tail) + eta[tau_low_id, 1]

##############################################################
# Save Outputs
##############################################################
n_log_points = 50

log_indices = np.unique(
    np.logspace(0, np.log10(len(t_tail) - 1),
                num=n_log_points,
                dtype=int)
)

np.savetxt(
    f"{base_path}/fit_tail_sacf.dat",
    np.column_stack([t_tail[log_indices], etan[log_indices]]),
    header="Time(ps)  Extrapolated viscosity integral",
    fmt="%.8e"
)
print("Final viscosity estimation completed.")