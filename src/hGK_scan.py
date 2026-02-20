"""
==============================================================
Viscosity Window Convergence Scan
==============================================================

Evaluates sensitivity of Green-Kubo viscosity predictions to
SACF tail fitting window length.

The SACF tail is fitted using a stretched exponential model
over progressively increasing fitting windows. The resulting
viscosity estimates are used to assess convergence with respect
to window length.

INPUT FILES
-----------
avgpoft.dat
avgetaoft.dat
in.viscosity

OUTPUT FILE
-----------
viscosity_window_scan.dat

Columns:
1 Fit window length (ps)
2 Extrapolated viscosity (mPa.s)
3 Slope (d eta / d tau)

AUTHORS
------
Akash Kumar Meel and Santosh Mogurampelly
"""

import numpy as np
from scipy.optimize import curve_fit

##############################################################
# User-defined paths
##############################################################

base_path = "."

##############################################################
# Load Data
##############################################################

print("Loading averaged viscosity data...")

data = np.loadtxt(f"{base_path}/avgpoft.dat", skiprows=1)
eta = np.loadtxt(f"{base_path}/avgetaoft.dat", skiprows=1)
in_params = np.loadtxt("in.hGK_scan", usecols=0)

##############################################################
# Extract Simulation Parameters
##############################################################

volume = in_params[0]        # nm^3
timeperframe = in_params[1]  # ps
temperature = in_params[2]   # K
p0 = in_params[3]            # bar^2
tau_low = in_params[4]       # ps

kb = 1.380649e-23
dt = timeperframe * 1e-12
prefactor = dt * 1e-14 * volume / (kb * temperature)

##############################################################
# Prepare SACF Data
##############################################################

tau = data[:, 0]
sacf = data[:, 1]

if tau_low >= tau[-1]:
    raise ValueError("tau_low exceeds SACF time range.")

tau_low_id = np.searchsorted(tau, tau_low)

##############################################################
# Stretched Exponential Fit Function
##############################################################

def stretched_exp(x, a0, a1, a2):
    return a0 * np.exp(-(x / a1) ** a2)

##############################################################
# Window Scan Setup
##############################################################

start_id = tau_low_id + 10
tau_upper_indices = range(start_id, len(tau), 10)

viscosities = []
fit_window_lengths = []

##############################################################
# Window Scan Loop
##############################################################

for tau_upper_id in tau_upper_indices:

    tau_upper = tau[tau_upper_id]

    xdata = tau[tau_low_id:tau_upper_id]
    ydata = sacf[tau_low_id:tau_upper_id]

    try:
        popt, _ = curve_fit(
            stretched_exp,
            xdata,
            ydata,
            bounds=(0, np.inf),
            maxfev=100000
        )
    except RuntimeError:
        print(f"Fit failed for window ending at {tau_upper:.2f} ps")
        continue

    # Reconstruct SACF tail
    t_tail = np.arange(tau[tau_low_id], tau[-1], 0.001)
    sacf_tail = stretched_exp(t_tail, *popt)

    # Tail viscosity contribution
    etan = p0 * prefactor * np.cumsum(sacf_tail) + eta[tau_low_id, 1]
    eta_total = etan[-1]

    viscosities.append(eta_total)
    fit_window_lengths.append(tau_upper - tau_low)

##############################################################
# Convergence Gradient
##############################################################

viscosities = np.array(viscosities)
fit_window_lengths = np.array(fit_window_lengths)

viscosity_gradient = np.gradient(viscosities, fit_window_lengths)

##############################################################
# Save Output
##############################################################

viscosity_window_scan = np.column_stack([
    fit_window_lengths,
    viscosities,
    viscosity_gradient
])

np.savetxt(
    f"{base_path}/viscosity_window_scan.dat",
    viscosity_window_scan,
    header=(
        "# Fit window length (ps)\n"
        "# Extrapolated viscosity (mPa.s)\n"
        "# Slope (d eta / d tau)"
    ),
    fmt="%.8e"
)

print("Window convergence scan completed.")
