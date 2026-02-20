"""
==============================================================
Accelerated Green-Kubo Viscosity Calculation (FFT-based ACF)
==============================================================
Implements FFT-based SACF evaluation and cumulative Green-Kubo
integration for efficient viscosity calculations.

INPUT FILES
-----------
1. pressure_components.xvg
   - Time evolution of off-diagonal pressure tensor elements
   - Columns:
        time  Pxy  Pxz  Pyz  Pyx  Pzx  Pzy

2. in.viscosity
   - Control parameters:
        volume (nm^3)
        time per frame (ps)
        temperature (K)

OUTPUT FILES
------------
1. poft_coarse.dat
   - Normalized SACF for each component and average

2. etaoft_coarse.dat
   - Running Green-Kubo viscosity integral

AUTHORS
------
Akash Kumar Meel and Santosh Mogurampelly
"""


import numpy as np
import time

######################### Load Data #########################
t0 = time.time()
stress = np.loadtxt('./pressure_components.xvg',skiprows=29, dtype=np.float64)
in_params = np.loadtxt('./in.viscosity', usecols=0)
volume = in_params[0]                          # nm^-3
timeperframe = in_params[1]                    # ps
temperature = in_params[2]                     # K
kb = 1.380649e-23                              # J/K
dt = timeperframe*1e-12                        # s
prefactor = dt*1e-14*volume/(kb*temperature)   # mPa.s
pxy = stress[:,1]
pxz = stress[:,2]
pyz = stress[:,3]
pyx = stress[:,4]
pzx = stress[:,5]
pzy = stress[:,6]
components = {'pxy': pxy, 'pxz': pxz, 'pyz': pyz, 'pyx': pyx, 'pzx': pzx, 'pzy': pzy}
tf = time.time()
print(f"Data loading took: {tf - t0:.2f} s")
######################### ACF via FFT #########################
acf = {}
acf_norm = {}
for component, data in components.items():
    print(f"Calculating ACF for {component}...")
    ti = time.time()
    n = len(data)
    p_padded = np.pad(data, (0, n))
    fft = np.fft.fft(p_padded)
    acf_full = np.fft.ifft(fft * np.conj(fft)).real[:2*n]
    norm_factors = np.arange(n, 0, -1)
    acf[f'acf_{component}'] = acf_full[:n]
    acf_norm[f'acf_{component}'] = acf_full[:n]/norm_factors
    tf = time.time()
    print(f"ACF done for {component} in: {tf - ti:.2f} s")

poft = (1/6)*(sum(acf_norm.values()))
t = stress[:,0]
log_id = np.unique(np.logspace(0, np.log10(len(t) - 1), num=10000, dtype=int))
acf_data = np.column_stack([t, poft/poft[0], acf_norm['acf_pxy']/acf_norm['acf_pxy'][0], acf_norm['acf_pxz']/acf_norm['acf_pxz'][0], acf_norm['acf_pyz']/acf_norm['acf_pyz'][0], acf_norm['acf_pyx']/acf_norm['acf_pyx'][0], acf_norm['acf_pzx']/acf_norm['acf_pzx'][0], acf_norm['acf_pzy']/acf_norm['acf_pzy'][0]])
np.savetxt("poft_coarse.dat", acf_data[log_id], header=f"# p0 = {poft[0]:.4f}; time(ps) pavg pxy pxz pyz pyx pzx pzy", fmt='%.8f')
######################### Integrating ACF #########################
eta = {}
for component, data in acf_norm.items():
    print(f"Calculating running integral for {component}...")
    ti = time.time()
    eta[component] = prefactor*np.cumsum(data)
    tf = time.time()
    print(f"Running integral for {component} calculated in: {tf - ti:.2f} s")

etaoft = (1/6)*(sum(eta.values()))
eta_data = np.column_stack([t, etaoft, eta['acf_pxy'], eta['acf_pxz'], eta['acf_pyz'], eta['acf_pyx'], eta['acf_pzx'], eta['acf_pzy']])
np.savetxt("etaoft_coarse.dat", eta_data[log_id], header="# time(ps) etaavg etaxy etaxz etayz etayx etazx etazy", fmt='%.8f')
t1 = time.time()
print(f"ACF and Eta(t) for all six components took: {t1 - t0:.2f} s")
