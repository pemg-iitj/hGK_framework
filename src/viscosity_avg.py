#!/usr/bin/env python3
"""
==============================================================
Green-Kubo Viscosity Run Averaging
==============================================================

Averages SACF and viscosity integrals from multiple independent
trajectories.

INPUT FILE
----------
in.average_viscosity

Expected Directory Structure
----------------------------
run1/
    poft_coarse.dat
    etaoft_coarse.dat

run2/
    poft_coarse.dat
    etaoft_coarse.dat
...

OUTPUT FILES
------------
avgpoft.dat
avgetaoft.dat

AUTHORS
------
Akash Kumar Meel and Santosh Mogurampelly
"""

import numpy as np

##############################################################
# USER INPUT SECTION
##############################################################

control_file = "in.average_viscosity"

##############################################################
# Load Control Parameters
##############################################################

print("Reading averaging control file...")

in_params = np.loadtxt('in.average_viscosity', usecols=0)

ntrjfiles = int(in_params[0])
nframes_control = int(in_params[1])  # retained for consistency

##############################################################
# Initialize Accumulators
##############################################################

print("Initializing accumulators...")

sample_eta = np.loadtxt("run1/etaoft_coarse.dat", comments="#")
nframes = sample_eta.shape[0]

sum_poft = np.zeros((nframes, 7))
sumsq_poft = np.zeros((nframes, 7))

sum_eta = np.zeros((nframes, 7))
sumsq_eta = np.zeros((nframes, 7))

##############################################################
# Loop Over Trajectories
##############################################################

for i in range(ntrjfiles):

    print(f"Processing run {i+1}")

    poft = np.loadtxt(f"run{i+1}/poft_coarse.dat", comments="#")
    eta = np.loadtxt(f"run{i+1}/etaoft_coarse.dat", comments="#")

    sum_poft += poft[:, 1:]
    sumsq_poft += poft[:, 1:]**2

    sum_eta += eta[:, 1:]
    sumsq_eta += eta[:, 1:]**2

##############################################################
# Compute Statistics
##############################################################

print("Computing averages and standard deviations...")

mean_poft = sum_poft / ntrjfiles
std_poft = np.sqrt(sumsq_poft / ntrjfiles - mean_poft**2)

mean_eta = sum_eta / ntrjfiles
std_eta = np.sqrt(sumsq_eta / ntrjfiles - mean_eta**2)

time = poft[:, 0]  # assumed identical across runs

##############################################################
# Save Averaged SACF
##############################################################

data_poft = np.column_stack(
    [time] +
    [mean_poft[:, i] for i in range(7)] +
    [std_poft[:, i] for i in range(7)]
)

header_poft = (
    "# Time(ps) "
    + " ".join([f"avgP{i+1}" for i in range(7)])
    + " "
    + " ".join([f"stdP{i+1}" for i in range(7)])
)

np.savetxt(
    "avgpoft.dat",
    data_poft,
    header=header_poft,
    comments="",
    fmt="%.8e"
)

##############################################################
# Save Averaged Viscosity Integral
##############################################################

data_eta = np.column_stack(
    [time] +
    [mean_eta[:, i] for i in range(7)] +
    [std_eta[:, i] for i in range(7)]
)

header_eta = (
    "# Time(ps) "
    + " ".join([f"eta{i+1}" for i in range(7)])
    + " "
    + " ".join([f"std{i+1}" for i in range(7)])
)

np.savetxt(
    "avgetaoft.dat",
    data_eta,
    header=header_eta,
    comments="",
    fmt="%.8e"
)

print("Trajectory averaging completed.")