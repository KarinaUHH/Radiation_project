#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 17:44:50 2026

@author: karina
Data: mg_2401; https://pds-atmospheres.nmsu.edu/cgi-bin/getdir.pl?dir=data&volume=mg_2401 
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==================================================
# User settings
# ==================================================
orbit = 3212
band = "X"

plotting = True
saving = True
saving_name = "venus_profiles_final.png"

# ==================================================
# Column definitions
# ==================================================
rtpd_names = [
    "WAVELENGTH", "ORBIT", "ALTITUDE",
    "REFRACTIVITY", "REFRACT_DEV",
    "TEMPERATURE", "TEMP_DEV",
    "PRESSURE", "PRESS_DEV",
    "DENSITY", "DENS_DEV",
    "LATITUDE", "LONGITUDE",
    "ZENITH_ANGLE", "LOCAL_TIME", "ERT"
]

abs_names = [
    "WAVELENGTH", "ORBIT", "ALTITUDE",
    "ABSORPTIVITY", "ABSORP_DEV",
    "H2SO4_VOLMIX", "H2SO4_VM_DEV",
    "LATITUDE", "LONGITUDE",
    "ZENITH_ANGLE", "LOCAL_TIME", "ERT"
]

# ==================================================
# Read data
# ==================================================
rtpd = pd.read_csv(
    "VenusData.txt",
    sep=r"\s+",
    header=None,
    names=rtpd_names,
    engine="python"
)

absorp = pd.read_csv(
    "VenusH2SO4.txt",
    sep=r"\s+",
    header=None,
    names=abs_names,
    engine="python"
)

# ==================================================
# Select one orbit & band
# ==================================================
rtpd_sel = rtpd[
    (rtpd["ORBIT"] == orbit) &
    (rtpd["WAVELENGTH"] == band)
].copy()

abs_sel = absorp[
    (absorp["ORBIT"] == orbit) &
    (absorp["WAVELENGTH"] == band)
].copy()

if rtpd_sel.empty:
    raise ValueError("No RTPD data found for this orbit/band")

if abs_sel.empty:
    raise ValueError("No ABS data found for this orbit/band")

# ==================================================
# Sort by altitude
# ==================================================
rtpd_sel = rtpd_sel.sort_values("ALTITUDE")
abs_sel = abs_sel.sort_values("ALTITUDE")

# ==================================================
# Cut out negativ vmx of H2SO4
# ==================================================
abs_sel["H2SO4_VOLMIX"] = abs_sel["H2SO4_VOLMIX"].where(
    abs_sel["H2SO4_VOLMIX"] > 0, np.nan
)


# ==================================================
# Unit conversion
# ==================================================
alt_rtpd = rtpd_sel["ALTITUDE"].values * 1e3       # km → m
temp_K   = rtpd_sel["TEMPERATURE"].values          # K
pres_Pa  = rtpd_sel["PRESSURE"].values * 1e5       # bar → Pa

alt_abs  = abs_sel["ALTITUDE"].values * 1e3        # km → m
h2so4_vm = abs_sel["H2SO4_VOLMIX"].values  *10**-6 # volume mixing ratio ppm -> mol/mol

# ==================================================
# Unit conversion in Dataframe
# ==================================================
rtpd_sel["ALTITUDE"] = rtpd_sel["ALTITUDE"].values * 1e3       # km → m
rtpd_sel["PRESSURE"]  = rtpd_sel["PRESSURE"].values * 1e5       # bar → Pa

abs_sel["ALTITUDE"]  = abs_sel["ALTITUDE"].values * 1e3        # km → m
abs_sel["H2SO4"] = abs_sel["H2SO4_VOLMIX"].values  *10**-6 # volume mixing ratio ppm -> mol/mol

from scipy.interpolate import interp1d

f_h2so4 = interp1d(
    abs_sel["ALTITUDE"].values,
    abs_sel["H2SO4_VOLMIX"].values,
    bounds_error=False,
    fill_value=np.nan
)

rtpd_sel['H2SO4'] = f_h2so4(rtpd_sel["ALTITUDE"])

#%%
#==================================================
# Maskieren Nans
#==================================================
# Auswahl der ersten 100 Punkte
alt_fit = alt_rtpd[:100]
pres_fit = pres_Pa[:100]

# Filter: nur gültige Werte (keine NaN, nur positive Drücke)
mask = (~np.isnan(pres_fit)) & (pres_fit > 0)

alt_fit = alt_fit[mask]
pres_fit = pres_fit[mask]

# Polyfit auf log(Pressure)
deltap, p0_log = np.polyfit(alt_fit, np.log(pres_fit), 1)
#==================================================
# Interpolation am Boden 
#==================================================
deltaT,T0 = np.polyfit(alt_rtpd[:100], temp_K[:100], 1)
#deltap,p0_log = np.polyfit(alt_rtpd[:100], np.log(pres_Pa[:100]), 1)
#deltaH2SO4,H2SO4_0 = np.polyfit(alt_rtpd[:25], np.log(h2so4_vm[:25]), 1)
altitude = np.linspace(0,alt_rtpd[0],100)[:-1]
T0=deltaT*altitude+T0
p0=np.exp(deltap*altitude+p0_log)
#h2so4= np.exp(deltaH2SO4*altitude+H2SO4_0)
h2so4 = np.ones(shape = 99)*h2so4_vm[0]
new_row = pd.DataFrame({
    'altitude': altitude,
    'pressure': p0,
    'temperature': T0,
    'H2SO4': h2so4
})
df = pd.DataFrame({'altitude':rtpd_sel['ALTITUDE'].values, 
                   'pressure': rtpd_sel['PRESSURE'].values,
                   'temperature': rtpd_sel['TEMPERATURE'].values,
                   'H2SO4':rtpd_sel['H2SO4'].values})
df = pd.concat([new_row, df], ignore_index=True)

plt.figure()
plt.plot(np.log(df['pressure'][:]),df['altitude'][:])
#%%
#==================================================
#xr atmosphere !!!!
#==================================================
import xarray as xr
prozent = 1/100
ppm = 10**-6
atm = xr.Dataset(
    {
        "t": ("alt", df['temperature']),
        "p": ("alt", df['pressure']),
        "H2SO4": ("alt", df['H2SO4']),
        "CO2": ("alt", np.ones_like(df['pressure']) * 96.5*prozent),
        "N2": ("alt", np.ones_like(df['pressure']) * 03.5*prozent),
        "He": ("alt", np.ones_like(df['pressure']) * 12*ppm),
        "Ar": ("alt", np.ones_like(df['pressure']) * 70*ppm),
        "H2O": ("alt", np.ones_like(df['pressure']) * 30*ppm),
    },
    coords={"alt": df['altitude'], "lat": 0, "lon": 0},
)
atm["t"].attrs = {
    "units": "K",
    " long_name": "Temperature",
}
atm["p"].attrs = {
    "units": "Pa",
    "long_name": "Pressure",
}
atm["H2SO4"].attrs = {
    "units": "mol/mol",
    "long_name": "H2SO4 volume mixing ratio",
}
atm["CO2"].attrs = {
    "units": "mol/mol",
    "long_name": "CO2 volume mixing ratio",
}
atm["N2"].attrs = {
    "units": "mol/mol",
    "long_name": "Nitrogen volume mixing ratio",
}
atm["He"].attrs = {
    "units": "mol/mol",
    "long_name": "Helium volume mixing ratio",
}
atm["Ar"].attrs = {
    "units": "mol/mol",
    "long_name": "Argon volume mixing ratio",
}
atm["H2O"].attrs = {
    "units": "mol/mol",
    "long_name": "Watervapor volume mixing ratio",
}
atm["alt"].attrs = {
    "units": "m",
    "long_name": "Geometric altitude",
}

atm_Earth = xr.Dataset(
    {
        "t": ("alt", df['temperature']),
        "p": ("alt", df['pressure']),
        "H2O": ("alt", np.ones_like(df['pressure'])*3*prozent),
        "O2": ("alt", np.ones_like(df['pressure']) * 0.21),
        "N2": ("alt", np.ones_like(df['pressure']) * 0.78),
        "CO2": ("alt", np.ones_like(df['pressure']) * 4e-4),
        "O3": ('alt', np.ones_like(df['pressure']) * 1e-6),
    },
    coords={"alt": df['altitude'], "lat": 0, "lon": 0},
)
atm_Earth["t"].attrs = {
    "units": "K",
    " long_name": "Temperature",
}
atm_Earth["p"].attrs = {
    "units": "Pa",
    "long_name": "Pressure",
}
atm_Earth["H2O"].attrs = {
    "units": "mol/mol",
    "long_name": "Water vapor volume mixing ratio",
}
atm_Earth["O2"].attrs = {
    "units": "mol/mol",
    "long_name": "Oxygen volume mixing ratio",
}
atm_Earth["N2"].attrs = {
    "units": "mol/mol",
    "long_name": "Nitrogen volume mixing ratio",
}
atm_Earth["alt"].attrs = {
    "units": "m",
    "long_name": "Geometric altitude",
}
#%%
# ==================================================
# Plot
# ==================================================
if plotting:
    fig, ax = plt.subplots(1, 3, figsize=(9, 6), sharey=True)

    # Temperature profile
    ax[0].plot(df['temperature'], df['altitude'], label='meassured')
    ax[0].plot(df['temperature'][:100], df['altitude'][:100], label = 'interpolated')
    ax[0].set_xlabel("Temperature [K]")
    ax[0].set_ylabel("Altitude [m]")
    ax[0].grid()
    ax[0].legend()

    # Pressure profile
    ax[1].plot(pres_Pa, alt_rtpd, label='meassured')
    ax[1].plot(df['pressure'][:100], df['altitude'][:100], label = 'interpolated')
    ax[1].set_xscale("log")
    ax[1].set_xlabel("Pressure [Pa]")
    ax[1].grid(which="both")
    ax[1].legend()

    # H2SO4 profile
    ax[2].plot(h2so4_vm, alt_abs, label='meassured')
    #ax[2].plot(df['H2SO4'][:100], df['altitude'][:100], label = 'interpolated')
    ax[2].set_xscale("log")
    ax[2].set_xlabel("H₂SO₄ volume mixing ratio")
    ax[2].grid(which="both")
    ax[2].legend()

    plt.suptitle(f"Venus vertical profiles – Orbit {orbit}, Band {band}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if saving:
        plt.savefig(saving_name, dpi=400)

    plt.show()
