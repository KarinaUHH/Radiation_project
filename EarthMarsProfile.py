#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 17:31:19 2025

@author: karina
"""

import numpy as np
import matplotlib.pyplot as plt
import typhon as ty
import pyarts3 as pyarts
import xarray as xr

pyarts.data.download()
T_s = 290
RH = 0.8
#%%
# define single column atmosphere model
def single_column_atmosphere(T_s=290, T_cp=200, RH=0.8):
    """
    Create a single column atmosphere profile with a given surface temperature,
    cold point temperature and relative humidity.

    Parameters:
    -----------
    T_s : float
        Surface temperature in Kelvin.
    T_cp : float
        Cold point temperature in Kelvin.
    RH : float
        Relative humidity.

    Returns:
    --------
    pressure_profile : array
        Pressure profile in Pa.
    temperature_profile : array
        Temperature profile in Kelvin.
    wvvmr_profile : array
        Water vapor volume mixing ratio profile.
    rel_hum : array
        Relative humidity profile.
    """
    
    pressure = np.linspace(1000e2, 1e2, 100)
    height = ty.physics.pressure2height(pressure)
    temperature = np.ones(pressure.shape)
    temperature[0] = T_s

    for i, p in enumerate(pressure[:-1]):
        if temperature[i] > 0:
            moist_adiabat = ty.physics.moist_lapse_rate(p, temperature[i], e_eq=None)
            temperature[i + 1] = temperature[i] - moist_adiabat * (
                height[i + 1] - height[i]
            )

    temperature[temperature <= T_cp] = T_cp

    vmr = ty.physics.relative_humidity2vmr(RH, pressure, temperature)
    vmr[temperature == T_cp] = vmr[temperature == T_cp][0]
    
    

    return pressure, temperature, vmr
#%%
# Create custom atmosphere
pressure_profile, temperature_profile, wvvmr_profile = (
    single_column_atmosphere(T_s = T_s, RH= RH)
)
#%%
prozent = 1/100
atm = xr.Dataset(
    {
        "t": ("alt", temperature_profile),
        "p": ("alt", pressure_profile),
        "O2": ("alt", np.ones_like(pressure_profile) * 0.146*prozent),
        "N2": ("alt", np.ones_like(pressure_profile) * 01.89*prozent),
        "CO2": ("alt", np.ones_like(pressure_profile) * 95.97*prozent),
        "Ar": ("alt", np.ones_like(pressure_profile) * 01.93*prozent),
        "CO": ("alt", np.ones_like(pressure_profile) * 0.0557*prozent),
    },
    coords={"alt": ty.physics.pressure2height(pressure_profile), "lat": 0, "lon": 0},
)
atm["t"].attrs = {
    "units": "K",
    " long_name": "Temperature",
}
atm["p"].attrs = {
    "units": "Pa",
    "long_name": "Pressure",
}
atm["O2"].attrs = {
    "units": "mol/mol",
    "long_name": "Oxygen volume mixing ratio",
}
atm["N2"].attrs = {
    "units": "mol/mol",
    "long_name": "Nitrogen volume mixing ratio",
}
atm["CO2"].attrs = {
    "units": "mol/mol",
    "long_name": "Carbondioxid volume mixing ratio",
}
atm["Ar"].attrs = {
    "units": "mol/mol",
    "long_name": "Argon volume mixing ratio",
}
atm["CO"].attrs = {
    "units": "mol/mol",
    "long_name": "Carbonmonoxid volume mixing ratio",
}
atm["alt"].attrs = {
    "units": "m",
    "long_name": "Geometric altitude",
}
#%%
# plot the profiles
fig, ax = plt.subplots(1, 1, figsize=(8, 6), sharey=True)
plt.title('Temperature profile')
ax.plot(atm["t"], atm["p"])
ax.set_xlabel("Temperature / K")
ax.set_ylabel("Pressure / Pa")
ax.spines[["top", "right"]].set_visible(False)
#fig.savefig("single_column_atmosphere.png", dpi=300, bbox_inches="tight")
