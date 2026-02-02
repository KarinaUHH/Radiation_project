#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 17:44:50 2026

@author: karina
Data: merimu_2001 Profile1; https://pds-atmospheres.nmsu.edu/data_and_services/atmospheres_data/catalog.htm#Mars
"""
#Rover 1 oder 2?
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

plotting = True
cutting = True
cut_value = 100000
saving = True
saving_name = './MarsProfiles_final.png'

# Feste Spalten (0-basiert!)
colspecs = [
    (0, 14),     # SCLK_TIME
    (23, 35),    # RADIAL_DISTANCE
    (497, 509),  # PRESS
    (521, 533),  # TEMP
]

names = ["time", "radius", "pressure", "temperature"]

# Datei einlesen
df = pd.read_fwf("./mars_profiledata.txt", colspecs=colspecs, names=names)

# Optional: Höhe über Mars-Oberfläche
mars_radius = 3390000.0  # Meter
df["altitude"] = df["radius"] - mars_radius
df_saving = df
df = df.iloc[::-1].reset_index(drop=True)
#%% earase douple values in height to have a monotoneous increasing height
counter = 1
prev_val = None

for i in range(len(df)):
    if prev_val is not None and df.loc[i, 'altitude'] == prev_val:
        counter += 1
        df.loc[i, 'altitude'] += 100 * (1 - 1 / counter)
    else:
        counter = 1
        prev_val = df.loc[i, 'altitude']
#%% Interpolate temperature to surface
"""
x = df['temperature'][45:100]
y = df['altitude'][45:100]

m, b = np.polyfit(x, y, 1)

plt.plot(x, y, 'yo', x, m*x+b, '--k')
plt.show()
#%% Interploate pressure to surface
x = np.log(df['pressure'])[:100]
y = df['altitude'][:100]

m, b = np.polyfit(x, y, 1)

plt.plot(x, y, 'yo', x, m*x+b, '--k')
plt.show()
"""
#%%Add Surface variables y = mx + c
deltaT,T0 = np.polyfit(df['altitude'][:45], df['temperature'][:45], 1)
deltap,p0_log = np.polyfit(df['altitude'][:100], np.log(df['pressure'][:100]), 1)
altitude = np.linspace(0,df['altitude'][0],100)[:-1]
T0=deltaT*altitude+T0
p0=np.exp(deltap*altitude+p0_log)
new_row = pd.DataFrame({
    'altitude': altitude,
    'pressure': p0,
    'temperature': T0
})

df = pd.concat([new_row, df], ignore_index=True)

plt.figure()
plt.plot(np.log(df['pressure'][:]),df['altitude'][:])

#%%
prozent = 1/100
atm = xr.Dataset(
    {
        "t": ("alt", df['temperature']),
        "p": ("alt", df['pressure']),
        "O2": ("alt", np.ones_like(df['pressure']) * 0.146*prozent),
        "N2": ("alt", np.ones_like(df['pressure']) * 01.89*prozent),
        "CO2": ("alt", np.ones_like(df['pressure']) * 95.97*prozent),
        "Ar": ("alt", np.ones_like(df['pressure']) * 01.93*prozent),
        "CO": ("alt", np.ones_like(df['pressure']) * 0.0557*prozent),
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
if cutting: 
    h_nearest = atm.alt.sel(alt=cut_value, method="nearest")
    atm = atm.sel(alt=slice(None, h_nearest))
    
    df_h_nearest = df.loc[(df["altitude"] - cut_value).abs().idxmin(), "altitude"]
    df = df[df["altitude"] <= df_h_nearest]


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
if plotting: 
    fig, ax = plt.subplots(1,2,figsize=(8,6))
    plt.suptitle("Mars vertical profiles - MER-Descent", fontsize=14)
    ax[0].plot(df["temperature"][:], df["altitude"][:], label='meassured')
    ax[0].plot(df['temperature'][0:100], df['altitude'][0:100], label = 'interpolated')
    #plt.gca().invert_yaxis()
    ax[0].set_xlabel("Temperature [K]")
    ax[0].set_ylabel("Altitude [m]")
    #ax[0].set_title("Temperatureprofile MER-EDL")
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(df["pressure"][:], df["altitude"][:],label = 'meassured')
    ax[1].plot(df['pressure'][0:100], df['altitude'][0:100], label = 'interpolated')
    #plt.gca().invert_yaxis()
    ax[1].set_xlabel("Pressure [Pa]")
    ax[1].set_ylabel("Altitude [m]")
    #ax[1].set_title("Pressureprofile MER-EDL")
    ax[1].grid()
    ax[1].legend()
    plt.tight_layout()
    if saving:
        plt.savefig(saving_name, dpi = 400)