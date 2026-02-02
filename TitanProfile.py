#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 17:44:50 2026

@author: karina
Data: hphasi_0001, https://pds-atmospheres.nmsu.edu/PDS/data/hphasi_0001/DATA/PROFILES/HASI_L4_ATMO_PROFILE_DESCEN.TAB
"""

import numpy as np
import pandas as pd
import xarray as xr
import typhon as ty
import matplotlib.pyplot as plt

save_plot = True
save_name = './TitanProfiles_final.png'
plotting = True
path = "./Decend_Saturn_Titan.txt"

df = pd.read_csv(
    path,
    sep=";",
    header=None,
    names=["Time", "altitude", "pressure", "temperature", "density"]
)

# Nur die gewünschten Spalten auswählen
df = df[["altitude", "pressure", "temperature"]]

# Optional: Höhe über Mars-Oberfläche
#mars_radius = 3390000.0  # Meter
#df["altitude"] = df["radius"] - mars_radius
df_saving = df
df = df.iloc[::-1].reset_index(drop=True)
#%%
"""
counter = 1
prev_val = None

for i in range(len(df)):
    if prev_val is not None and df.loc[i, 'altitude'] == prev_val:
        counter += 1
        df.loc[i, 'altitude'] += 100 * (1 - 1 / counter)
    else:
        counter = 1
        prev_val = df.loc[i, 'altitude']
"""
#%%
"""
x = df['temperature'][:100]
y = df['altitude'][:100]

m, b = np.polyfit(x, y, 1)

plt.plot(x, y, 'yo', x, m*x+b, '--k')
plt.show()
#%%
x = np.log(df['pressure'])[:200]
y = df['altitude'][:200]

m, b = np.polyfit(x, y, 1)

plt.plot(x, y, 'yo', x, m*x+b, '--k')
plt.show()
"""
#%%Add Surface variables
#calculate deltaT
deltaT,T0 = np.polyfit(df['altitude'][:45], df['temperature'][:45], 1)
deltap,p0_log = np.polyfit(df['altitude'][:100], np.log(df['pressure'][:100]), 1)
p0=np.exp(p0_log)
new_row = pd.DataFrame({
    'altitude': [0],
    'pressure': [p0],
    'temperature': [T0]
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
        "CH4": ("alt", np.ones_like(df['pressure']) * 05.00*prozent),
        "N2": ("alt", np.ones_like(df['pressure']) * 95.00*prozent),
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
atm["CH4"].attrs = {
    "units": "mol/mol",
    "long_name": "Methane volume mixing ratio",
}
atm["N2"].attrs = {
    "units": "mol/mol",
    "long_name": "Carbondioxid volume mixing ratio",
}

atm_Earth = xr.Dataset(
    {
        "t": ("alt", df['temperature']),
        "p": ("alt", df['pressure']),
        "H2O": ("alt", np.ones_like(df['pressure'])* 3*prozent),
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
    plt.suptitle(f"Titan vertical profiles - HASI", fontsize=14)
    ax[0].plot(df["temperature"][:], df["altitude"][:], label='meassured')
    ax[0].plot(df['temperature'][0:2], df['altitude'][0:2], label = 'interpolated')
    #plt.gca().invert_yaxis()
    ax[0].set_xlabel("Temperature [K]")
    ax[0].set_ylabel("Altitude [m]")
    #ax[0].set_title("Temperatureprofile MER-EDL")
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(df["pressure"][:], df["altitude"][:],label = 'meassured')
    ax[1].plot(df['pressure'][0:2], df['altitude'][0:2], label = 'interpolated')
    #plt.gca().invert_yaxis()
    ax[1].set_xlabel("Pressure [Pa]")
    ax[1].set_ylabel("Altitude [m]")
    #ax[1].set_title("Pressureprofile MER-EDL")
    ax[1].grid()
    ax[1].legend()
    plt.tight_layout()
    if save_plot:
        plt.savefig(save_name, dpi=400)
    