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
#save_name = f"OLR_diff{int(T_adjust)}{int(vmr_adjust)}_300K.png"
#save_name_2 = "Forcing.png"
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
    single_column_atmosphere()
)
#%%
atm = xr.Dataset(
    {
        "t": ("alt", temperature_profile),
        "p": ("alt", pressure_profile),
        "H2O": ("alt", wvvmr_profile),
        "O2": ("alt", np.ones_like(pressure_profile) * 0.21),
        "N2": ("alt", np.ones_like(pressure_profile) * 0.78),
        "CO2": ("alt", np.ones_like(pressure_profile) * 4e-4),
        "O3": ('alt', np.ones_like(pressure_profile) * 1e-6),
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
atm["H2O"].attrs = {
    "units": "mol/mol",
    "long_name": "Water vapor volume mixing ratio",
}
atm["O2"].attrs = {
    "units": "mol/mol",
    "long_name": "Oxygen volume mixing ratio",
}
atm["N2"].attrs = {
    "units": "mol/mol",
    "long_name": "Nitrogen volume mixing ratio",
}
atm["alt"].attrs = {
    "units": "m",
    "long_name": "Geometric altitude",
}
#print(atm)
#%%

# plot the profiles
fig, ax = plt.subplots(1, 3, figsize=(12, 6), sharey=True)
plt.suptitle(f"Earth vertical profiles – single column model", fontsize=14)
ax[0].plot(atm["t"], atm["p"] / 1e2)
ax[0].set_xlabel("Temperature / K")
ax[0].set_ylabel("Pressure / hPa")
ax[1].plot(atm["H2O"], atm["p"] / 1e2)
ax[1].set_xlabel("H2O VMR")
# ax[1].set_xscale("log")
ax[2].plot(ty.physics.vmr2relative_humidity(atm['H2O'], atm['p'], atm['t']), atm["p"] / 1e2)
ax[2].set_xlabel("Relative humidity")
ax[2].invert_yaxis()

for ax in ax:
    ax.spines[["top", "right"]].set_visible(False)
fig.savefig("Earth_profile.png", dpi=400, bbox_inches="tight")

#%%
def calculate_olr(atm, f_grid, species):
    """
    Calculates the outgoing longwave radiation (OLR) ath the top of the atmosphere (TOA) for a given atmospheric profile.

    Parameters:
    -----------
    atm : xarray.Dataset
        Atmospheric profile.
        Coordinates:
        - alt : Geometric altitude in meters.
        - lat : Latitude in degrees.
        - lon : Longitude in degrees.
        MUST hold the following species profiles:
        - t : Temperature profile in Kelvin.
        - p : Pressure profile in Pa.
        CAN hold all other species known to arts, like H2O, O2, N2, O3, CO2, etc.

    f_grid : array
        Frequency grid in Hz.
        
    species : list
        List of species to be considered in the radiative transfer calculation.


    Returns:
    --------
    LW_arr : xarray.DataArray
        Longwave radiation at TOA in W cm / m^2
    """

    # Set some default parameters
    NQuad = 16
    max_level_step = 1e3
    cutoff = ["ByLine", 750e9]
    remove_lines_percentile = 70
    planet = "Earth"

    # Create a pyarts workspace
    ws = pyarts.Workspace()
    ws.frequency_grid = f_grid

    # Set the atmospheric profile and find according absorption species
    ws.atmospheric_field = pyarts.data.to_atmospheric_field(atm)
    ws.absorption_species = species
    ws.ReadCatalogData(ignore_missing=True)
    ws.propagation_matrix_agendaAuto(T_extrapolfac=1e9)

    # Specify cutoffs for absorption bands to save computational time
    for band in ws.absorption_bands:
        ws.absorption_bands[band].cutoff = cutoff[0]
        ws.absorption_bands[band].cutoff_value = cutoff[1]

    # Remove lines with low absorption
    ws.absorption_bands.keep_hitran_s(remove_lines_percentile)

    # Set the surface properties
    ws.surface_fieldPlanet(option=planet)
    ws.surface_field["t"] = atm["t"].sel(alt=0).values

    # Set disort settings
    ws.disort_quadrature_dimension = NQuad
    ws.disort_fourier_mode_dimension = 1
    ws.disort_legendre_polynomial_dimension = 1

    # Set the ray path
    ws.ray_pathGeometricDownlooking(
        latitude=atm["lat"].values,
        longitude=atm["lon"].values,
        max_step=max_level_step,
    )

    # Set up geometry of observation
    pos = [100e3, 0, 0]
    los = [180.0, 0.0]
    ws.ray_pathGeometric(pos=pos, los=los, max_step=1000.0)
    ws.spectral_radianceClearskyEmission()

    # Extract the OLR and build a xarray.DataArray
    LW_up = (
        ws.spectral_radiance[:, 0]*3e10*np.pi # Convert from W/m2/Hz/sr to W/cm2/cm-1
    )
    kayser = pyarts.arts.convert.freq2kaycm(ws.frequency_grid)
    LW_arr = xr.DataArray(LW_up, dims=["wavenum"], coords={"wavenum": kayser})
    LW_arr["wavenum"].attrs = {
        "units": "cm-1",
        "long_name": "Wavenumber",
    }
    LW_arr.attrs = {
        "units": "W cm m-2",
        "long_name": "Upwelling LW flux at TOA",
    }

    return LW_arr
#%%
# Calculate OLR with ARTS
olr_all = calculate_olr(atm, pyarts.arts.convert.kaycm2freq(np.linspace(1, 2500, 1000)), ['H2O', 'CO2', 'O3','N2', 'O2'])
#olr_no_co2 = calculate_olr(atm, pyarts.arts.convert.kaycm2freq(np.linspace(1, 2500, 1000)), ['H2O', 'H2O-ForeignContCKDMT400', 'H2O-SelfContCKDMT400', 'O3'])
#olr_no_o3 = calculate_olr(atm, pyarts.arts.convert.kaycm2freq(np.linspace(1, 2500, 1000)), ['H2O', 'H2O-ForeignContCKDMT400', 'H2O-SelfContCKDMT400', 'CO2'])
#olr_only_h2o = calculate_olr(atm, pyarts.arts.convert.kaycm2freq(np.linspace(1, 2500, 1000)), ['H2O', 'H2O-ForeignContCKDMT400', 'H2O-SelfContCKDMT400'])
#%%
"""
# Plot the OLR
fig, ax = plt.subplots(2, 2, figsize=(12, 6))
lw = 0.75
olr_all.plot(ax=ax[0,0], label='$H_2O, CO_2, O_3$', lw = lw)
olr_no_o3.plot(ax=ax[1,0], label='$H_2O, CO_2$', color='orange', lw = lw)
olr_no_co2.plot(ax=ax[0,1], label='$H_2O, O_3$', color='green', lw = lw)
olr_only_h2o.plot(ax=ax[1,1], label='$H_2O$', color='red' , lw = lw)
ax[0,1].set_xlabel("")
ax[0,0].set_xlabel("")
ax[1,0].set_xlabel("Wavenumber / cm$^{-1}$")
ax[0,0].set_ylabel("OLR / W m$^{-2}$ cm")
ax[1,0].set_ylabel("OLR / W m$^{-2}$ cm")
olrs = [olr_all, olr_no_co2, olr_no_o3, olr_only_h2o]
for i, axs in enumerate(ax.flatten()):
    axs.spines[["top", "right"]].set_visible(False)
    axs.legend()
    axs.set_title(f"Flux = {np.trapezoid(olrs[i], x=olrs[i]['wavenum'], axis=0):0.2f}" + "W m$^{{-2}}$")

fig.tight_layout()
fig.savefig("olr_arts.png", dpi=300, bbox_inches="tight")
"""
#%%

# Calculate planck curve with typhon for different temperatures
temps = [200, 230, 260, 290]
lw_planck = []
for temp in temps:
    lw_planck.append(ty.physics.planck_wavenumber(olr_all["wavenum"]*1e2, temp)*np.pi*1e2)

#%%
# Plot TOA from ARTS together with Planck curves

fig, ax = plt.subplots(2,1, figsize=(8, 6), gridspec_kw={'height_ratios': [2, 1]})
olr_all.plot(ax=ax[0], label="OLR ARTS", color='grey', alpha = 0.6)
olr_all.groupby_bins('wavenum', np.linspace(1, 2500, 200)).mean('wavenum').plot(ax=ax[0], color='black', label="OLR ARTS binned")
for i, temp in enumerate(temps):
    ax[0].plot(olr_all["wavenum"], lw_planck[i], label=f"Planck {temp} K")

ax[0].set_xlabel("Wavenumber / cm$^{-1}$")
ax[0].set_ylabel("OLR / W m$^{-2}$ cm")
ax[0].spines[["top", "right"]].set_visible(False)
ax[0].legend()

plot_format = dict(s=1, alpha=0.7)
hz2kayser = pyarts.arts.convert.freq2kaycm(1.0)

frequenz = np.linspace(1, 2500, 1000)
gases = ['H2O', 'CO2', 'O3','N2', 'O2']

wn_kayser = frequenz  # wavenumbers in kayser
freq_hz = pyarts.arts.convert.kaycm2freq(wn_kayser)  # frequencies in Hz
for gas in gases:
    gas_recipe = pyarts.recipe.SingleSpeciesAbsorption(species=gas, cutoff=750e9)
    
    atm = pyarts.arts.AtmPoint()
    atm.set_species_vmr(gas, 0.5) #ppm
    atm.temperature = 260.0
    atm.pressure = 100e2
    
    gas_abs = gas_recipe(freq_hz, atm) / atm.number_density(gas) / hz2kayser
    ax[1].scatter(wn_kayser, gas_abs, label=gas, **plot_format)
ax[1].set_title('Absorption cross section per molecule dependent on wavenumber')
ax[1].set_xlabel("Wavenumber / cm$^{-1}$")
ax[1].set_ylabel("1/ m$^2$") #Molecule absorption coeffiecient/ m^2
ax[1].legend(ncol=2, markerscale=5)
ax[1].set_yscale("log")
ax[1].set_ylim(1e-21)
plt.tight_layout()


fig.savefig("olr_Earth.png", dpi=400, bbox_inches="tight")

#%%
"""
# Plot TOA difference from ARTS
fig, ax = plt.subplots(figsize=(8, 6))
olr_diff = olr_all_adjust-olr_all
olr_diff.plot(ax=ax, label="OLR ARTS", color='grey', alpha = 0.6)
olr_diff.groupby_bins('wavenum', np.linspace(1, 2500, 200)).mean('wavenum').plot(ax=ax, color='black', label="OLR ARTS binned")
ax.set_xlabel("Wavenumber / cm$^{-1}$")
ax.set_ylabel("OLR / W m$^{-2}$ cm")
ax.spines[["top", "right"]].set_visible(False)
ax.legend()
#fig.savefig(save_name, dpi=300, bbox_inches="tight")
"""
#%%
"""
atm_dummy = atm
olr_all = calculate_olr(atm, pyarts.arts.convert.kaycm2freq(np.linspace(1, 2500, 1000)), ['CO2'])
change=[0.25,0.5,1,2,4]
force = []
for forcing in change:
    atm_dummy["CO2"] = forcing * 4e-4

    #atm_dummy["CO2"] = np.ones_like(pressure_profile)* forcing *4e-4
    olr_dummy = calculate_olr(atm, pyarts.arts.convert.kaycm2freq(np.linspace(1, 2500, 1000)), ['CO2'])
    olr_diff_force = olr_dummy-olr_all
    force.append(olr_diff_force.integrate("wavenum"))
"""
#%%
"""
fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(change, force)
ax.set_title("Forcing of variating CO2, T_s=300K")
ax.set_xlabel("x times CO2")
ax.set_ylabel("Forcing / W m$^{-2}$ cm")
ax.set_xscale('log')
ax.spines[["top", "right"]].set_visible(False)
ax.legend()
#fig.savefig(save_name_2, dpi=300, bbox_inches="tight")
"""


