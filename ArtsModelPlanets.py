#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 10:46:11 2026

@author: karina
"""

import numpy as np
import matplotlib.pyplot as plt
import typhon as ty
import pyarts3 as pyarts
import xarray as xr
from VenusProfile import atm, atm_Earth

pyarts.data.download()
vmr = 0.5

gases = ['O2','N2', 'CO2','CO'] #Complete Mars Missing Argon
frequenz = np.linspace(1, 2500, 1000) #Mars and Earth
#planck_temperature = [180, 250] #for Mars


#gases = ['CH4','N2'] # Complete Titan
#frequenz = np.linspace(1, 1000, 500) #Titan
#planck_temperature = [75, 93, 140, 167] #for Titan + Earth

#planck_temperature = [290, 200] #for Earth
#gases = ['H2O', 'O2','N2', 'CO2','O3'] #for Earth

gases = ['H2O', 'CO2', 'N2'] #for Venus missing Ar, He, H2SO4
planck_temperature = [708, 575, 250] #for Venus
frequenz = np.linspace(1, 5000, 1000) #for Venus

Earth = False
planck_curve = True
single_gas = False
gases_plot = gases
save_plot = False
save_name = './Plots/OLR_EarthProfMarsGas_final.png'

if Earth: atm = atm_Earth
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
    atm = atm.sortby("alt")
    atm = atm.where((atm.t > 0) & (atm.p > 0), drop=True)
    atm = atm.dropna(dim="alt", how="any")
    atm = atm.expand_dims(lat=[0.0], lon=[0.0])
    
    ws.atmospheric_field = pyarts.data.to_atmospheric_field(atm)

    #ws.atmospheric_field = pyarts.data.to_atmospheric_field(atm)
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
olr_all = calculate_olr(atm, pyarts.arts.convert.kaycm2freq(frequenz), gases)

if planck_curve:
    #calculate planck curves
    temps = planck_temperature
    lw_planck = []
    for temp in temps:
        lw_planck.append(ty.physics.planck_wavenumber(olr_all["wavenum"]*1e2, temp)*np.pi*1e2)
if single_gas:
    #calculate olr for only a single gas
    olr_only= []
    for gas in gases:    
        olr_only.append(calculate_olr(atm, pyarts.arts.convert.kaycm2freq(frequenz), [gas]))
#%%
# Plot TOA from ARTS
fig, ax = plt.subplots(2,1, figsize=(8, 6), gridspec_kw={'height_ratios': [2, 1]})

if planck_curve:
    for i, temp in enumerate(temps):
        ax[0].plot(olr_all["wavenum"], lw_planck[i], label=f"Planck {temp} K")
if single_gas: 
    for i, gas in enumerate(gases):
        if gas in gases_plot:
            ax[0].plot(olr_all["wavenum"], olr_only[i], label=f"With only {gas} in the atmosphere")
olr_all.plot(ax=ax[0], label="OLR ARTS", color='grey', alpha = 0.6)
olr_all.groupby_bins('wavenum', frequenz).mean('wavenum').plot(ax=ax[0], color='black', label="OLR ARTS binned")
ax[0].set_title('OLR in dependence of wavenumber')
ax[0].set_xlabel("Wavenumber / cm$^{-1}$")
ax[0].set_ylabel("OLR / W m$^{-2}$ cm")
ax[0].spines[["top", "right"]].set_visible(False)
ax[0].legend()

plot_format = dict(s=1, alpha=0.7)
hz2kayser = pyarts.arts.convert.freq2kaycm(1.0)

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
if save_plot:
    plt.savefig(save_name, dpi = 400)

