"""
Plotting spectral absorption coefficients for H2O, CO2, and O3
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
import pyarts3 as pyarts

pyarts.data.download()
plot_format = dict(s=1, alpha=0.7)
plot = plt.scatter
hz2kayser = pyarts.arts.convert.freq2kaycm(1.0)
# plot_format = dict(lw=0.25, alpha=1)
# plot = plt.plot
# %%
plt.figure(figsize=(10, 6))
wn_kayser = np.linspace(10, 2500, 10000)  # wavenumbers in kayser
freq_hz = pyarts.arts.convert.kaycm2freq(wn_kayser)  # frequencies in Hz
"""gases=['H2O', 'CO2', 'O3', 'N2O', "CH4", "O2", "NO", "SO2", "NO2", "HNO3", "OH", "HF", "HCl",
    "HBr", "HI", "ClO", "OCS", "H2CO", "HOCl", "N2",
    "CH3Cl", "H2O2", "C2H2", "C2H6", "PH3", "COF2",
    "H2O", "HO2", "O", "NO+",
    "HOBr", "C2H4", "CH3OH", "CH3Br", "C4H2", "H2", "CS", "SO3", "C2N2", "COCl2", "SO",
    "CS2", "CH3", "C3H4", "C2F6", "C3F8", "C4F10", "C5F12",
    "C6F14", "C8F18", "cC4F8", "CCl4", "CFC11", "CFC113", "CFC114",
    "CFC115", "CFC12", "CH2Cl2", "CH3CCl3", "CHCl3", "Halon1211",
    "Halon1301", "Halon2402", "HCFC141b", "HCFC142b", "HCFC22",
    "HFC125", "HFC134a", "HFC143a", "HFC152a", "HFC227ea",
    "HFC23", "HFC236fa", "HFC245fa", "HFC32", "HFC365mfc", "SO2F2", "HFC4310mee", 
    "GeH4", "CH3I", "CH3F"
]"""
gases = ["He"]
# NH3, Ar, HDCO, D2CO, HCN, SF6, H2S, HCOOH, DCOOH, HCOOD, ClONO2, CH3CN, CH2DCN, CF4,
#HC3N, H2SO4, HNC, BrO, OClO, He, Cl2O2, H, NF3
for gas in gases:
    gas_recipe = pyarts.recipe.SingleSpeciesAbsorption(species=gas, cutoff=750e9)
    
    atm = pyarts.arts.AtmPoint()
    atm.set_species_vmr("NH3", 0.5) #ppm
    atm.temperature = 260.0
    atm.pressure = 100e2
    
    gas_abs = gas_recipe(freq_hz, atm) / atm.number_density(gas) / hz2kayser
    plt.scatter(wn_kayser, gas_abs, label=gas, **plot_format)

plt.xlabel("Wavenumber / cm$^{-1}$")
plt.ylabel("Mass absorption coefficient / m$^2$") #Molecule absorption coeffiecient/ m^2
#plt.legend()
plt.yscale("log")
plt.ylim(1e-21)
plt.tight_layout()
plt.show()
# %%
