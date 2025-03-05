# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 09:46:00 2025

@author: islam9
"""
import numpy as np
import matplotlib.pyplot as plt


def calculate_power_latent_heat_loss(mass_grams, latent_heat_fusion, molar_mass, time_seconds):

    n = mass_grams / molar_mass
    
    Q_kJ = n * latent_heat_fusion  
    
    Q_J = Q_kJ * 1000
    
    power = Q_J / time_seconds
    return  Q_J


molar_mass_Li = 6.94  # in g/mol
latent_heat_fusion_Li = 3.0  # in kJ/mol
time_to_melt = 1e-3  # time for phase change in seconds (example: 100 seconds)

# Mass range (in grams) for the plot
mass_range = np.linspace(1e-3, 10, 100)  # from 10g to 1000g

# Calculate power for each mass in the range
power_values = [calculate_power_latent_heat_loss(mass, latent_heat_fusion_Li, molar_mass_Li, time_to_melt) for mass in mass_range]

# Plotting the results
plt.figure(figsize=(4, 3))
plt.plot(mass_range, (power_values), color='blue', linewidth=2)
plt.title("Q = L$_f$*m, L$_f$ $\sim$ 0.433 kJ/g", fontsize=14)
plt.xlabel("Mass of Lithium (grams)", fontsize=12)
plt.ylabel("Heat loss (J)", fontsize=12)
#plt.xlim([0, 10])
#plt.ylim([0, 5])
plt.grid(True)
#plt.legend()
plt.show()


density_lithium = 0.534  # in g/cm³
dx = 1e-3  # in cm (length)
dy = 1e-2  # in cm (width)
thickness = 0.2  # in cm (thickness)

# Calculate the area
area = dx * dy

# Calculate the volume
volume = area * thickness

# Calculate the mass
mass = density_lithium * volume

# Output the result
print(f"The mass of the lithium layer is {mass:.6e} grams.")
