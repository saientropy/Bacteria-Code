import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

# Load the data
pressure_data = pd.read_csv('plot_data_first_1.csv')
area_data = pd.read_csv('plot_data_second_1.csv')

# Rename columns for easier access (based on your CSV info)
pressure_data.columns = ['Pressure_atm', 'PM_Tension', 'PG_Tension', 'Total_Tension']
area_data.columns = ['Surface_Area_um2', 'PG_Tension', 'PM_Tension', 'Total_Tension']

# We need to associate pressure values with surface area
# We'll use the total tension as a common reference to map between the two datasets

# Create a mapping between surface area and tension
area_to_tension = dict(zip(area_data['Surface_Area_um2'], area_data['Total_Tension']))
tension_to_pressure = dict(zip(pressure_data['Total_Tension'], pressure_data['Pressure_atm']))

# Function to find the closest key in a dictionary
def find_closest_key(dictionary, value):
    return min(dictionary.keys(), key=lambda k: abs(k - value))

# Create a new dataframe with pressure and corresponding surface area
volume_pressure_data = pd.DataFrame()
volume_pressure_data['Pressure_atm'] = pressure_data['Pressure_atm']

# Find corresponding surface areas for each pressure
surface_areas = []
for tension in pressure_data['Total_Tension']:
    closest_tension = find_closest_key(area_to_tension, tension)
    closest_area = find_closest_key(area_data['Surface_Area_um2'], closest_tension)
    surface_areas.append(closest_area)

volume_pressure_data['Surface_Area_um2'] = surface_areas

# Calculate radius from surface area (assuming spherical shape)
# A = 4πr², so r = sqrt(A/(4π))
volume_pressure_data['Radius_um'] = np.sqrt(volume_pressure_data['Surface_Area_um2'] / (4 * math.pi))

# Calculate volume from radius
# V = (4/3)πr³
volume_pressure_data['Volume_um3'] = (4/3) * math.pi * volume_pressure_data['Radius_um']**3

# Calculate the initial volume (at lowest pressure)
V0 = volume_pressure_data['Volume_um3'].min()
volume_pressure_data['Volume_Change'] = (volume_pressure_data['Volume_um3'] - V0) / V0 * 100  # percentage change

# Sort by pressure for plotting
volume_pressure_data = volume_pressure_data.sort_values('Pressure_atm')

# Create volume vs pressure plot
plt.figure(figsize=(10, 6))
plt.plot(volume_pressure_data['Pressure_atm'], volume_pressure_data['Volume_um3'])
plt.xlabel('Pressure [atm]')
plt.ylabel('Volume [μm³]')
plt.title('Volume vs Pressure')
plt.grid(True)
plt.savefig('volume_vs_pressure.png')

# Create percentage volume change vs pressure plot
plt.figure(figsize=(10, 6))
plt.plot(volume_pressure_data['Pressure_atm'], volume_pressure_data['Volume_Change'])
plt.xlabel('Pressure [atm]')
plt.ylabel('Volume Change [%]')
plt.title('Percentage Volume Change vs Pressure')
plt.grid(True)
plt.savefig('volume_change_vs_pressure.png')

# Calculate volume elasticity (dP/dV × V)
# First calculate dP/dV
dP = np.gradient(volume_pressure_data['Pressure_atm'].values)
dV = np.gradient(volume_pressure_data['Volume_um3'].values)
dP_dV = dP / dV
volume_pressure_data['Volume_Elasticity'] = dP_dV * volume_pressure_data['Volume_um3']

# Create volume elasticity vs pressure plot
plt.figure(figsize=(10, 6))
plt.plot(volume_pressure_data['Pressure_atm'], volume_pressure_data['Volume_Elasticity'])
plt.xlabel('Pressure [atm]')
plt.ylabel('Volume Elasticity [atm]')
plt.title('Volume Elasticity vs Pressure')
plt.grid(True)
plt.savefig('volume_elasticity_vs_pressure.png')

# Save volume analysis data to CSV
volume_pressure_data.to_csv('volume_pressure_analysis.csv', index=False)

print("Volume vs pressure analysis complete. Generated plots and saved data to 'volume_pressure_analysis.csv'")
print("\nFirst 5 rows of volume analysis data:")
print(volume_pressure_data.head())
