import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV files
pressure_data = pd.read_csv('plot_data_first_1.csv')
area_data = pd.read_csv('plot_data_second_1.csv')

# Rename columns for easier access (based on your CSV info)
pressure_data.columns = ['Pressure_atm', 'PM_Tension', 'PG_Tension', 'Total_Tension']
area_data.columns = ['Surface_Area_um2', 'PG_Tension', 'PM_Tension', 'Total_Tension']

# Calculate strain (ΔA/A₀) where A₀ is the minimum surface area
A0 = area_data['Surface_Area_um2'].min()
area_data['Strain'] = (area_data['Surface_Area_um2'] - A0) / A0

# Sort by surface area for better visualization
area_data = area_data.sort_values(by='Surface_Area_um2')

# Create strain vs tension plot
plt.figure(figsize=(10, 6))
plt.plot(area_data['Strain'], area_data['Total_Tension'], label='Total', color='red', linewidth=2)
plt.plot(area_data['Strain'], area_data['PG_Tension'], label='PG', color='black', linewidth=2)
plt.plot(area_data['Strain'], area_data['PM_Tension'], label='PM', color='blue', linewidth=2)
plt.axhline(y=1.5, color='green', linestyle='--', linewidth=1, label='Rupture Tension')

plt.xlabel('Strain (ΔA/A₀)')
plt.ylabel('Tension [10^2 N/m]')
plt.title('Tension vs Strain')
plt.grid(True)
plt.legend()
plt.savefig('tension_vs_strain.png')

# Calculate strain rate (derivative of strain with respect to pressure)
# First, we need to map pressure to strain

# Create a mapping between surface area and pressure
# We'll use the total tension values as the common link
tension_to_pressure = {}
for i, row in pressure_data.iterrows():
    tension_to_pressure[row['Total_Tension']] = row['Pressure_atm']

# Map tensions in area_data to corresponding pressures
pressures = []
for tension in area_data['Total_Tension']:
    # Find the nearest tension value in the pressure data
    closest_tension = min(tension_to_pressure.keys(), key=lambda x: abs(x - tension))
    pressures.append(tension_to_pressure[closest_tension])

area_data['Pressure_atm'] = pressures

# Sort by pressure for derivative calculation
area_data = area_data.sort_values(by='Pressure_atm')

# Calculate strain rate (dε/dP)
strain_rate = np.gradient(area_data['Strain'].values, area_data['Pressure_atm'].values)
area_data['Strain_Rate'] = strain_rate

# Create strain vs pressure plot
plt.figure(figsize=(10, 6))
plt.plot(area_data['Pressure_atm'], area_data['Strain'])
plt.xlabel('Pressure [atm]')
plt.ylabel('Strain (ΔA/A₀)')
plt.title('Strain vs Pressure')
plt.grid(True)
plt.savefig('strain_vs_pressure.png')

# Create strain rate vs pressure plot
plt.figure(figsize=(10, 6))
plt.plot(area_data['Pressure_atm'], area_data['Strain_Rate'])
plt.xlabel('Pressure [atm]')
plt.ylabel('Strain Rate (dε/dP) [atm⁻¹]')
plt.title('Strain Rate vs Pressure')
plt.grid(True)
plt.savefig('strain_rate_vs_pressure.png')

# Save results to CSV
strain_data = area_data[['Pressure_atm', 'Surface_Area_um2', 'Strain', 'Strain_Rate', 'Total_Tension']]
strain_data.to_csv('strain_analysis.csv', index=False)

print("Strain analysis complete. Generated plots and saved data to 'strain_analysis.csv'")
print("\nFirst 5 rows of strain analysis data:")
print(strain_data.head())
