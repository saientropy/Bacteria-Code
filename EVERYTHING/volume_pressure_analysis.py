"""
Analyses the relationship between volume and pressure using the combined
datasets. Computes derivatives and plots results.
"""

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

# Sort data by pressure and surface area for consistency
pressure_data = pressure_data.sort_values('Pressure_atm').reset_index(drop=True)
area_data = area_data.sort_values('Surface_Area_um2').reset_index(drop=True)

print("Data loaded successfully")

# DIRECT APPROACH: Calculate volume for each surface area
# For a sphere: A = 4πr² and V = (4/3)πr³
# Therefore r = sqrt(A/(4π)) and V = (4/3)π * [sqrt(A/(4π))]³

# Calculate radius from surface area
area_data['Radius_um'] = np.sqrt(area_data['Surface_Area_um2'] / (4 * np.pi))

# Calculate volume from radius
area_data['Volume_um3'] = (4/3) * np.pi * area_data['Radius_um']**3

print("Calculated volume from surface area")

# Now we need to map each pressure to a volume
# We'll create an interpolation function based on tension

# First, ensure we have a common range of tension values
min_tension = max(pressure_data['Total_Tension'].min(), area_data['Total_Tension'].min())
max_tension = min(pressure_data['Total_Tension'].max(), area_data['Total_Tension'].max())

print(f"Tension range for interpolation: {min_tension} to {max_tension}")

# Subset data to common tension range
pressure_subset = pressure_data[(pressure_data['Total_Tension'] >= min_tension) & 
                               (pressure_data['Total_Tension'] <= max_tension)]
                               
area_subset = area_data[(area_data['Total_Tension'] >= min_tension) & 
                        (area_data['Total_Tension'] <= max_tension)]

print(f"Subset size - Pressure: {len(pressure_subset)}, Area: {len(area_subset)}")

# Create interpolation functions
from scipy.interpolate import interp1d

# Function to create monotonic data for interpolation
def make_monotonic(x, y):
    # Sort by x values
    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices]
    
    # Remove duplicates in x by averaging corresponding y values
    unique_x, indices = np.unique(x_sorted, return_index=True)
    
    if len(unique_x) < 2:
        print("Warning: Not enough unique values for interpolation")
        return x, y
    
    # For each unique x, average all corresponding y values
    unique_y = np.array([np.mean(y_sorted[x_sorted == x_val]) for x_val in unique_x])
    
    return unique_x, unique_y

# Create monotonic data for interpolation
tension_p, pressure = make_monotonic(
    pressure_subset['Total_Tension'].values,
    pressure_subset['Pressure_atm'].values
)

tension_a, volume = make_monotonic(
    area_subset['Total_Tension'].values,
    area_subset['Volume_um3'].values
)

print(f"Monotonic data - Tension-Pressure: {len(tension_p)}, Tension-Volume: {len(tension_a)}")

# Now we can create interpolation functions (if we have enough data points)
if len(tension_p) >= 2 and len(tension_a) >= 2:
    # Create interpolation functions
    tension_to_pressure = interp1d(tension_p, pressure, bounds_error=False, fill_value="extrapolate")
    tension_to_volume = interp1d(tension_a, volume, bounds_error=False, fill_value="extrapolate")
    
    # Create a range of tension values within the common range
    tension_range = np.linspace(min_tension, max_tension, 100)
    
    # Map tensions to pressure and volume
    pressures = tension_to_pressure(tension_range)
    volumes = tension_to_volume(tension_range)
    
    # Create a dataframe with the results
    volume_pressure_df = pd.DataFrame({
        'Tension': tension_range,
        'Pressure_atm': pressures,
        'Volume_um3': volumes
    })
    
    # Sort by pressure
    volume_pressure_df = volume_pressure_df.sort_values('Pressure_atm')
    
    # Calculate initial volume and volume change
    initial_volume = volume_pressure_df['Volume_um3'].iloc[0]
    volume_pressure_df['Volume_Change_percent'] = ((volume_pressure_df['Volume_um3'] - initial_volume) / 
                                                  initial_volume * 100)
    
    print("Calculated volume-pressure relationship successfully")
    
    # Create volume vs pressure plot
    plt.figure(figsize=(10, 6))
    plt.plot(volume_pressure_df['Pressure_atm'], volume_pressure_df['Volume_um3'])
    plt.xlabel('Pressure [atm]')
    plt.ylabel('Volume [μm³]')
    plt.title('Volume vs Pressure')
    plt.grid(True)
    plt.savefig('volume_vs_pressure.png')
    
    # Create volume change vs pressure plot
    plt.figure(figsize=(10, 6))
    plt.plot(volume_pressure_df['Pressure_atm'], volume_pressure_df['Volume_Change_percent'])
    plt.xlabel('Pressure [atm]')
    plt.ylabel('Volume Change [%]')
    plt.title('Percentage Volume Change vs Pressure')
    plt.grid(True)
    plt.savefig('volume_change_vs_pressure.png')
    
    # Save results to CSV
    volume_pressure_df.to_csv('volume_vs_pressure.csv', index=False)
    
    print("Volume vs pressure analysis complete. Generated plots and saved data.")
    print("\nFirst 5 rows of volume vs pressure data:")
    print(volume_pressure_df.head())
else:
    print("Error: Not enough unique data points for interpolation.")
    print(f"Tension-Pressure points: {len(tension_p)}")
    print(f"Tension-Volume points: {len(tension_a)}")
    
    # Try a direct approach without interpolation
    # Calculate radius directly from pressure using Young-Laplace (T = PR/2)
    print("\nAttempting direct calculation using Young-Laplace equation...")
    pressure_data['Radius_um'] = 2 * pressure_data['Total_Tension'] * 100 / (pressure_data['Pressure_atm'] * 101325) * 1e6
    pressure_data['Volume_um3'] = (4/3) * np.pi * pressure_data['Radius_um']**3
    
    # Calculate volume change
    initial_volume = pressure_data['Volume_um3'].iloc[0]
    pressure_data['Volume_Change_percent'] = ((pressure_data['Volume_um3'] - initial_volume) / 
                                             initial_volume * 100)
    
    # Create plots
    plt.figure(figsize=(10, 6))
    plt.plot(pressure_data['Pressure_atm'], pressure_data['Volume_um3'])
    plt.xlabel('Pressure [atm]')
    plt.ylabel('Volume [μm³]')
    plt.title('Volume vs Pressure (Direct Calculation)')
    plt.grid(True)
    plt.savefig('volume_vs_pressure_direct.png')
    
    # Save results
    pressure_data[['Pressure_atm', 'Total_Tension', 'Radius_um', 'Volume_um3', 'Volume_Change_percent']].to_csv(
        'volume_vs_pressure_direct.csv', index=False)
    
    print("Direct calculation complete. Generated plot and saved data.")
    print("\nFirst 5 rows of volume vs pressure data (direct calculation):")
    print(pressure_data[['Pressure_atm', 'Radius_um', 'Volume_um3', 'Volume_Change_percent']].head())