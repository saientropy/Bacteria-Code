import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

# Load the CSV files
pressure_data = pd.read_csv('plot_data_first_1.csv')
area_data = pd.read_csv('plot_data_second_1.csv')

# Rename columns for easier access (based on your CSV info)
pressure_data.columns = ['Pressure_atm', 'PM_Tension', 'PG_Tension', 'Total_Tension']
area_data.columns = ['Surface_Area_um2', 'PG_Tension', 'PM_Tension', 'Total_Tension']

print("Data loaded successfully")

# Sort both dataframes to ensure consistent ordering
pressure_data = pressure_data.sort_values('Pressure_atm').reset_index(drop=True)
area_data = area_data.sort_values('Surface_Area_um2').reset_index(drop=True)

print(f"Number of data points - Pressure: {len(pressure_data)}, Area: {len(area_data)}")

# Calculate strain (ΔA/A₀) where A₀ is the minimum surface area
A0 = area_data['Surface_Area_um2'].min()
print(f"Initial surface area (A0): {A0} μm²")

area_data['Strain'] = (area_data['Surface_Area_um2'] - A0) / A0
print("Strain calculation complete")

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
print("Tension vs Strain plot created")

# Now we need to map strain to pressure
# We'll use the tension values as the common link between datasets

# Create a mapping from tension to pressure (using only monotonically increasing portions)
def create_monotonic_mapping(x, y):
    # Sort data
    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices]
    
    # Find unique x values and average corresponding y values
    unique_x = []
    unique_y = []
    
    for i in range(len(x_sorted)):
        if i == 0 or x_sorted[i] != x_sorted[i-1]:
            unique_x.append(x_sorted[i])
            unique_y.append(y_sorted[i])
        else:
            # Average y values for same x
            unique_y[-1] = (unique_y[-1] + y_sorted[i]) / 2
    
    return np.array(unique_x), np.array(unique_y)

# Create monotonic mapping between tension and pressure
print("Creating tension-pressure mapping...")
total_tensions_p, pressures = create_monotonic_mapping(
    pressure_data['Total_Tension'].values,
    pressure_data['Pressure_atm'].values
)

# Create monotonic mapping between tension and strain
print("Creating tension-strain mapping...")
total_tensions_s, strains = create_monotonic_mapping(
    area_data['Total_Tension'].values,
    area_data['Strain'].values
)

# Find common tension range
min_tension = max(total_tensions_p.min(), total_tensions_s.min())
max_tension = min(total_tensions_p.max(), total_tensions_s.max())
print(f"Common tension range: {min_tension} to {max_tension}")

# Filter to common tension range
pressure_tension_indices = (total_tensions_p >= min_tension) & (total_tensions_p <= max_tension)
strain_tension_indices = (total_tensions_s >= min_tension) & (total_tensions_s <= max_tension)

t_p_filtered = total_tensions_p[pressure_tension_indices]
p_filtered = pressures[pressure_tension_indices]
t_s_filtered = total_tensions_s[strain_tension_indices]
s_filtered = strains[strain_tension_indices]

print(f"Filtered data points - Pressure mapping: {len(t_p_filtered)}, Strain mapping: {len(t_s_filtered)}")

if len(t_p_filtered) > 1 and len(t_s_filtered) > 1:
    # Create interpolation functions
    from scipy.interpolate import interp1d
    
    print("Creating interpolation functions...")
    tension_to_pressure = interp1d(t_p_filtered, p_filtered, bounds_error=False, fill_value="extrapolate")
    tension_to_strain = interp1d(t_s_filtered, s_filtered, bounds_error=False, fill_value="extrapolate")
    
    # Create a range of tension values within the common range
    tension_range = np.linspace(min_tension, max_tension, 100)
    
    # Use the functions to map tensions to pressure and strain
    corresponding_pressures = tension_to_pressure(tension_range)
    corresponding_strains = tension_to_strain(tension_range)
    
    # Create a dataframe with pressure-strain pairs
    strain_pressure_df = pd.DataFrame({
        'Tension': tension_range,
        'Pressure_atm': corresponding_pressures,
        'Strain': corresponding_strains
    })
    
    # Sort by pressure
    strain_pressure_df = strain_pressure_df.sort_values('Pressure_atm')
    
    print("Strain-pressure mapping created successfully")
    
    # Create strain vs pressure plot
    plt.figure(figsize=(10, 6))
    plt.plot(strain_pressure_df['Pressure_atm'], strain_pressure_df['Strain'])
    plt.xlabel('Pressure [atm]')
    plt.ylabel('Strain (ΔA/A₀)')
    plt.title('Strain vs Pressure')
    plt.grid(True)
    plt.savefig('strain_vs_pressure.png')
    
    # Calculate strain rate (derivative of strain with respect to pressure)
    strain_rate = np.gradient(strain_pressure_df['Strain'].values, strain_pressure_df['Pressure_atm'].values)
    strain_pressure_df['Strain_Rate'] = strain_rate
    
    # Remove any NaN values that might have been introduced
    strain_pressure_df = strain_pressure_df.dropna()
    
    # Create strain rate vs pressure plot
    plt.figure(figsize=(10, 6))
    plt.plot(strain_pressure_df['Pressure_atm'], strain_pressure_df['Strain_Rate'])
    plt.xlabel('Pressure [atm]')
    plt.ylabel('Strain Rate (dε/dP) [atm⁻¹]')
    plt.title('Strain Rate vs Pressure')
    plt.grid(True)
    plt.savefig('strain_rate_vs_pressure.png')
    
    # Save results to CSV
    strain_pressure_df.to_csv('strain_analysis.csv', index=False)
    
    print("Strain analysis complete. Generated plots and saved data.")
    print("\nFirst 5 rows of strain analysis data:")
    print(strain_pressure_df.head())
else:
    print("Error: Not enough data points for interpolation in the common tension range.")
    print("Using direct Young-Laplace approach to relate strain and pressure...")
    
    # Calculate radius from surface area
    area_data['Radius_um'] = np.sqrt(area_data['Surface_Area_um2'] / (4 * np.pi))
    
    # Use Young-Laplace to estimate pressure: P = 2T/R
    # Convert tension from 10^2 N/m to N/m, and radius from μm to m
    area_data['Pressure_atm'] = 2 * area_data['Total_Tension'] * 100 / (area_data['Radius_um'] * 1e-6) / 101325
    
    # Sort by pressure
    area_data = area_data.sort_values('Pressure_atm')
    
    # Create strain vs pressure plot
    plt.figure(figsize=(10, 6))
    plt.plot(area_data['Pressure_atm'], area_data['Strain'])
    plt.xlabel('Pressure [atm]')
    plt.ylabel('Strain (ΔA/A₀)')
    plt.title('Strain vs Pressure (Direct Calculation)')
    plt.grid(True)
    plt.savefig('strain_vs_pressure_direct.png')
    
    # Calculate strain rate
    area_data['Strain_Rate'] = np.gradient(area_data['Strain'].values, area_data['Pressure_atm'].values)
    
    # Remove any NaN values
    area_data = area_data.dropna(subset=['Strain_Rate'])
    
    # Create strain rate vs pressure plot
    plt.figure(figsize=(10, 6))
    plt.plot(area_data['Pressure_atm'], area_data['Strain_Rate'])
    plt.xlabel('Pressure [atm]')
    plt.ylabel('Strain Rate (dε/dP) [atm⁻¹]')
    plt.title('Strain Rate vs Pressure (Direct Calculation)')
    plt.grid(True)
    plt.savefig('strain_rate_vs_pressure_direct.png')
    
    # Save results
    area_data[['Pressure_atm', 'Surface_Area_um2', 'Strain', 'Strain_Rate', 'Total_Tension']].to_csv(
        'strain_analysis_direct.csv', index=False)
    
    print("Direct calculation complete. Generated plots and saved data.")
    print("\nFirst 5 rows of strain analysis data (direct calculation):")
    print(area_data[['Pressure_atm', 'Surface_Area_um2', 'Strain', 'Strain_Rate', 'Total_Tension']].head())