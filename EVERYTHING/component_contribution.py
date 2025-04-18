import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
pressure_data = pd.read_csv('plot_data_first_1.csv')
area_data = pd.read_csv('plot_data_second_1.csv')

# Rename columns for easier access (based on your CSV info)
pressure_data.columns = ['Pressure_atm', 'PM_Tension', 'PG_Tension', 'Total_Tension']
area_data.columns = ['Surface_Area_um2', 'PG_Tension', 'PM_Tension', 'Total_Tension']

# Calculate relative contribution of each component to total tension
pressure_data['PM_Contribution'] = pressure_data['PM_Tension'] / pressure_data['Total_Tension'] * 100
pressure_data['PG_Contribution'] = pressure_data['PG_Tension'] / pressure_data['Total_Tension'] * 100

# Handle NaN or inf values (in case Total_Tension = 0)
pressure_data['PM_Contribution'] = pressure_data['PM_Contribution'].fillna(0)
pressure_data['PG_Contribution'] = pressure_data['PG_Contribution'].fillna(0)
pressure_data.replace([np.inf, -np.inf], 0, inplace=True)

# Create a stacked area plot showing component contributions vs pressure
plt.figure(figsize=(10, 6))
plt.stackplot(pressure_data['Pressure_atm'], 
              [pressure_data['PG_Contribution'], pressure_data['PM_Contribution']], 
              labels=['PG Component', 'PM Component'],
              colors=['black', 'blue'], alpha=0.7)
plt.xlabel('Pressure [atm]')
plt.ylabel('Contribution to Total Tension [%]')
plt.title('Relative Contribution of Components vs Pressure')
plt.legend(loc='upper right')
plt.xlim(0, 4)
plt.ylim(0, 100)
plt.grid(True)
plt.savefig('component_contribution_vs_pressure.png')

# Create a similar analysis for the area data
area_data['PM_Contribution'] = area_data['PM_Tension'] / area_data['Total_Tension'] * 100
area_data['PG_Contribution'] = area_data['PG_Tension'] / area_data['Total_Tension'] * 100

# Handle NaN or inf values
area_data['PM_Contribution'] = area_data['PM_Contribution'].fillna(0)
area_data['PG_Contribution'] = area_data['PG_Contribution'].fillna(0)
area_data.replace([np.inf, -np.inf], 0, inplace=True)

# Create stacked area plot for component contributions vs surface area
plt.figure(figsize=(10, 6))
plt.stackplot(area_data['Surface_Area_um2'], 
              [area_data['PG_Contribution'], area_data['PM_Contribution']], 
              labels=['PG Component', 'PM Component'],
              colors=['black', 'blue'], alpha=0.7)
plt.xlabel('Surface Area [μm²]')
plt.ylabel('Contribution to Total Tension [%]')
plt.title('Relative Contribution of Components vs Surface Area')
plt.legend(loc='upper right')
plt.xlim(15, 40)
plt.ylim(0, 100)
plt.grid(True)
plt.savefig('component_contribution_vs_area.png')

# Create a bar chart showing component values at specific pressure points
pressure_points = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
pm_values = []
pg_values = []

for p in pressure_points:
    # Find nearest pressure value
    idx = (pressure_data['Pressure_atm'] - p).abs().idxmin()
    pm_values.append(pressure_data.loc[idx, 'PM_Tension'])
    pg_values.append(pressure_data.loc[idx, 'PG_Tension'])

# Create grouped bar chart
x = np.arange(len(pressure_points))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x - width/2, pg_values, width, label='PG Component', color='black')
ax.bar(x + width/2, pm_values, width, label='PM Component', color='blue')

ax.set_xlabel('Pressure [atm]')
ax.set_ylabel('Tension [10^2 N/m]')
ax.set_title('Component Contributions at Different Pressures')
ax.set_xticks(x)
ax.set_xticklabels(pressure_points)
ax.legend()
ax.grid(True, alpha=0.3)

plt.savefig('component_comparison_bars.png')

# Save component contribution data to CSV
contribution_data = pressure_data[['Pressure_atm', 'PM_Tension', 'PG_Tension', 
                                  'Total_Tension', 'PM_Contribution', 'PG_Contribution']]
contribution_data.to_csv('component_contribution.csv', index=False)

print("Component contribution analysis complete. Generated plots and saved data to 'component_contribution.csv'")
print("\nFirst 5 rows of component contribution data:")
print(contribution_data.head())
