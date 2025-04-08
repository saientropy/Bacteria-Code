import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import math

# Load the data
pressure_data = pd.read_csv('plot_data_first_1.csv')
area_data = pd.read_csv('plot_data_second_1.csv')

# Rename columns for easier access
pressure_data.columns = ['Pressure_atm', 'PM_Tension', 'PG_Tension', 'Total_Tension']
area_data.columns = ['Surface_Area_um2', 'PG_Tension', 'PM_Tension', 'Total_Tension']

# Calculate radius from surface area (assuming spherical shape)
# A = 4πr², so r = sqrt(A/(4π))
area_data['Radius_um'] = np.sqrt(area_data['Surface_Area_um2'] / (4 * math.pi))

# Create a mapping between pressure and radius
pressures = pressure_data['Pressure_atm'].values
total_tensions = pressure_data['Total_Tension'].values

# Create a function to find closest tension value in area_data
def find_radius_for_tension(tension):
    idx = (area_data['Total_Tension'] - tension).abs().idxmin()
    return area_data.loc[idx, 'Radius_um']

# Map pressures to radii
radii = []
for tension in total_tensions:
    radii.append(find_radius_for_tension(tension))

# Convert to numpy array for easier indexing
radii = np.array(radii)

# Define layer thicknesses (in μm)
inner_layer_thickness = 0.001  # 1 nm converted to μm
outer_layer_thickness = 0.006  # 6 nm converted to μm

# Create a custom colormap for tension (blue to red)
cmap = LinearSegmentedColormap.from_list("tension_map", ["blue", "green", "yellow", "red"])

# Create the figure with two subplots (vesicle and tension plot)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

# Function to create circle patches for the vesicle layers
def create_circle(radius, center=(0,0)):
    theta = np.linspace(0, 2*np.pi, 100)
    x = radius * np.cos(theta) + center[0]
    y = radius * np.sin(theta) + center[1]
    return x, y

# Initial inner circle (PG layer)
inner_radius = radii[0]
inner_x, inner_y = create_circle(inner_radius)
inner_circle = ax1.fill(inner_x, inner_y, color=cmap(pressure_data['PG_Tension'][0]/10), alpha=0.7, label='PG Layer')

# Initial outer circle (PM layer)
outer_radius = inner_radius + outer_layer_thickness
outer_x, outer_y = create_circle(outer_radius)
outer_circle = ax1.fill(outer_x, outer_y, color=cmap(pressure_data['PM_Tension'][0]/10), alpha=0.3, label='PM Layer')

# Add PG and PM text labels
pg_text = ax1.text(0, 0, 'PG', ha='center', va='center', fontsize=12, fontweight='bold')
pm_text = ax1.text(0, inner_radius + outer_layer_thickness/2, 'PM', ha='center', va='center', fontsize=12, fontweight='bold')

# Set up the tension vs pressure plot
ax2.plot(pressure_data['Pressure_atm'], pressure_data['PG_Tension'], 'k-', label='PG')
ax2.plot(pressure_data['Pressure_atm'], pressure_data['PM_Tension'], 'b-', label='PM')
ax2.plot(pressure_data['Pressure_atm'], pressure_data['Total_Tension'], 'r-', label='Total')
ax2.axhline(y=1.5, color='g', linestyle='--', label='Rupture')
ax2.set_xlabel('Pressure [atm]')
ax2.set_ylabel('Tension [10^2 N/m]')
ax2.set_title('Tension vs Pressure')
ax2.legend()
ax2.grid(True)

# Add a vertical line to show current pressure
pressure_line = ax2.axvline(x=pressures[0], color='black', linestyle='-')

# Add pressure text
pressure_text = ax1.text(0.05, 0.95, f"Pressure: {pressures[0]:.2f} atm", transform=ax1.transAxes)

# Add tension values text
pg_tension_text = ax1.text(0.05, 0.90, f"PG Tension: {pressure_data['PG_Tension'][0]:.2f}", transform=ax1.transAxes)
pm_tension_text = ax1.text(0.05, 0.85, f"PM Tension: {pressure_data['PM_Tension'][0]:.2f}", transform=ax1.transAxes)
total_tension_text = ax1.text(0.05, 0.80, f"Total Tension: {pressure_data['Total_Tension'][0]:.2f}", transform=ax1.transAxes)

# Add colorbar
norm = plt.Normalize(0, 10)  # Normalize to tension range 0-10
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax1, label='Tension [10^2 N/m]')

# Set axis for vesicle plot
ax1.set_xlabel('X [μm]')
ax1.set_ylabel('Y [μm]')
ax1.set_title('Vesicle Cross-Section')
ax1.set_aspect('equal')
ax1.grid(True)

# Set consistent axis limits for vesicle plot
max_radius = max(radii) + outer_layer_thickness + 0.002
ax1.set_xlim(-max_radius, max_radius)
ax1.set_ylim(-max_radius, max_radius)

# Function to update the animation
def update(frame):
    # Clear previous circles
    for collection in [inner_circle[0], outer_circle[0]]:
        collection.remove()
    
    # Update radius based on current frame (pressure)
    inner_radius = radii[frame]
    outer_radius = inner_radius + outer_layer_thickness
    
    # Get tension values for current frame
    pg_tension = pressure_data['PG_Tension'][frame]
    pm_tension = pressure_data['PM_Tension'][frame]
    total_tension = pressure_data['Total_Tension'][frame]
    
    # Update circles
    inner_x, inner_y = create_circle(inner_radius)
    inner_circle[0] = ax1.fill(inner_x, inner_y, color=cmap(pg_tension/10), alpha=0.7, label='PG Layer')[0]
    
    outer_x, outer_y = create_circle(outer_radius)
    outer_circle[0] = ax1.fill(outer_x, outer_y, color=cmap(pm_tension/10), alpha=0.3, label='PM Layer')[0]
    
    # Update text positions
    pg_text.set_position((0, 0))
    pm_text.set_position((0, inner_radius + outer_layer_thickness/2))
    
    # Update pressure line
    pressure_line.set_xdata([pressures[frame], pressures[frame]])
    
    # Update text displays
    pressure_text.set_text(f"Pressure: {pressures[frame]:.2f} atm")
    pg_tension_text.set_text(f"PG Tension: {pg_tension:.2f}")
    pm_tension_text.set_text(f"PM Tension: {pm_tension:.2f}")
    total_tension_text.set_text(f"Total Tension: {total_tension:.2f}")
    
    return inner_circle, outer_circle, pressure_line, pressure_text, pg_tension_text, pm_tension_text, total_tension_text, pg_text, pm_text

# Create animation
ani = animation.FuncAnimation(fig, update, frames=len(pressures), interval=50, blit=False)

# Save animation
ani.save('vesicle_tension_animation_2d.mp4', writer='ffmpeg', fps=30, dpi=200)
print("Animation saved as 'vesicle_tension_animation_2d.mp4'")

# Display final message
print("2D Animation creation complete!")
