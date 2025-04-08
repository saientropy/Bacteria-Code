import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
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

# Function to create the 3D sphere with tension values
def create_sphere(radius, tension, layer_thickness):
    # Create a meshgrid for the sphere
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
    
    # Create color array based on tension (same value for all points)
    colors = np.ones_like(x) * tension
    
    return x, y, z, colors

# Create the figure and 3D axis
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Create initial inner sphere (PG layer)
inner_radius = radii[0]
inner_x, inner_y, inner_z, inner_colors = create_sphere(inner_radius, pressure_data['PG_Tension'][0], inner_layer_thickness)

# Create initial outer sphere (PM layer)
outer_radius = inner_radius + outer_layer_thickness
outer_x, outer_y, outer_z, outer_colors = create_sphere(outer_radius, pressure_data['PM_Tension'][0], outer_layer_thickness)

# Initial plots
inner_plot = ax.plot_surface(inner_x, inner_y, inner_z, facecolors=cmap(inner_colors/10), alpha=0.7, label='PG Layer')
outer_plot = ax.plot_surface(outer_x, outer_y, outer_z, facecolors=cmap(outer_colors/10), alpha=0.3, label='PM Layer')

# Add pressure text
pressure_text = ax.text2D(0.05, 0.95, f"Pressure: {pressures[0]:.2f} atm", transform=ax.transAxes)

# Add colorbar
norm = plt.Normalize(0, 10)  # Normalize to tension range 0-10
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, label='Tension [10^2 N/m]')

# Set axis labels
ax.set_xlabel('X [μm]')
ax.set_ylabel('Y [μm]')
ax.set_zlabel('Z [μm]')
ax.set_title('Vesicle Tension Animation')

# Function to update the animation
def update(frame):
    # Clear previous plots
    ax.collections = []
    
    # Update radius based on current frame (pressure)
    inner_radius = radii[frame]
    outer_radius = inner_radius + outer_layer_thickness
    
    # Get tension values for current frame
    pg_tension = pressure_data['PG_Tension'][frame]
    pm_tension = pressure_data['PM_Tension'][frame]
    
    # Create updated spheres
    inner_x, inner_y, inner_z, inner_colors = create_sphere(inner_radius, pg_tension, inner_layer_thickness)
    outer_x, outer_y, outer_z, outer_colors = create_sphere(outer_radius, pm_tension, outer_layer_thickness)
    
    # Plot updated spheres
    inner_plot = ax.plot_surface(inner_x, inner_y, inner_z, facecolors=cmap(pg_tension/10), alpha=0.7)
    outer_plot = ax.plot_surface(outer_x, outer_y, outer_z, facecolors=cmap(pm_tension/10), alpha=0.3)
    
    # Update pressure text
    pressure_text.set_text(f"Pressure: {pressures[frame]:.2f} atm")
    
    # Set consistent axis limits
    max_radius = max(radii) + outer_layer_thickness + 0.002
    ax.set_xlim(-max_radius, max_radius)
    ax.set_ylim(-max_radius, max_radius)
    ax.set_zlim(-max_radius, max_radius)
    
    return inner_plot, outer_plot, pressure_text

# Create animation
ani = animation.FuncAnimation(fig, update, frames=len(pressures), interval=50, blit=False)

# Save animation
ani.save('vesicle_tension_animation.mp4', writer='ffmpeg', fps=30, dpi=200)
print("Animation saved as 'vesicle_tension_animation.mp4'")

# Display final message
print("Animation creation complete!")
