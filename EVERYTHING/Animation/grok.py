"""
Minimal example showing how to animate a rotating vector plot. Serves as a
sandbox for experimenting with Matplotlib's animation tools.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

# Load the data files
pressure_data = pd.read_csv('plot_data_first_1.csv')
area_data = pd.read_csv('plot_data_second_1.csv')

# Rename columns for easier access
pressure_data.columns = ['Pressure_atm', 'PM_Tension', 'PG_Tension', 'Total_Tension']
area_data.columns = ['Surface_Area', 'PG_Tension', 'PM_Tension', 'Total_Tension']

# Constants for the bacteria model
outer_thickness = 0.008  # 8nm converted to μm
inner_thickness = 0.001  # 1nm converted to μm

# Function to calculate radius from surface area (assuming spherical shape)
def surface_area_to_radius(area):
    """Convert surface area to radius, assuming a sphere"""
    return np.sqrt(area / (4 * np.pi))

# Add radius column to area_data
area_data['Radius'] = surface_area_to_radius(area_data['Surface_Area'])

# Function to create a sphere with 1/4 cutout
def create_cutout_sphere(radius, resolution=50):
    """Create a sphere with a larger cutout to better see inside."""
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
    
    # Create a larger cutout - 1/4 of the sphere
    mask = np.zeros_like(x, dtype=bool)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i, j] > 0 and y[i, j] > 0:
                mask[i, j] = True
    
    # Apply mask
    x_cut = x.copy()
    y_cut = y.copy()
    z_cut = z.copy()
    
    x_cut[mask] = np.nan
    y_cut[mask] = np.nan
    z_cut[mask] = np.nan
    
    return x_cut, y_cut, z_cut

# Create a function for wireframe edges to show layer boundaries
def create_wireframe_edges(radius, resolution=25):
    """Create edge lines that will highlight layer boundaries"""
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    
    # Create full sphere coordinates
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
    
    # Apply cutout mask
    mask = np.zeros_like(x, dtype=bool)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i, j] > 0 and y[i, j] > 0:
                mask[i, j] = True
    
    # Create empty arrays for edge coordinates
    edge_x, edge_y, edge_z = [], [], []
    
    # Add coordinates for cutout edge - vertical edge (constant x)
    idx_x = np.argmin(np.abs(x[:, 0]))  # Find index closest to x=0
    for j in range(x.shape[1]):
        if not mask[idx_x, j]:
            edge_x.append(x[idx_x, j])
            edge_y.append(y[idx_x, j])
            edge_z.append(z[idx_x, j])
    
    # Add coordinates for cutout edge - horizontal edge (constant y)
    idx_y = np.argmin(np.abs(y[0, :]))  # Find index closest to y=0
    for i in range(x.shape[0]):
        if not mask[i, idx_y]:
            edge_x.append(x[i, idx_y])
            edge_y.append(y[i, idx_y])
            edge_z.append(z[i, idx_y])
    
    return edge_x, edge_y, edge_z

# Function to get tension values for a given area
def get_tension_for_area(area_val, area_df=area_data):
    """Get the closest PG and PM tension values for a given surface area."""
    idx = (area_df['Surface_Area'] - area_val).abs().idxmin()
    return area_df.loc[idx, 'PG_Tension'], area_df.loc[idx, 'PM_Tension']

# Function to get pressure for a given area
def get_pressure_for_area(area_val, area_df=area_data, pressure_df=pressure_data):
    """Get the closest pressure value for a given surface area."""
    pg_tension, pm_tension = get_tension_for_area(area_val)
    
    tension_distance = (
        (pressure_df['PG_Tension'] - pg_tension)**2 + 
        (pressure_df['PM_Tension'] - pm_tension)**2
    )
    idx = tension_distance.idxmin()
    return pressure_df.loc[idx, 'Pressure_atm']

# Calculate the maximum possible radius for fixed axes
max_possible_radius = (surface_area_to_radius(area_data['Surface_Area'].max()) + 
                      inner_thickness + outer_thickness) * 1.5

# Create figure with better proportions
fig = plt.figure(figsize=(14, 9))

# Create a centered axes for the 3D plot with space on left for info
ax = fig.add_subplot(111, projection='3d')
ax.set_position([0.25, 0.05, 0.70, 0.85])  # Move plot to the right to make space for info on left

# Create DISTINCT colormaps for the two layers
pg_cmap = cm.Blues_r     # Blue for PG layer (outer)
pm_cmap = cm.Reds_r      # Red for PM layer (inner)

# Set up normalization for the color scales
pg_norm = mpl.colors.Normalize(vmin=0, vmax=area_data['PG_Tension'].max())
pm_norm = mpl.colors.Normalize(vmin=0, vmax=area_data['PM_Tension'].max())

# Add information texts on the LEFT side
pressure_text = fig.text(0.12, 0.9, "Pressure: 0.00 atm", 
                 fontsize=12, weight='bold',
                 bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3', edgecolor='gray'),
                 ha='center')

# Other metric texts - stacked vertically on the left
surface_area_text = fig.text(0.12, 0.82, "Surface Area: 0.00 µ²", 
                    fontsize=12,
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3', edgecolor='gray'),
                    ha='center')

pg_tension_text = fig.text(0.12, 0.74, "PG Tension: 0.00 [10² N/m]", 
                  fontsize=12, color='darkblue',
                  bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3', edgecolor='gray'),
                  ha='center')

pm_tension_text = fig.text(0.12, 0.66, "PM Tension: 0.00 [10² N/m]", 
                  fontsize=12, color='darkred',
                  bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3', edgecolor='gray'),
                  ha='center')

# Add colorbars on opposite sides
cax1 = fig.add_axes([0.12, 0.3, 0.03, 0.25])  # PG colorbar on LEFT
cax2 = fig.add_axes([0.85, 0.3, 0.03, 0.25])  # PM colorbar on RIGHT

cbar1 = plt.colorbar(cm.ScalarMappable(norm=pg_norm, cmap=pg_cmap), cax=cax1)
cbar2 = plt.colorbar(cm.ScalarMappable(norm=pm_norm, cmap=pm_cmap), cax=cax2)

cbar1.set_label('PG Tension [10² N/m]', fontsize=10)
cbar2.set_label('PM Tension [10² N/m]', fontsize=10)

# Add layer labels on proper sides
fig.text(0.12, 0.56, 'PG Layer (8nm)', fontsize=10, ha='center', color='darkblue', weight='bold')
fig.text(0.85, 0.56, 'PM Layer (1nm)', fontsize=10, ha='center', color='darkred', weight='bold')

# Function to update animation
def update(frame):
    ax.clear()
    
    # Get current data
    area_val = area_data.iloc[frame]['Surface_Area']
    radius = area_data.iloc[frame]['Radius']
    pg_tension = area_data.iloc[frame]['PG_Tension']
    pm_tension = area_data.iloc[frame]['PM_Tension']
    pressure = get_pressure_for_area(area_val)
    
    # Calculate radii for both layers (inner and outer surface of each layer)
    # PG layer (outer)
    pg_inner_radius = radius + inner_thickness  # Inner surface of PG layer
    pg_outer_radius = radius + inner_thickness + outer_thickness  # Outer surface of PG layer
    
    # PM layer (inner)
    pm_inner_radius = radius  # Inner surface of PM layer
    pm_outer_radius = radius + inner_thickness  # Outer surface of PM layer
    
    # Create main surfaces
    x_pg, y_pg, z_pg = create_cutout_sphere(pg_outer_radius, resolution=50)
    x_pm, y_pm, z_pm = create_cutout_sphere(pm_outer_radius, resolution=50)
    
    # Create edge highlighting lines
    edge_x_pg, edge_y_pg, edge_z_pg = create_wireframe_edges(pg_outer_radius)
    edge_x_pm, edge_y_pm, edge_z_pm = create_wireframe_edges(pm_outer_radius)
    
    # Plot inner sphere (PM layer) - red colors
    pm_surf = ax.plot_surface(
        x_pm, y_pm, z_pm,
        color='red', alpha=0.9,  # Fairly opaque
        rstride=1, cstride=1,
        linewidth=0, antialiased=True
    )
    
    # Color PM surface based on PM tension using RED colormap
    pm_color = pm_cmap(pm_norm(pm_tension))
    pm_surf.set_facecolor(pm_color)
    
    # Plot outer sphere (PG layer) with more transparency - blue colors
    pg_surf = ax.plot_surface(
        x_pg, y_pg, z_pg,
        color='blue', alpha=0.6,  # Semi-transparent
        rstride=1, cstride=1,
        linewidth=0, antialiased=True
    )
    
    # Color PG surface based on PG tension using BLUE colormap
    pg_color = pg_cmap(pg_norm(pg_tension))
    pg_surf.set_facecolor(pg_color)
    
    # Add wireframe edges to highlight layers at the cutout (outer PG)
    ax.plot(edge_x_pg, edge_y_pg, edge_z_pg, color='black', linewidth=1.5, alpha=0.6)
    
    # Add wireframe edges for PM layer at the cutout
    ax.plot(edge_x_pm, edge_y_pm, edge_z_pm, color='black', linewidth=1.5, alpha=0.6)
    
    # Add additional cross-section lines to clearly show layer thickness at the cutout
    # For PG layer boundary (vertical cross-section at y=0)
    x_range = np.linspace(0, pg_outer_radius, 50)
    z_range = np.zeros_like(x_range)
    y_slice = np.zeros_like(x_range)
    
    # Draw cross-section line for outer PG surface
    ax.plot(x_range, y_slice, z_range, color='black', linewidth=1.0)
    
    # Draw cross-section line for PM-PG interface
    x_range_inner = np.linspace(0, pm_outer_radius, 50)
    ax.plot(x_range_inner, y_slice, z_range, color='black', linewidth=1.0)
    
    # Set fixed axis limits - ALWAYS THE SAME regardless of frame
    ax.set_xlim([-max_possible_radius, max_possible_radius])
    ax.set_ylim([-max_possible_radius, max_possible_radius])
    ax.set_zlim([-max_possible_radius, max_possible_radius])
    
    # Set fixed aspect ratio to prevent axis scaling
    ax.set_box_aspect([1, 1, 1])
    
    # Update text elements
    pressure_text.set_text(f"Pressure: {pressure:.2f} atm")
    surface_area_text.set_text(f"Surface Area: {area_val:.2f} µ²")
    pg_tension_text.set_text(f"PG Tension: {pg_tension:.2f} [10² N/m]")
    pm_tension_text.set_text(f"PM Tension: {pm_tension:.2f} [10² N/m]")
    
    # Set labels
    ax.set_xlabel('X [µm]', fontsize=10)
    ax.set_ylabel('Y [µm]', fontsize=10)
    ax.set_zlabel('Z [µm]', fontsize=10)
    ax.set_title('Bacterial Cell Expansion', fontsize=14, y=1.0)
    
    # Improve visibility of the layer distinction by adding annotation
    arrow_length = 0.2 * max_possible_radius
    
    # Annotate PG layer
    pg_arrow_x = 0.6 * max_possible_radius
    pg_arrow_y = 0
    pg_arrow_z = 0
    ax.text(pg_arrow_x, pg_arrow_y, pg_arrow_z, "PG Layer", color='blue', fontsize=9)
    
    # Annotate PM layer
    pm_arrow_x = 0.3 * max_possible_radius
    pm_arrow_y = 0
    pm_arrow_z = 0
    ax.text(pm_arrow_x, pm_arrow_y, pm_arrow_z, "PM Layer", color='red', fontsize=9)
    
    # Adjust the view angle for better visualization of the cutout
    ax.view_init(elev=20, azim=30)  # Lower elevation angle to see inside better
    
    return [pg_surf, pm_surf]

# Create animation with better frame rate for smoother rotation
frames = range(0, len(area_data), 2)  # Less skipping for smoother animation
anim = animation.FuncAnimation(
    fig, update, frames=frames,
    interval=70, blit=False
)

# Set the first frame
update(0)

# Show the plot
plt.show()

# Uncomment to save animation
# anim.save('bacteria_expansion_improved.mp4', writer='ffmpeg', fps=25, dpi=150)

print("Animation displayed. Use mouse to rotate the model.")