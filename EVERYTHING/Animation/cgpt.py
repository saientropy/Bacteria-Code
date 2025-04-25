import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

# Set backend for better interactivity
import matplotlib
matplotlib.use('TkAgg')  # TkAgg backend supports better interactive features

# Load the data files
pressure_data = pd.read_csv('plot_data_first_1.csv')
area_data = pd.read_csv('plot_data_second_1.csv')

# Rename columns for easier access
pressure_data.columns = ['Pressure_atm', 'PM_Tension', 'PG_Tension', 'Total_Tension']
area_data.columns = ['Surface_Area', 'PG_Tension', 'PM_Tension', 'Total_Tension']

# Constants for the bacteria model - MUCH thicker for visibility
outer_thickness = 0.5  # 0.5 μm - thick enough to be clearly visible
inner_thickness = 0.3  # 0.3 μm - thick enough to be clearly visible

# Function to calculate radius from surface area (assuming spherical shape)
def surface_area_to_radius(area):
    """Convert surface area to radius, assuming a sphere"""
    return np.sqrt(area / (4 * np.pi))

# Add radius column to area_data
area_data['Radius'] = surface_area_to_radius(area_data['Surface_Area'])

# Function to create a sphere with 1/4 cutout - CAD style with clean edges
def create_cutout_sphere(radius, resolution=50):
    """Create a sphere with a clean cutout for CAD-like appearance"""
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
# Green-to-black gradient for PG layer
pg_cmap = mpl.colors.LinearSegmentedColormap.from_list('GreenToBlack', ['forestgreen', 'darkgreen', 'black'])

# Custom colormap for PM layer: white for values < 0.2, then red gradient
# We'll create this using a ListedColormap with boundaries
pm_colors = ['white']  # Start with white
pm_colors.extend(cm.Reds_r(np.linspace(0, 1, 256)))  # Add red gradient colors

# Create a BoundaryNorm to specify the ranges for each color
pm_bounds = [0, 0.2, area_data['PM_Tension'].max()]
pm_norm = mpl.colors.BoundaryNorm(pm_bounds, len(pm_colors))

# Set up normalization for the PG color scale
pg_norm = mpl.colors.Normalize(vmin=0, vmax=area_data['PG_Tension'].max())

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
                  fontsize=12, color='forestgreen',
                  bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3', edgecolor='gray'),
                  ha='center')

pm_tension_text = fig.text(0.12, 0.66, "PM Tension: 0.00 [10² N/m]", 
                  fontsize=12, color='darkred',
                  bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3', edgecolor='gray'),
                  ha='center')

# Add colorbars on opposite sides
cax1 = fig.add_axes([0.12, 0.3, 0.03, 0.25])  # PG colorbar on LEFT
cax2 = fig.add_axes([0.85, 0.3, 0.03, 0.25])  # PM colorbar on RIGHT

# Create custom colorbar for PG tension
cbar1 = plt.colorbar(cm.ScalarMappable(norm=pg_norm, cmap=pg_cmap), cax=cax1)
cbar1.set_label('PG Tension [10² N/m]', fontsize=10)

# Create custom colorbar for PM tension with white for values < 0.2
# We need to create a custom colormap for this
pm_levels = np.linspace(0, area_data['PM_Tension'].max(), 100)
# Custom colormap: white for values < 0.2, then transition to red gradient
white_red_cmap = mpl.colors.ListedColormap(['white'] + [cm.Reds_r(i) for i in np.linspace(0, 1, 99)])
bounds = [0, 0.2] + list(np.linspace(0.2, area_data['PM_Tension'].max(), 99))
norm = mpl.colors.BoundaryNorm(bounds, white_red_cmap.N)

# Create the PM colorbar
cbar2 = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=white_red_cmap), cax=cax2)
cbar2.set_label('PM Tension [10² N/m]', fontsize=10)

# Add a text marker to show the 0.2 threshold on the PM colorbar
cax2.text(1.5, 0.2 / area_data['PM_Tension'].max() * cax2.get_ylim()[1], 
          "0.2", color='black', fontsize=8, ha='left', va='center')

# Add layer labels on proper sides
fig.text(0.12, 0.56, 'PG Layer (8nm)', fontsize=10, ha='center', color='forestgreen', weight='bold')
fig.text(0.85, 0.56, 'PM Layer (1nm)', fontsize=10, ha='center', color='darkred', weight='bold')

# Global variable to track current frame
current_frame = 0

# Function to update animation
def update(frame):
    global current_frame
    current_frame = frame
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
    
    # Plot inner sphere (PM layer) with solid color
    pm_surf = ax.plot_surface(
        x_pm, y_pm, z_pm,
        color='red' if pm_tension >= 0.2 else 'white', 
        alpha=0.9,  # Fairly opaque
        rstride=1, cstride=1,
        linewidth=0, antialiased=True
    )
    
    # Plot outer sphere (PG layer) with green color
    pg_surf = ax.plot_surface(
        x_pg, y_pg, z_pg,
        color='forestgreen',
        alpha=0.7,  # Semi-transparent
        rstride=1, cstride=1,
        linewidth=0, antialiased=True
    )
    
    # Draw clean cross-section lines to show layer thickness - NO UNWANTED LINES
    # Only draw the essential cross-sections
    
    # Draw cross-section for PG layer (outer layer)
    x_pg_range = np.linspace(-pg_outer_radius, 0, 50)
    x_pg_range = x_pg_range[x_pg_range <= 0]  # Only keep left half for the cutout
    
    # Outer surface of PG
    y_pg_outer = np.zeros_like(x_pg_range)
    z_pg_outer = np.sqrt(pg_outer_radius**2 - x_pg_range**2)
    
    # Inner surface of PG
    y_pg_inner = np.zeros_like(x_pg_range)
    z_pg_inner = np.sqrt(pg_inner_radius**2 - x_pg_range**2)
    
    # Draw PG layer edges with green color
    ax.plot(x_pg_range, y_pg_outer, z_pg_outer, color='black', linewidth=1.0)
    ax.plot(x_pg_range, y_pg_inner, z_pg_inner, color='black', linewidth=1.0)
    
    # Draw cross-section for PM layer (inner layer)
    x_pm_range = np.linspace(-pm_outer_radius, 0, 50)
    x_pm_range = x_pm_range[x_pm_range <= 0]  # Only keep left half for the cutout
    
    # Outer surface of PM
    y_pm_outer = np.zeros_like(x_pm_range)
    z_pm_outer = np.sqrt(pm_outer_radius**2 - x_pm_range**2)
    
    # Inner surface of PM
    y_pm_inner = np.zeros_like(x_pm_range)
    z_pm_inner = np.sqrt(pm_inner_radius**2 - x_pm_range**2)
    
    # Draw PM layer edges with black for CAD-like look
    ax.plot(x_pm_range, y_pm_outer, z_pm_outer, color='black', linewidth=1.0)
    ax.plot(x_pm_range, y_pm_inner, z_pm_inner, color='black', linewidth=1.0)
    
    # NO extra connecting lines or decorations - keep it clean
    
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
    
    # Clean CAD-like style settings
    ax.grid(True, linestyle='-', linewidth=0.5, alpha=0.7)  # Clean grid
    
    # Set view
    if not hasattr(update, 'view_initialized'):
        ax.view_init(elev=20, azim=30)
        update.view_initialized = True
    
    # Ensure mouse controls work
    ax.mouse_init()
    
    return [pg_surf, pm_surf]

# Create animation with better frame rate
frames = range(0, len(area_data), 2)
anim = animation.FuncAnimation(
    fig, update, frames=frames,
    interval=70, blit=False
)

# Set the first frame
update(0)

# Enable interactive mode for better mouse control
plt.ion()

# Show the plot
plt.show(block=True)

print("Clean CAD-style visualization displayed.")
print("Use mouse to rotate the model: left-click and drag")