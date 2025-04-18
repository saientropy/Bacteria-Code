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

# Function to create a sphere with 1/4 cutout (larger cutout)
def create_cutout_sphere(radius, resolution=50):
    """Create a sphere with a larger cutout to better see inside."""
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
    
    # Create a larger cutout - 1/4 of the sphere instead of 1/8
    mask = np.zeros_like(x, dtype=bool)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            # Expanded cutout criteria to show more of the interior
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

# Create custom colormaps with specific thresholding for PM values
pg_cmap = cm.viridis     # Blue-green, dark to light for PG layer
pm_cmap = cm.magma_r     # Yellow-red-purple, light to dark for PM layer

# For PM, create a custom normalized colormap where values < 0.2 have the same color
class CustomNorm(mpl.colors.Normalize):
    def __init__(self, vmin=0, vmax=1, threshold=0.2, clip=False):
        self.threshold = threshold
        super().__init__(vmin, vmax, clip)
    
    def __call__(self, value, clip=None):
        # Make all values below threshold map to the same color
        if np.iterable(value):
            result = np.array([self(val) for val in value])
            return result
        else:
            if value < self.threshold:
                return super().__call__(self.threshold)
            else:
                return super().__call__(value)

# Use custom norm for PM with threshold at 0.2
pg_norm = mpl.colors.Normalize(vmin=0, vmax=area_data['PG_Tension'].max())
pm_norm = CustomNorm(vmin=0, vmax=area_data['PM_Tension'].max(), threshold=0.2)

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
                  fontsize=12, color='darkmagenta',
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
fig.text(0.85, 0.56, 'PM Layer (1nm)', fontsize=10, ha='center', color='darkmagenta', weight='bold')

# Function to update animation
def update(frame):
    ax.clear()
    
    # Get current data
    area_val = area_data.iloc[frame]['Surface_Area']
    radius = area_data.iloc[frame]['Radius']
    pg_tension = area_data.iloc[frame]['PG_Tension']
    pm_tension = area_data.iloc[frame]['PM_Tension']
    pressure = get_pressure_for_area(area_val)
    
    # Calculate radii for both layers
    inner_radius = radius + inner_thickness/2
    outer_radius = radius + inner_thickness + outer_thickness/2
    
    # Create spheres with larger cutout
    x_inner, y_inner, z_inner = create_cutout_sphere(inner_radius, resolution=50)
    x_outer, y_outer, z_outer = create_cutout_sphere(outer_radius, resolution=50)
    
    # Plot inner sphere (PM layer) - clearly visible
    inner_surf = ax.plot_surface(
        x_inner, y_inner, z_inner,
        color='magenta', alpha=1.0,  # Fully opaque for visibility
        rstride=1, cstride=1,
        linewidth=0, antialiased=True
    )
    
    # Color inner surface based on PM tension
    inner_surf.set_facecolor(pm_cmap(pm_norm(pm_tension)))
    
    # Plot outer sphere (PG layer) with more transparency
    outer_surf = ax.plot_surface(
        x_outer, y_outer, z_outer,
        color='blue', alpha=0.7,  # Semi-transparent
        rstride=1, cstride=1,
        linewidth=0, antialiased=True
    )
    
    # Color outer surface based on PG tension
    outer_surf.set_facecolor(pg_cmap(pg_norm(pg_tension)))
    
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
    
    # Adjust the view angle for better visualization of the cutout
    ax.view_init(elev=20, azim=30)  # Lower elevation angle to see inside better
    
    return [inner_surf, outer_surf]

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
# anim.save('bacteria_expansion.mp4', writer='ffmpeg', fps=25, dpi=150)

print("Animation displayed. Use mouse to rotate the model.")