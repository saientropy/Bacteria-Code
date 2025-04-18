import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from scipy.interpolate import interp1d
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button, Slider
from matplotlib.patches import Rectangle

# Load the data files
pressure_data = pd.read_csv('plot_data_first_1.csv')
area_data = pd.read_csv('plot_data_second_1.csv')

# Extract column names for clarity
pressure_col = 'Internal Pressure [atm]'
area_col = 'Internal Surface Area [µ²]'
pg_tension_col = 'PG Tension [10^2 N/m]'
pm_tension_col = 'PM Tension [10^2 N/m]'
total_tension_col = 'Total Tension [10^2 N/m]'

# Create a mapping between pressure and area
# First, ensure both datasets are sorted
pressure_data = pressure_data.sort_values(by=pressure_col).reset_index(drop=True)
area_data = area_data.sort_values(by=area_col).reset_index(drop=True)

# Create interpolation functions
# Map pressure to area
pressure_to_area = interp1d(pressure_data[pressure_col], 
                           np.linspace(area_data[area_col].min(), 
                                      area_data[area_col].max(), 
                                      len(pressure_data)),
                           bounds_error=False, fill_value="extrapolate")

# Map area to tensions
area_to_pg_tension = interp1d(area_data[area_col], area_data[pg_tension_col], 
                             bounds_error=False, fill_value="extrapolate")
area_to_pm_tension = interp1d(area_data[area_col], area_data[pm_tension_col], 
                             bounds_error=False, fill_value="extrapolate")

# Calculate radius from surface area (assuming spherical shape)
def area_to_radius(area):
    return np.sqrt(area / (4 * np.pi))

# Set physical dimensions
pg_thickness = 8.0  # nm (outer layer)
pm_thickness = 1.0  # nm (inner layer)

# Function to create a sphere with a 1/8 cutout
def create_cutout_sphere(radius, resolution=40):
    # Create a full sphere
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
    
    # Create cutout (remove points where x<0, y<0, or z<0)
    mask = (x < 0) | (y < 0) | (z < 0)
    x_cutout = x.copy()
    y_cutout = y.copy()
    z_cutout = z.copy()
    
    # Set masked points to NaN (will not be displayed)
    x_cutout[mask] = np.nan
    y_cutout[mask] = np.nan
    z_cutout[mask] = np.nan
    
    return x_cutout, y_cutout, z_cutout

# Function to map tension values to the sphere surface with variations
def map_tension_to_sphere(x, y, z, base_tension, variation=0.1):
    """Create a realistic tension distribution with spatial variations"""
    # Start with base tension
    tension = np.ones_like(x) * base_tension
    
    # Add spatial variations to make it look more realistic
    # Normalize coordinates
    max_val = np.nanmax(np.sqrt(x**2 + y**2 + z**2))
    x_norm = x / max_val
    y_norm = y / max_val
    z_norm = z / max_val
    
    # Create a pattern of variation
    pattern = np.sin(4*x_norm) * np.cos(4*y_norm) + np.sin(4*z_norm)
    pattern_norm = (pattern - np.nanmin(pattern)) / (np.nanmax(pattern) - np.nanmin(pattern))
    
    # Apply the variation to the base tension
    tension = base_tension * (1 + variation * (pattern_norm - 0.5))
    
    return tension

# Set up the figure and axes
fig = plt.figure(figsize=(16, 10))
gs = gridspec.GridSpec(3, 6, height_ratios=[5, 1, 0.5])

# Main 3D plot
ax_3d = fig.add_subplot(gs[0, :4], projection='3d')

# Information panel
ax_info = fig.add_subplot(gs[0, 4:])
ax_info.axis('off')

# Color bars for PG and PM tensions
ax_pg_cbar = fig.add_subplot(gs[1, 1:3])
ax_pm_cbar = fig.add_subplot(gs[1, 3:5])

# Controls area
ax_slider = fig.add_subplot(gs[2, 1:5])

# Set up color normalization based on data ranges
pg_norm = colors.Normalize(vmin=0, vmax=np.max(pressure_data[pg_tension_col]))
pm_norm = colors.Normalize(vmin=0, vmax=np.max(pressure_data[pm_tension_col]))

# Use different colormaps for the layers
pg_cmap = plt.cm.viridis
pm_cmap = plt.cm.plasma

# Create color bars
pg_sm = cm.ScalarMappable(cmap=pg_cmap, norm=pg_norm)
pm_sm = cm.ScalarMappable(cmap=pm_cmap, norm=pm_norm)
pg_sm.set_array([])
pm_sm.set_array([])

fig.colorbar(pg_sm, cax=ax_pg_cbar, orientation='horizontal', 
             label=f'PG Tension [{pg_tension_col.split("[")[1]}')
fig.colorbar(pm_sm, cax=ax_pm_cbar, orientation='horizontal', 
             label=f'PM Tension [{pm_tension_col.split("[")[1]}')

# Create a frame slider
frame_slider = Slider(
    ax=ax_slider,
    label='Frame',
    valmin=0,
    valmax=len(pressure_data)-1,
    valinit=0,
    valstep=1
)

# Add buttons for play/pause
ax_play = plt.axes([0.8, 0.025, 0.1, 0.04])
button_play = Button(ax_play, 'Play/Pause')

# Animation control variables
is_playing = False
current_frame = 0

# Initialize plot elements
pg_surf = None
pm_surf = None
interface_surf = None
info_text = None
pressure_gauge = None

def update_plot(frame):
    """Update the plot for the given frame"""
    global pg_surf, pm_surf, interface_surf, info_text, pressure_gauge
    
    # Clear previous frame
    ax_3d.clear()
    ax_info.clear()
    
    # Get current data values
    current_pressure = pressure_data[pressure_col].iloc[frame]
    current_area = pressure_to_area(current_pressure)
    
    # Get current tensions
    current_pg_tension = area_to_pg_tension(current_area)
    current_pm_tension = area_to_pm_tension(current_area)
    
    # Calculate radii for the layers
    inner_radius = area_to_radius(current_area)  # Inner PM surface
    pm_outer_radius = inner_radius + pm_thickness  # PM outer / PG inner surface
    pg_outer_radius = pm_outer_radius + pg_thickness  # PG outer surface
    
    # Create sphere geometries with cutout
    # PG (outer) layer
    x_pg, y_pg, z_pg = create_cutout_sphere(pg_outer_radius)
    pg_tensions = map_tension_to_sphere(x_pg, y_pg, z_pg, current_pg_tension)
    
    # Interface between layers
    x_interface, y_interface, z_interface = create_cutout_sphere(pm_outer_radius)
    
    # PM (inner) layer
    x_pm, y_pm, z_pm = create_cutout_sphere(inner_radius)
    pm_tensions = map_tension_to_sphere(x_pm, y_pm, z_pm, current_pm_tension, variation=0.15)
    
    # Plot the surfaces with tension-based coloring
    # PG layer (outer)
    pg_surf = ax_3d.plot_surface(
        x_pg, y_pg, z_pg,
        facecolors=pg_cmap(pg_norm(pg_tensions)),
        rstride=1, cstride=1,
        linewidth=0, antialiased=True,
        alpha=0.8
    )
    
    # Interface between layers
    interface_surf = ax_3d.plot_surface(
        x_interface, y_interface, z_interface,
        color='gray', alpha=0.3,
        rstride=2, cstride=2
    )
    
    # PM layer (inner)
    pm_surf = ax_3d.plot_surface(
        x_pm, y_pm, z_pm,
        facecolors=pm_cmap(pm_norm(pm_tensions)),
        rstride=1, cstride=1,
        linewidth=0, antialiased=True,
        alpha=0.8
    )
    
    # Add cutting planes for better visualization
    # XY plane (z=0)
    xx, yy = np.meshgrid(np.linspace(0, pg_outer_radius, 10), np.linspace(0, pg_outer_radius, 10))
    zz = np.zeros_like(xx)
    ax_3d.plot_surface(xx, yy, zz, alpha=0.1, color='lightgray')
    
    # YZ plane (x=0)
    yy, zz = np.meshgrid(np.linspace(0, pg_outer_radius, 10), np.linspace(0, pg_outer_radius, 10))
    xx = np.zeros_like(yy)
    ax_3d.plot_surface(xx, yy, zz, alpha=0.1, color='lightgray')
    
    # XZ plane (y=0)
    xx, zz = np.meshgrid(np.linspace(0, pg_outer_radius, 10), np.linspace(0, pg_outer_radius, 10))
    yy = np.zeros_like(xx)
    ax_3d.plot_surface(xx, yy, zz, alpha=0.1, color='lightgray')
    
    # Set axis properties
    ax_3d.set_xlabel('X [nm]')
    ax_3d.set_ylabel('Y [nm]')
    ax_3d.set_zlabel('Z [nm]')
    ax_3d.set_title('Bacterial Cell Wall Expansion', fontsize=14)
    
    # Set equal aspect ratio and limits
    max_range = pg_outer_radius * 1.2
    ax_3d.set_xlim(0, max_range)
    ax_3d.set_ylim(0, max_range)
    ax_3d.set_zlim(0, max_range)
    
    # Set viewing angle
    ax_3d.view_init(elev=25, azim=30)
    
    # Update information panel
    info_str = (
        f"Frame: {frame+1}/{len(pressure_data)}\n\n"
        f"Internal Pressure: {current_pressure:.2f} [atm]\n\n"
        f"Internal Surface Area: {current_area:.2f} [µ²]\n\n"
        f"PG Layer (Outer):\n"
        f"  - Thickness: {pg_thickness:.1f} nm\n"
        f"  - Tension: {current_pg_tension:.3f} [10^2 N/m]\n\n"
        f"PM Layer (Inner):\n"
        f"  - Thickness: {pm_thickness:.1f} nm\n"
        f"  - Tension: {current_pm_tension:.3f} [10^2 N/m]\n\n"
        f"Inner Radius: {inner_radius:.2f} nm\n"
        f"Total Radius: {pg_outer_radius:.2f} nm\n\n"
        f"Note: PM tension range is smaller\n"
        f"than PG tension range by a factor\n"
        f"of {pg_norm.vmax / pm_norm.vmax:.1f}x"
    )
    
    info_text = ax_info.text(0.05, 0.95, info_str,
                           va='top', ha='left',
                           fontsize=11,
                           transform=ax_info.transAxes)
    
    # Add pressure gauge
    gauge_height = 0.2
    gauge_width = 0.1
    gauge_x = 0.7
    gauge_y = 0.2
    
    # Gauge background
    ax_info.add_patch(Rectangle((gauge_x, gauge_y), gauge_width, gauge_height,
                              fill=True, color='lightgray', alpha=0.5))
    
    # Gauge level (normalized pressure)
    pressure_level = current_pressure / np.max(pressure_data[pressure_col])
    level_height = pressure_level * gauge_height
    
    pressure_gauge = ax_info.add_patch(
        Rectangle((gauge_x, gauge_y), gauge_width, level_height,
                fill=True, color='red', alpha=0.7)
    )
    
    # Add gauge labels
    ax_info.text(gauge_x + gauge_width/2, gauge_y - 0.02, '0', ha='center', fontsize=9)
    ax_info.text(gauge_x + gauge_width/2, gauge_y + gauge_height + 0.02,
               f'{np.max(pressure_data[pressure_col]):.1f}', ha='center', fontsize=9)
    ax_info.text(gauge_x + gauge_width/2, gauge_y - 0.05, 'Pressure [atm]',
               ha='center', fontsize=10)
    
    # Add layer structure diagram
    layer_x = 0.6
    layer_y = 0.6
    layer_width = 0.3
    total_height = 0.3
    
    # Calculate proportional heights
    pg_layer_height = (pg_thickness / (pg_thickness + pm_thickness)) * total_height
    pm_layer_height = (pm_thickness / (pg_thickness + pm_thickness)) * total_height
    
    # PG layer
    ax_info.add_patch(Rectangle(
        (layer_x, layer_y + pm_layer_height), layer_width, pg_layer_height,
        fill=True, color=pg_cmap(0.7), alpha=0.7,
        ec='black', lw=1
    ))
    
    # PM layer
    ax_info.add_patch(Rectangle(
        (layer_x, layer_y), layer_width, pm_layer_height,
        fill=True, color=pm_cmap(0.7), alpha=0.7,
        ec='black', lw=1
    ))
    
    # Layer labels
    ax_info.text(layer_x + layer_width/2, layer_y + pm_layer_height + pg_layer_height/2,
               'PG', ha='center', va='center', fontsize=10)
    ax_info.text(layer_x + layer_width/2, layer_y + pm_layer_height/2,
               'PM', ha='center', va='center', fontsize=10)
    
    ax_info.text(layer_x - 0.05, layer_y + total_height/2,
               'Layer Structure', ha='right', va='center', fontsize=10, rotation=90)
    
    # Add scale bar
    scale_bar_length = 10  # nm
    scale_x = 0.05
    scale_y = 0.05
    
    ax_3d.plot([scale_x, scale_x + scale_bar_length], [scale_y, scale_y], [0, 0], 
              color='black', linewidth=2)
    ax_3d.text(scale_x + scale_bar_length/2, scale_y + 1, 0, f'{scale_bar_length} nm', 
              ha='center', va='bottom')
    
    # Update slider value
    frame_slider.set_val(frame)
    
    return pg_surf, pm_surf, interface_surf, pressure_gauge, info_text

def animate(frame):
    global current_frame
    current_frame = frame
    return update_plot(frame)

def on_slider_changed(val):
    global current_frame
    current_frame = int(val)
    update_plot(current_frame)
    fig.canvas.draw_idle()

def toggle_animation(event):
    global is_playing, ani
    if is_playing:
        ani.event_source.stop()
        is_playing = False
    else:
        ani.event_source.start()
        is_playing = True

# Connect events
frame_slider.on_changed(on_slider_changed)
button_play.on_clicked(toggle_animation)

# Add a title
fig.suptitle('Bacterial Cell Wall Expansion Simulation', fontsize=16, y=0.98)

# Create animation
ani = animation.FuncAnimation(
    fig, animate, frames=len(pressure_data),
    interval=100, blit=False
)

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# Uncomment to save the animation
ani.save('bacteria_expansion.mp4', writer='ffmpeg', fps=10, dpi=200)