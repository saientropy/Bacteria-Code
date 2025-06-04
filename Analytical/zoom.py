"""
Helper module that adds mouse-controlled zooming to the interactive plots. Lets
users inspect fine details of the tension curves.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, CheckButtons, TextBox
import matplotlib.patches as patches
from functools import partial
import time
from matplotlib.backend_bases import MouseButton

# Create the figure and axes
fig, ax = plt.subplots(figsize=(10, 6))
plt.subplots_adjust(left=0.25, bottom=0.38)

# Initial parameter values (reference values)
epsilon_c = 0.04
K = 0.25          # membrane elasticity constant (N/m)
G0 = 0.01         # cell wall elasticity constant (N/m)
delta_eps_star = 0.15  # slack threshold
beta = 1.0        # aspect ratio (L/D)
use_exponential = True  # Toggle for exponential vs. linear model for CW only

# Store reference values for display
K_ref = K
G0_ref = G0

# Create strain array with fewer points for better performance
strain = np.linspace(0, 0.25, 200)  # Further reduced resolution

# Cache for formula text
formula_last_updated = 0
formula_update_interval = 0.3  # seconds between formula updates

# Throttling variables
last_update_time = 0
update_interval = 0.1  # seconds between updates
updating = False
pending_update = False
last_ylim_update = 0
ylim_update_interval = 0.5  # Limit y-axis updates to reduce lag

# Define strain hardening function g(ε) - precompute for better performance
def g_epsilon(eps, eps_c, use_exp=True):
    if use_exp:
        return eps_c * (np.exp(eps / eps_c) - 1)
    else:
        return eps

# Optimized calculation function with caching for common values
def calculate_tensions(strain, eps_c, K, G0, delta_eps_star, beta, use_exp):
    # Calculate excess strain (max with 0 to ensure no negative values)
    excess_strain = np.maximum(0, strain - delta_eps_star)
    
    # Calculate PM tension (T_pm) - ALWAYS LINEAR regardless of exponential model setting
    # T_pm = 2K × Δε
    T_pm = 2 * K * excess_strain
    
    # Calculate CW tension (T_cw) - can be exponential or linear
    # Precompute geometric factor to save time
    geometric_factor = (3*beta - 1) / beta
    
    # Calculate g(ε) based on model
    g_eps = g_epsilon(strain, eps_c, use_exp)
    
    # Calculate Cell Wall tension
    # T_cw = G0 * geometric_factor * (2g(ε)/(1+ε))
    T_cw = G0 * geometric_factor * (2 * g_eps) / (1 + strain)
    
    # Calculate composite tension (σ)
    composite = T_cw + T_pm
    
    return T_cw, T_pm, composite

# Initial plot
T_cw, T_pm, composite = calculate_tensions(
    strain, epsilon_c, K, G0, delta_eps_star, beta, use_exponential
)

# Convert to 10^-2 N/m for display
T_cw_display = T_cw * 100
T_pm_display = T_pm * 100
composite_display = composite * 100

# Create the plots
line_cw, = ax.plot(strain, T_cw_display, 'k-', label='T_cw (Cell Wall)')
line_pm, = ax.plot(strain, T_pm_display, 'k--', label='T_pm (Plasma Membrane)')
line_composite, = ax.plot(strain, composite_display, 'r-', label='σ (Composite)')

# Add horizontal line at tension = 1×10^-2 N/m (rupture tension)
rupture_line = ax.axhline(y=1.0, color='r', linestyle=':', linewidth=1.5)
ax.text(0.02, 1.05, 'Rupture Tension', color='r', fontsize=8)

# Set up plot labels and legend
ax.set_xlabel('Strain (ε)')
ax.set_ylabel('Tension (10$^{-2}$ N/m)')
ax.set_title('Force-from-Lipid Mechanism: Tension vs Strain')
ax.legend()
ax.grid(True)

# Set axes limits
ax.set_xlim(0, 0.25)
ax.set_ylim(0, 4)

# Store original view limits for reset function
original_xlim = ax.get_xlim()
original_ylim = ax.get_ylim()

# Create sliders and text input fields for adjusting parameters
ax_eps_c_slider = plt.axes([0.25, 0.23, 0.55, 0.03])
ax_eps_c_text = plt.axes([0.85, 0.23, 0.08, 0.03])

ax_K_slider = plt.axes([0.25, 0.18, 0.55, 0.03])
ax_K_text = plt.axes([0.85, 0.18, 0.08, 0.03])

ax_delta_eps_slider = plt.axes([0.25, 0.13, 0.55, 0.03])
ax_delta_eps_text = plt.axes([0.85, 0.13, 0.08, 0.03])

ax_G0_slider = plt.axes([0.25, 0.08, 0.55, 0.03]) 
ax_G0_text = plt.axes([0.85, 0.08, 0.08, 0.03])

ax_beta_slider = plt.axes([0.25, 0.03, 0.55, 0.03])
ax_beta_text = plt.axes([0.85, 0.03, 0.08, 0.03])

# Create sliders with updated ranges
slider_eps_c = Slider(ax_eps_c_slider, 'ε$_c$', 0.01, 0.1, valinit=epsilon_c)
slider_K = Slider(ax_K_slider, 'K (N/m)', 0.1, 0.7, valinit=K)
slider_delta_eps = Slider(ax_delta_eps_slider, 'Δε*', 0, 0.2, valinit=delta_eps_star)
slider_G0 = Slider(ax_G0_slider, 'G$_0$ (N/m)', 0.001, 0.2, valinit=G0)  # Range now goes to 0.2
slider_beta = Slider(ax_beta_slider, 'β', 1.0, 5.0, valinit=beta)

# Create text boxes with shorter values to avoid computational overhead
text_eps_c = TextBox(ax_eps_c_text, '', initial=f"{epsilon_c:.2f}")
text_K = TextBox(ax_K_text, '', initial=f"{K:.2f}")
text_delta_eps = TextBox(ax_delta_eps_text, '', initial=f"{delta_eps_star:.2f}")
text_G0 = TextBox(ax_G0_text, '', initial=f"{G0:.3f}")
text_beta = TextBox(ax_beta_text, '', initial=f"{beta:.1f}")

# Create radio buttons for model selection
ax_radio = plt.axes([0.025, 0.5, 0.15, 0.15])
radio = RadioButtons(ax_radio, ('Exponential CW', 'Linear CW'))

# Create checkboxes for toggling tension components
ax_check = plt.axes([0.025, 0.7, 0.15, 0.15])
check = CheckButtons(ax_check, ('T_cw', 'T_pm', 'Composite'), (True, True, True))

# Create a text box for formulas with cached updates
ax_formulas = plt.axes([0.025, 0.15, 0.2, 0.3])
ax_formulas.axis('off')

# Add a background patch
formula_background = patches.Rectangle((0, 0), 1, 1, transform=ax_formulas.transAxes,
                                       facecolor='white', alpha=0.8)
ax_formulas.add_patch(formula_background)

# Function to update formula text - now with caching
def update_formula_text(force=False):
    global formula_last_updated
    
    # Only update formulas occasionally to improve performance
    current_time = time.time()
    if not force and current_time - formula_last_updated < formula_update_interval:
        return
    
    formula_last_updated = current_time
    
    ax_formulas.clear()
    ax_formulas.axis('off')
    ax_formulas.add_patch(formula_background)
    
    # Title
    ax_formulas.text(0.05, 0.95, "Formulas:", fontsize=10, fontweight='bold')
    
    # PM tension formula
    ax_formulas.text(0.05, 0.8, r"$T_{pm} = 2K \times \max(0, \varepsilon - \Delta\varepsilon_*)$", fontsize=9)
    
    # CW tension formula
    if use_exponential:
        ax_formulas.text(0.05, 0.7, r"$g(\varepsilon) = \varepsilon_c [e^{\varepsilon/\varepsilon_c} - 1]$", fontsize=9)
    else:
        ax_formulas.text(0.05, 0.7, r"$g(\varepsilon) = \varepsilon$", fontsize=9)
    
    ax_formulas.text(0.05, 0.6, r"$T_{cw} = G_0 \frac{(3\beta-1)}{(\beta)} \frac{(2g(\varepsilon))}{(1+\varepsilon)}$", fontsize=9)
    
    # Composite formula
    ax_formulas.text(0.05, 0.45, r"$\sigma = T_{cw} + T_{pm}$", fontsize=9)
    
    # Parameter values - completely reformatted for better spacing
    # Parameter: K
    k_val = slider_K.val
    ax_formulas.text(0.05, 0.3, "K =", fontsize=8)
    ax_formulas.text(0.12, 0.3, f"{k_val:.2f}", fontsize=8)
    ax_formulas.text(0.18, 0.3, "N/m", fontsize=8)
    ax_formulas.text(0.25, 0.3, f"[{K_ref:.2f}]", fontsize=8, color='red')
    
    # Parameter: G₀
    g0_val = slider_G0.val
    ax_formulas.text(0.05, 0.23, "G₀ =", fontsize=8)
    ax_formulas.text(0.12, 0.23, f"{g0_val:.3f}", fontsize=8)
    ax_formulas.text(0.18, 0.23, "N/m", fontsize=8)
    ax_formulas.text(0.25, 0.23, f"[{G0_ref:.3f}]", fontsize=8, color='red')
    
    beta_val = slider_beta.val
    ax_formulas.text(0.05, 0.16, f"β = {beta_val:.1f}", fontsize=8)
    
    delta_eps = slider_delta_eps.val
    ax_formulas.text(0.05, 0.09, f"Δε* = {delta_eps:.2f}", fontsize=8)
    
    if use_exponential:
        eps_c = slider_eps_c.val
        ax_formulas.text(0.05, 0.02, f"εc = {eps_c:.2f}", fontsize=8)

# Initialize formula text
update_formula_text(force=True)

# Adaptive update interval based on G0 value (higher G0 = more throttling)
def get_dynamic_update_interval(g0_val):
    # Scale update interval based on G0 value
    if g0_val > 0.1:
        return 0.2  # Slower updates for very large G0
    elif g0_val > 0.05:
        return 0.15  # Medium throttling
    else:
        return 0.1  # Normal speed for small G0

# Main update function with throttling and optimization
def update(val=None):
    global updating, pending_update, last_update_time, last_ylim_update
    
    current_time = time.time()
    
    # Get current parameter values
    eps_c = slider_eps_c.val
    k_val = slider_K.val
    delta_eps = slider_delta_eps.val
    g0_val = slider_G0.val
    beta_val = slider_beta.val
    
    # Get dynamic update interval based on G0 value
    dynamic_interval = get_dynamic_update_interval(g0_val)
    
    # Check if we're already updating or if we need to throttle
    if updating:
        pending_update = True
        return
    
    if current_time - last_update_time < dynamic_interval:
        # Schedule update for later
        if not pending_update:
            pending_update = True
            fig.canvas.draw_idle()
        return
    
    updating = True
    pending_update = False
    last_update_time = current_time
    
    # Update text boxes to match sliders (with minimal formatting)
    if not text_eps_c.active:
        text_eps_c.set_val(f"{eps_c:.2f}")
    if not text_K.active:
        text_K.set_val(f"{k_val:.2f}")
    if not text_delta_eps.active:
        text_delta_eps.set_val(f"{delta_eps:.2f}")
    if not text_G0.active:
        text_G0.set_val(f"{g0_val:.3f}")
    if not text_beta.active:
        text_beta.set_val(f"{beta_val:.1f}")
    
    # Recalculate tensions
    T_cw, T_pm, composite = calculate_tensions(
        strain, eps_c, k_val, g0_val, delta_eps, beta_val, use_exponential
    )
    
    # Convert to 10^-2 N/m for display
    T_cw_display = T_cw * 100
    T_pm_display = T_pm * 100
    composite_display = composite * 100
    
    # Update plot data
    line_cw.set_ydata(T_cw_display)
    line_pm.set_ydata(T_pm_display)
    line_composite.set_ydata(composite_display)
    
    # Update formula text
    update_formula_text()
    
    # Adjust y-axis limits - but not too frequently for large G0 values
    update_ylim = (current_time - last_ylim_update > ylim_update_interval)
    
    if update_ylim:
        max_val = max(np.max(T_cw_display), np.max(T_pm_display), np.max(composite_display))
        
        # Add extra headroom for larger G0 values
        headroom = 1.2 if g0_val < 0.05 else 1.5
        
        # Check if we need to update the y-axis (avoid minor changes)
        current_ylim = ax.get_ylim()[1]
        if max_val * headroom > current_ylim or max_val * headroom < current_ylim * 0.5:
            ax.set_ylim(0, max_val * headroom)
            last_ylim_update = current_time
    
    fig.canvas.draw_idle()
    updating = False
    
    # Check if another update is pending
    if pending_update:
        # Use a timer callback to update after the interval
        timer = fig.canvas.new_timer(interval=dynamic_interval*1000)
        timer.add_callback(update)
        timer.start()

# Connect sliders to update function
slider_eps_c.on_changed(update)
slider_K.on_changed(update)
slider_delta_eps.on_changed(update)
slider_G0.on_changed(update)
slider_beta.on_changed(update)

# Simplified text input handlers
def submit_text(text, slider, min_val, max_val, format_str):
    try:
        value = float(text)
        if min_val <= value <= max_val:
            slider.set_val(value)
    except ValueError:
        pass  # Ignore invalid inputs

# Connect text inputs with simplified handlers - updated max_val for G0 to 0.2
text_eps_c.on_submit(partial(submit_text, slider=slider_eps_c, min_val=0.01, max_val=0.1, format_str="{:.2f}"))
text_K.on_submit(partial(submit_text, slider=slider_K, min_val=0.1, max_val=0.7, format_str="{:.2f}"))
text_delta_eps.on_submit(partial(submit_text, slider=slider_delta_eps, min_val=0, max_val=0.2, format_str="{:.2f}"))
text_G0.on_submit(partial(submit_text, slider=slider_G0, min_val=0.001, max_val=0.2, format_str="{:.3f}"))  # Updated to 0.2
text_beta.on_submit(partial(submit_text, slider=slider_beta, min_val=1.0, max_val=5.0, format_str="{:.1f}"))

# Function to handle radio button selection
def model_select(label):
    global use_exponential
    use_exponential = (label == 'Exponential CW')
    update()
    # Force formula update when model changes
    update_formula_text(force=True)

radio.on_clicked(model_select)

# Function to handle checkbox toggles
def toggle_visibility(label):
    if label == 'T_cw':
        line_cw.set_visible(not line_cw.get_visible())
    elif label == 'T_pm':
        line_pm.set_visible(not line_pm.get_visible())
    elif label == 'Composite':
        line_composite.set_visible(not line_composite.get_visible())
    fig.canvas.draw_idle()

check.on_clicked(toggle_visibility)

# Add a button for resetting zoom
ax_reset = plt.axes([0.85, 0.33, 0.08, 0.03])
button_reset = plt.Button(ax_reset, 'Reset Zoom')

def reset_zoom(event):
    ax.set_xlim(original_xlim)
    ax.set_ylim(original_ylim)
    fig.canvas.draw_idle()

button_reset.on_clicked(reset_zoom)

# Add mouse wheel zoom functionality
def on_scroll(event):
    # If the event happened in the plot area
    if event.inaxes == ax:
        # Get the current x and y limits
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        
        # Get the mouse position in data coordinates
        x_data = event.xdata
        y_data = event.ydata
        
        # Set the zoom factor
        zoom_factor = 1.1
        
        # Zoom in or out depending on scroll direction
        if event.button == 'up':  # Scroll up to zoom in
            # Compute new limits centered around mouse position
            new_width = (x_max - x_min) / zoom_factor
            new_height = (y_max - y_min) / zoom_factor
            
            # Calculate distances as fractions
            fractional_x = (x_data - x_min) / (x_max - x_min)
            fractional_y = (y_data - y_min) / (y_max - y_min)
            
            # Calculate new limits while maintaining the mouse position
            new_x_min = x_data - fractional_x * new_width
            new_x_max = new_x_min + new_width
            new_y_min = y_data - fractional_y * new_height
            new_y_max = new_y_min + new_height
            
        elif event.button == 'down':  # Scroll down to zoom out
            # Compute new limits centered around mouse position
            new_width = (x_max - x_min) * zoom_factor
            new_height = (y_max - y_min) * zoom_factor
            
            # Calculate distances as fractions
            fractional_x = (x_data - x_min) / (x_max - x_min)
            fractional_y = (y_data - y_min) / (y_max - y_min)
            
            # Calculate new limits while maintaining the mouse position
            new_x_min = x_data - fractional_x * new_width
            new_x_max = new_x_min + new_width
            new_y_min = y_data - fractional_y * new_height
            new_y_max = new_y_min + new_height
        
        # Set the new limits
        ax.set_xlim(new_x_min, new_x_max)
        ax.set_ylim(new_y_min, new_y_max)
        
        # Redraw the canvas
        fig.canvas.draw_idle()

# Connect the scroll event
fig.canvas.mpl_connect('scroll_event', on_scroll)

# Add simple panning functionality with mouse drag
def on_press(event):
    if event.inaxes == ax:
        ax._pan_start = (event.xdata, event.ydata, ax.get_xlim(), ax.get_ylim())

def on_motion(event):
    if hasattr(ax, '_pan_start') and event.inaxes == ax and event.button == MouseButton.LEFT:
        start_x, start_y, xlim, ylim = ax._pan_start
        dx = event.xdata - start_x
        dy = event.ydata - start_y
        ax.set_xlim(xlim[0] - dx, xlim[1] - dx)
        ax.set_ylim(ylim[0] - dy, ylim[1] - dy)
        fig.canvas.draw_idle()

# Connect the mouse events for panning
fig.canvas.mpl_connect('button_press_event', on_press)
fig.canvas.mpl_connect('motion_notify_event', on_motion)

# Add a help text about zooming functionality
ax.text(0.02, 3.8, "Use mouse wheel to zoom, left-click drag to pan, or click 'Reset Zoom'", 
        fontsize=8, color='blue', alpha=0.7)

plt.show()