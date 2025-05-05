import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, CheckButtons, TextBox
import matplotlib.patches as patches
from functools import partial

# Create the figure and axes
fig, ax = plt.subplots(figsize=(10, 6))
plt.subplots_adjust(left=0.25, bottom=0.38)

# Initial parameter values
epsilon_c = 0.04
K = 0.25          # membrane elasticity constant (N/m)
G0 = 0.01         # cell wall elasticity constant (N/m)
delta_eps_star = 0.15  # slack threshold
beta = 1.0        # aspect ratio (L/D)
strain_factor = 0.05   # strain/pressure ratio (replaces delta_p)
max_strain = 0.2  # Only used for plot range reference
use_exponential = True  # Toggle for exponential vs. linear model for CW only

# Create pressure array
pressure = np.linspace(0, 4.0, 200)  # pressure in atm

# Define the pressure-strain relationship - LINEAR instead of plateauing
def pressure_to_strain(p, strain_factor=0.05):
    """Convert pressure to strain using a linear relationship"""
    return p * strain_factor  # Simple linear relationship

# Define strain hardening function g(ε)
def g_epsilon(eps, eps_c, use_exp=True):
    if use_exp:
        return eps_c * (np.exp(eps / eps_c) - 1)
    else:
        return eps

# Function to calculate tensions based on pressure
def calculate_tensions(p, eps_c, K, G0, delta_eps_star, beta, strain_factor, use_exp):
    # Convert pressure to strain - linear relationship
    strain = pressure_to_strain(p, strain_factor)
    
    # Calculate excess strain (max with 0 to ensure no negative values)
    excess_strain = np.maximum(0, strain - delta_eps_star)
    
    # Calculate PM tension (T_pm) - ALWAYS LINEAR regardless of model
    T_pm = 2 * K * excess_strain
    
    # Calculate CW tension (T_cw)
    geometric_factor = (3*beta - 1) / beta
    g_eps = g_epsilon(strain, eps_c, use_exp)
    T_cw = G0 * geometric_factor * (2 * g_eps) / (1 + strain)
    
    # Calculate composite tension
    composite = T_cw + T_pm
    
    return T_cw, T_pm, composite, strain

# Initial calculation
T_cw, T_pm, composite, strain = calculate_tensions(
    pressure, epsilon_c, K, G0, delta_eps_star, beta, strain_factor, use_exponential
)

# Convert to 10^-2 N/m for display
T_cw_display = T_cw * 100
T_pm_display = T_pm * 100
composite_display = composite * 100

# Create the plots
line_cw, = ax.plot(pressure, T_cw_display, 'k-', label='T_cw (Cell Wall)')
line_pm, = ax.plot(pressure, T_pm_display, 'b-', label='T_pm (Plasma Membrane)')
line_composite, = ax.plot(pressure, composite_display, 'r-', label='σ (Composite)')

# Add horizontal line at tension = 1.5×10^-2 N/m (rupture tension)
rupture_line = ax.axhline(y=1.5, color='r', linestyle='--', linewidth=1.5, label='Rupture Tension')

# Set up plot labels and legend
ax.set_xlabel('Internal Pressure (atm)')
ax.set_ylabel('Tension (10$^{-2}$ N/m)')
ax.set_title('Force-from-Lipid Mechanism: Tension vs Pressure')
ax.legend()
ax.grid(True)

# Set axes limits - auto-adjusting later
ax.set_xlim(0, 4.0)
ax.set_ylim(0, 8)

# Create sliders and text input fields for adjusting parameters
ax_eps_c_slider = plt.axes([0.25, 0.28, 0.55, 0.03])
ax_eps_c_text = plt.axes([0.85, 0.28, 0.08, 0.03])

ax_strain_factor_slider = plt.axes([0.25, 0.23, 0.55, 0.03])
ax_strain_factor_text = plt.axes([0.85, 0.23, 0.08, 0.03])

ax_K_slider = plt.axes([0.25, 0.18, 0.55, 0.03])
ax_K_text = plt.axes([0.85, 0.18, 0.08, 0.03])

ax_delta_eps_slider = plt.axes([0.25, 0.13, 0.55, 0.03])
ax_delta_eps_text = plt.axes([0.85, 0.13, 0.08, 0.03])

ax_G0_slider = plt.axes([0.25, 0.08, 0.55, 0.03]) 
ax_G0_text = plt.axes([0.85, 0.08, 0.08, 0.03])

ax_beta_slider = plt.axes([0.25, 0.03, 0.55, 0.03])
ax_beta_text = plt.axes([0.85, 0.03, 0.08, 0.03])

# Create sliders
slider_eps_c = Slider(ax_eps_c_slider, 'ε$_c$', 0.01, 0.1, valinit=epsilon_c)
slider_strain_factor = Slider(ax_strain_factor_slider, 'Strain Factor', 0.01, 0.2, valinit=strain_factor)
slider_K = Slider(ax_K_slider, 'K (N/m)', 0.1, 0.7, valinit=K)
slider_delta_eps = Slider(ax_delta_eps_slider, 'Δε*', 0, 0.2, valinit=delta_eps_star)
slider_G0 = Slider(ax_G0_slider, 'G$_0$ (N/m)', 0.001, 0.05, valinit=G0)
slider_beta = Slider(ax_beta_slider, 'β', 1.0, 5.0, valinit=beta)

# Create text boxes
text_eps_c = TextBox(ax_eps_c_text, '', initial=f"{epsilon_c:.2f}")
text_strain_factor = TextBox(ax_strain_factor_text, '', initial=f"{strain_factor:.2f}")
text_K = TextBox(ax_K_text, '', initial=f"{K:.2f}")
text_delta_eps = TextBox(ax_delta_eps_text, '', initial=f"{delta_eps_star:.2f}")
text_G0 = TextBox(ax_G0_text, '', initial=f"{G0:.3f}")
text_beta = TextBox(ax_beta_text, '', initial=f"{beta:.1f}")

# Create radio buttons for model selection
ax_radio = plt.axes([0.025, 0.5, 0.15, 0.15])
radio = RadioButtons(ax_radio, ('Exponential CW', 'Linear CW'))

# Create checkboxes for toggling tension components
ax_check = plt.axes([0.025, 0.7, 0.15, 0.15])
check = CheckButtons(ax_check, ('T_cw', 'T_pm', 'Composite', 'Rupture'), (True, True, True, True))

# Create a text box for formulas
ax_formulas = plt.axes([0.025, 0.15, 0.2, 0.3])
ax_formulas.axis('off')

# Add a background patch
formula_background = patches.Rectangle((0, 0), 1, 1, transform=ax_formulas.transAxes,
                                       facecolor='white', alpha=0.8)
ax_formulas.add_patch(formula_background)

# Function to update formula text
def update_formula_text():
    ax_formulas.clear()
    ax_formulas.axis('off')
    ax_formulas.add_patch(formula_background)
    
    # Title
    ax_formulas.text(0.05, 0.97, "Formulae:", fontsize=10, fontweight='bold')
    
    # Pressure-Strain Relationship
    ax_formulas.text(0.05, 0.92, "Pressure-Strain Relationship:", fontsize=9, fontweight='bold')
    ax_formulas.text(0.05, 0.87, r"$\varepsilon(P) = P \times \text{Strain Factor}$", fontsize=9)
    ax_formulas.text(0.05, 0.82, r"- Linear relationship", fontsize=8)
    
    # PM tension formula
    ax_formulas.text(0.05, 0.70, r"$T_{pm} = 2K \times \max(0, \varepsilon - \Delta\varepsilon_*)$", fontsize=9)
    
    # CW tension formula
    if use_exponential:
        ax_formulas.text(0.05, 0.63, r"$g(\varepsilon) = \varepsilon_c [e^{\varepsilon/\varepsilon_c} - 1]$", fontsize=9)
    else:
        ax_formulas.text(0.05, 0.63, r"$g(\varepsilon) = \varepsilon$", fontsize=9)
    
    ax_formulas.text(0.05, 0.56, r"$T_{cw} = G_0 \frac{(3\beta-1)}{(\beta)} \frac{(2g(\varepsilon))}{(1+\varepsilon)}$", fontsize=9)
    
    # Composite formula
    ax_formulas.text(0.05, 0.48, r"$\sigma = T_{cw} + T_{pm}$", fontsize=9)
    
    # Parameter values
    ax_formulas.text(0.05, 0.40, f"Strain Factor = {slider_strain_factor.val:.2f}", fontsize=8)
    ax_formulas.text(0.05, 0.35, f"K = {slider_K.val:.2f} N/m", fontsize=8)
    ax_formulas.text(0.05, 0.30, f"G₀ = {slider_G0.val:.3f} N/m", fontsize=8)
    ax_formulas.text(0.05, 0.25, f"β = {slider_beta.val:.1f}", fontsize=8)
    ax_formulas.text(0.05, 0.20, f"Δε* = {slider_delta_eps.val:.2f}", fontsize=8)
    
    if use_exponential:
        ax_formulas.text(0.05, 0.15, f"εc = {slider_eps_c.val:.2f}", fontsize=8)

# Initialize formula text
update_formula_text()

# Main update function
def update(val=None):
    # Get current parameter values
    eps_c = slider_eps_c.val
    strain_factor_val = slider_strain_factor.val
    k_val = slider_K.val
    delta_eps = slider_delta_eps.val
    g0_val = slider_G0.val
    beta_val = slider_beta.val
    
    # Update text boxes to match sliders
    if not text_eps_c.active:
        text_eps_c.set_val(f"{eps_c:.2f}")
    if not text_strain_factor.active:
        text_strain_factor.set_val(f"{strain_factor_val:.2f}")
    if not text_K.active:
        text_K.set_val(f"{k_val:.2f}")
    if not text_delta_eps.active:
        text_delta_eps.set_val(f"{delta_eps:.2f}")
    if not text_G0.active:
        text_G0.set_val(f"{g0_val:.3f}")
    if not text_beta.active:
        text_beta.set_val(f"{beta_val:.1f}")
    
    # Recalculate tensions
    T_cw, T_pm, composite, _ = calculate_tensions(
        pressure, eps_c, k_val, g0_val, delta_eps, beta_val, 
        strain_factor_val, use_exponential
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
    
    # Adjust y-axis limits if needed
    max_val = max(np.max(T_cw_display), np.max(T_pm_display), np.max(composite_display), 1.5)
    if max_val > ax.get_ylim()[1] * 0.8 or max_val < ax.get_ylim()[1] * 0.5:
        ax.set_ylim(0, max_val * 1.2)
    
    # Redraw
    fig.canvas.draw_idle()

# Connect sliders to update function
slider_eps_c.on_changed(update)
slider_strain_factor.on_changed(update)
slider_K.on_changed(update)
slider_delta_eps.on_changed(update)
slider_G0.on_changed(update)
slider_beta.on_changed(update)

# Simplified text input handlers
def submit_text(text, slider, min_val, max_val):
    try:
        value = float(text)
        if min_val <= value <= max_val:
            slider.set_val(value)
    except ValueError:
        pass  # Ignore invalid inputs

# Connect text inputs
text_eps_c.on_submit(partial(submit_text, slider=slider_eps_c, min_val=0.01, max_val=0.1))
text_strain_factor.on_submit(partial(submit_text, slider=slider_strain_factor, min_val=0.01, max_val=0.2))
text_K.on_submit(partial(submit_text, slider=slider_K, min_val=0.1, max_val=0.7))
text_delta_eps.on_submit(partial(submit_text, slider=slider_delta_eps, min_val=0, max_val=0.2))
text_G0.on_submit(partial(submit_text, slider=slider_G0, min_val=0.001, max_val=0.05))
text_beta.on_submit(partial(submit_text, slider=slider_beta, min_val=1.0, max_val=5.0))

# Function to handle radio button selection
def model_select(label):
    global use_exponential
    use_exponential = (label == 'Exponential CW')
    update()
    update_formula_text()

radio.on_clicked(model_select)

# Function to handle checkbox toggles
def toggle_visibility(label):
    if label == 'T_cw':
        line_cw.set_visible(not line_cw.get_visible())
    elif label == 'T_pm':
        line_pm.set_visible(not line_pm.get_visible())
    elif label == 'Composite':
        line_composite.set_visible(not line_composite.get_visible())
    elif label == 'Rupture':
        rupture_line.set_visible(not rupture_line.get_visible())
    fig.canvas.draw_idle()

check.on_clicked(toggle_visibility)

plt.show()