import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, CheckButtons, TextBox
import matplotlib.patches as patches
from functools import partial

# ──────────────── Figure and axes ────────────────
fig, ax = plt.subplots(figsize=(10, 6))
plt.subplots_adjust(left=0.25, bottom=0.48)   # shifted down a bit to make room for new boxes

# ──────────────── Initial parameters ────────────────
epsilon_c = 0.04
K = 0.25
G0 = 0.01
delta_eps_star = 0.15
beta = 1.0
strain_factor = 0.05
max_strain = 0.2
use_exponential = True

pressure = np.linspace(0, 4.0, 200)  # atm

def pressure_to_strain(p, strain_factor=0.05):
    return p * strain_factor   # linear

def g_epsilon(eps, eps_c, use_exp=True):
    return eps_c * (np.exp(eps/eps_c) - 1) if use_exp else eps

def calculate_tensions(p, eps_c, K, G0, delta_eps_star, beta, strain_factor, use_exp):
    strain = pressure_to_strain(p, strain_factor)
    excess_strain = np.maximum(0, strain - delta_eps_star)
    T_pm = 2 * K * excess_strain
    geometric_factor = (3*beta - 1)/beta
    g_eps = g_epsilon(strain, eps_c, use_exp)
    T_cw = G0 * geometric_factor * (2 * g_eps) / (1 + strain)
    return T_cw, T_pm, T_cw + T_pm, strain

T_cw, T_pm, composite, strain = calculate_tensions(
    pressure, epsilon_c, K, G0, delta_eps_star, beta, strain_factor, use_exponential
)

# Convert to 10‑2 N/m for display
T_cw_disp, T_pm_disp, comp_disp = [arr * 100 for arr in (T_cw, T_pm, composite)]

# ──────────────── Plot curves ────────────────
line_cw, = ax.plot(pressure, T_cw_disp, 'k-', label='T_cw (Cell Wall)')
line_pm, = ax.plot(pressure, T_pm_disp, 'b-', label='T_pm (Plasma Membrane)')
line_comp, = ax.plot(pressure, comp_disp, 'r-', label='σ (Composite)')
rupture_line = ax.axhline(y=1.5, color='r', ls='--', lw=1.5, label='Rupture Tension')

ax.set_xlabel('Internal Pressure (atm)')
ax.set_ylabel('Tension (10$^{-2}$ N/m)')
ax.set_title('Force‑from‑Lipid Mechanism: Tension vs Pressure')
ax.legend()
ax.grid(True)
ax.set_xlim(0, 4.0)
ax.set_ylim(0, 8)

# ──────────────── Sliders and text boxes ────────────────
def add_slider(ax_rect, label, vmin, vmax, vinit):
    ax_s = plt.axes(ax_rect)
    return Slider(ax_s, label, vmin, vmax, valinit=vinit)

def add_textbox(ax_rect, label, init):
    ax_t = plt.axes(ax_rect)
    return TextBox(ax_t, label, initial=init)

slider_eps_c       = add_slider([0.25, 0.38, 0.55, 0.03], 'ε$_c$',      0.01, 0.1,  epsilon_c)
slider_strain_fact = add_slider([0.25, 0.33, 0.55, 0.03], 'Strain Factor', 0.01, 0.2, strain_factor)
slider_K           = add_slider([0.25, 0.28, 0.55, 0.03], 'K (N/m)',   0.1,  0.7,  K)
slider_delta_eps   = add_slider([0.25, 0.23, 0.55, 0.03], 'Δε*',        0,    0.2,  delta_eps_star)
slider_G0          = add_slider([0.25, 0.18, 0.55, 0.03], 'G$_0$ (N/m)', 0.001, 0.05, G0)
slider_beta        = add_slider([0.25, 0.13, 0.55, 0.03], 'β',          1.0,  5.0,  beta)

text_eps_c         = add_textbox([0.84, 0.38, 0.10, 0.03], '', f"{epsilon_c:.2f}")
text_strain_fact   = add_textbox([0.84, 0.33, 0.10, 0.03], '', f"{strain_factor:.2f}")
text_K             = add_textbox([0.84, 0.28, 0.10, 0.03], '', f"{K:.2f}")
text_delta_eps     = add_textbox([0.84, 0.23, 0.10, 0.03], '', f"{delta_eps_star:.2f}")
text_G0            = add_textbox([0.84, 0.18, 0.10, 0.03], '', f"{G0:.3f}")
text_beta          = add_textbox([0.84, 0.13, 0.10, 0.03], '', f"{beta:.1f}")

# ──────────────── NEW axis‑limit text boxes ────────────────
ax_ymax_box = plt.axes([0.025, 0.08, 0.18, 0.04])
ax_xmax_box = plt.axes([0.025, 0.02, 0.18, 0.04])
text_ymax   = TextBox(ax_ymax_box, 'Y max', initial='')
text_xmax   = TextBox(ax_xmax_box, 'X max', initial='')

# Overrides (None = auto)
y_upper_override = None
x_upper_override = None

# ──────────────── Radio buttons, check boxes, formula panel ────────────────
radio = RadioButtons(plt.axes([0.025, 0.55, 0.15, 0.15]),
                     ('Exponential CW', 'Linear CW'))

check = CheckButtons(plt.axes([0.025, 0.75, 0.15, 0.15]),
                     ('T_cw', 'T_pm', 'Composite', 'Rupture'),
                     (True, True, True, True))

ax_formulas = plt.axes([0.025, 0.18, 0.18, 0.33])
ax_formulas.axis('off')
formula_bg = patches.Rectangle((0, 0), 1, 1, transform=ax_formulas.transAxes,
                               facecolor='white', alpha=0.8)
ax_formulas.add_patch(formula_bg)

def update_formula_text():
    ax_formulas.clear(); ax_formulas.axis('off'); ax_formulas.add_patch(formula_bg)
    ax_formulas.text(0.05, 0.97, "Formulae:", fontsize=10, weight='bold')
    ax_formulas.text(0.05, 0.92, "Pressure‑Strain:", fontsize=9, weight='bold')
    ax_formulas.text(0.05, 0.87, r"$\varepsilon(P)=P\!\times\!\text{Strain Factor}$", fontsize=9)
    ax_formulas.text(0.05, 0.70, r"$T_{pm}=2K\max(0,\varepsilon-\Delta\varepsilon_*)$", fontsize=9)
    ax_formulas.text(0.05, 0.63,
        r"$g(\varepsilon)=" + (r"\varepsilon_c\!\left[e^{\varepsilon/\varepsilon_c}-1\right]" if use_exponential else r"\varepsilon") + "$",
        fontsize=9)
    ax_formulas.text(0.05, 0.56, r"$T_{cw}=G_0\frac{(3\beta-1)}{\beta}\frac{2g(\varepsilon)}{1+\varepsilon}$", fontsize=9)
    ax_formulas.text(0.05, 0.48, r"$\sigma=T_{cw}+T_{pm}$", fontsize=9)
    ax_formulas.text(0.05, 0.38, f"Strain Factor={slider_strain_fact.val:.2f}")
    ax_formulas.text(0.05, 0.34, f"K={slider_K.val:.2f} N/m")
    ax_formulas.text(0.05, 0.30, f"G₀={slider_G0.val:.3f} N/m")
    ax_formulas.text(0.05, 0.26, f"β={slider_beta.val:.1f}")
    ax_formulas.text(0.05, 0.22, f"Δε*={slider_delta_eps.val:.2f}")
    if use_exponential:
        ax_formulas.text(0.05, 0.18, f"εc={slider_eps_c.val:.2f}")

update_formula_text()

# ──────────────── Main update routine ────────────────
def update(_=None):
    global y_upper_override, x_upper_override
    eps_c, sf, k_val = slider_eps_c.val, slider_strain_fact.val, slider_K.val
    delta_eps, g0_val, beta_val = slider_delta_eps.val, slider_G0.val, slider_beta.val

    T_cw, T_pm, comp, _ = calculate_tensions(
        pressure, eps_c, k_val, g0_val, delta_eps, beta_val,
        sf, use_exponential
    )
    T_cw_d, T_pm_d, comp_d = [arr*100 for arr in (T_cw, T_pm, comp)]

    line_cw.set_ydata(T_cw_d); line_pm.set_ydata(T_pm_d); line_comp.set_ydata(comp_d)

    # --- Y‑axis handling
    if y_upper_override is None:
        ymax = max(np.max(T_cw_d), np.max(T_pm_d), np.max(comp_d), 1.5)
        if ymax > ax.get_ylim()[1]*0.8 or ymax < ax.get_ylim()[1]*0.5:
            ax.set_ylim(0, ymax*1.2)
    else:
        ax.set_ylim(0, y_upper_override)

    # --- X‑axis handling
    if x_upper_override is not None:
        ax.set_xlim(0, x_upper_override)
    else:
        ax.set_xlim(0, 4.0)

    update_formula_text()
    fig.canvas.draw_idle()

# ──────────────── Helpers to sync sliders ↔ text boxes ────────────────
def link_box(textbox, slider, vmin, vmax):
    def cb(text):
        try:
            val = float(text); 
            if vmin <= val <= vmax: slider.set_val(val)
        except ValueError: pass
    textbox.on_submit(cb)

for tb, sl, mn, mx in (
    (text_eps_c, slider_eps_c,       0.01, 0.10),
    (text_strain_fact, slider_strain_fact, 0.01, 0.20),
    (text_K,   slider_K,             0.10, 0.70),
    (text_delta_eps, slider_delta_eps, 0.00, 0.20),
    (text_G0,  slider_G0,            0.001, 0.05),
    (text_beta, slider_beta,         1.0,  5.0)
): link_box(tb, sl, mn, mx)

# ──────────────── NEW: axis‑limit callbacks ────────────────
def set_ymax(text):
    global y_upper_override
    try:
        val = float(text); y_upper_override = val if val > 0 else None
    except ValueError:
        y_upper_override = None
    update()

def set_xmax(text):
    global x_upper_override
    try:
        val = float(text); x_upper_override = val if val > 0 else None
    except ValueError:
        x_upper_override = None
    update()

text_ymax.on_submit(set_ymax)
text_xmax.on_submit(set_xmax)

# ──────────────── Event wiring ────────────────
for s in (slider_eps_c, slider_strain_fact, slider_K,
          slider_delta_eps, slider_G0, slider_beta):
    s.on_changed(update)

def model_select(label):
    global use_exponential
    use_exponential = (label == 'Exponential CW')
    update()
radio.on_clicked(model_select)

def toggle_visibility(label):
    mapping = {'T_cw':line_cw, 'T_pm':line_pm, 'Composite':line_comp, 'Rupture':rupture_line}
    mapping[label].set_visible(not mapping[label].get_visible())
    fig.canvas.draw_idle()
check.on_clicked(toggle_visibility)

plt.show()
