"""
Simplified variant of the interactive tension model. Shares the same parameter
set but removes extra options for a lighter UI.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, CheckButtons, TextBox
import matplotlib.patches as patches
from functools import partial
import time

# ────────────────── Figure and axes ──────────────────
fig, ax = plt.subplots(figsize=(10, 6))
plt.subplots_adjust(left=0.25, bottom=0.38)   # no change needed

# ────────────────── Initial parameters ──────────────────
epsilon_c = 0.04
K = 0.25
G0 = 0.01
delta_eps_star = 0.15
beta = 1.0
use_exponential = True

K_ref, G0_ref = K, G0
strain = np.linspace(0, 0.25, 200)

# ────────────────── Performance helpers ──────────────────
formula_last_updated = 0
formula_update_interval = 0.3     # s

last_update_time = 0
ylim_update_interval = 0.5        # s
updating = pending_update = False
last_ylim_update = 0

# NEW: axis‑override storage
y_upper_override = None
x_upper_override = None

# ────────────────── Mechanics ──────────────────
def g_epsilon(eps, eps_c, use_exp=True):
    return eps_c * (np.exp(eps/eps_c) - 1) if use_exp else eps

def calculate_tensions(strain, eps_c, K, G0, delta_eps_star, beta, use_exp):
    excess = np.maximum(0, strain - delta_eps_star)
    T_pm = 2 * K * excess
    geom = (3*beta - 1) / beta
    g_eps = g_epsilon(strain, eps_c, use_exp)
    T_cw = G0 * geom * (2 * g_eps) / (1 + strain)
    return T_cw, T_pm, T_cw + T_pm

T_cw, T_pm, comp = calculate_tensions(strain, epsilon_c, K, G0, delta_eps_star, beta, use_exponential)
T_cw_d, T_pm_d, comp_d = [x*100 for x in (T_cw, T_pm, comp)]

# ────────────────── Base plot ──────────────────
line_cw,  = ax.plot(strain, T_cw_d, 'k-',  label='T_cw (Cell Wall)')
line_pm,  = ax.plot(strain, T_pm_d, 'k--', label='T_pm (Plasma Membrane)')
line_comp,= ax.plot(strain, comp_d, 'r-',  label='σ (Composite)')
rupture_line = ax.axhline(y=1.0, color='r', ls=':', lw=1.5)
ax.text(0.02, 1.05, 'Rupture Tension', color='r', fontsize=8)

ax.set_xlabel('Strain (ε)')
ax.set_ylabel('Tension (10$^{-2}$ N/m)')
ax.set_title('Force‑from‑Lipid Mechanism: Tension vs Strain')
ax.legend(); ax.grid(True)
ax.set_xlim(0, 0.25); ax.set_ylim(0, 4)

# ────────────────── Sliders & text boxes ──────────────────
slider_eps_c = Slider(plt.axes([0.25, 0.23, 0.55, 0.03]), 'ε$_c$', 0.01, 0.1, valinit=epsilon_c)
slider_K     = Slider(plt.axes([0.25, 0.18, 0.55, 0.03]), 'K (N/m)', 0.1, 0.7, valinit=K)
slider_delp  = Slider(plt.axes([0.25, 0.13, 0.55, 0.03]), 'Δε*', 0, 0.2, valinit=delta_eps_star)
slider_G0    = Slider(plt.axes([0.25, 0.08, 0.55, 0.03]), 'G$_0$ (N/m)', 0.001, 0.2, valinit=G0)
slider_beta  = Slider(plt.axes([0.25, 0.03, 0.55, 0.03]), 'β', 1.0, 5.0, valinit=beta)

text_eps_c = TextBox(plt.axes([0.85, 0.23, 0.08, 0.03]), '', f"{epsilon_c:.2f}")
text_K     = TextBox(plt.axes([0.85, 0.18, 0.08, 0.03]), '', f"{K:.2f}")
text_delp  = TextBox(plt.axes([0.85, 0.13, 0.08, 0.03]), '', f"{delta_eps_star:.2f}")
text_G0    = TextBox(plt.axes([0.85, 0.08, 0.08, 0.03]), '', f"{G0:.3f}")
text_beta  = TextBox(plt.axes([0.85, 0.03, 0.08, 0.03]), '', f"{beta:.1f}")

# NEW: Axis‑limit text boxes
ax_ymax_box = plt.axes([0.025, 0.08, 0.18, 0.04])
ax_xmax_box = plt.axes([0.025, 0.02, 0.18, 0.04])
text_ymax   = TextBox(ax_ymax_box, 'Y max', initial='')
text_xmax   = TextBox(ax_xmax_box, 'X max', initial='')

# ────────────────── Radio & check buttons ──────────────────
radio = RadioButtons(plt.axes([0.025, 0.5, 0.15, 0.15]),
                     ('Exponential CW', 'Linear CW'))
check = CheckButtons(plt.axes([0.025, 0.7, 0.15, 0.15]),
                     ('T_cw', 'T_pm', 'Composite'),
                     (True, True, True))

# ────────────────── Formula panel ──────────────────
ax_formulas = plt.axes([0.025, 0.15, 0.2, 0.3]); ax_formulas.axis('off')
formula_bg = patches.Rectangle((0, 0), 1, 1, transform=ax_formulas.transAxes,
                               facecolor='white', alpha=0.8)
ax_formulas.add_patch(formula_bg)

def update_formula_text(force=False):
    global formula_last_updated
    if not force and time.time() - formula_last_updated < formula_update_interval:
        return
    formula_last_updated = time.time()
    ax_formulas.clear(); ax_formulas.axis('off'); ax_formulas.add_patch(formula_bg)
    ax_formulas.text(0.05, 0.92, r"$T_{pm}=2K\max(0,\varepsilon-\Delta\varepsilon_*)$", fontsize=9)
    ax_formulas.text(0.05, 0.83,
        r"$g(\varepsilon)=" + (r"\varepsilon_c(e^{\varepsilon/\varepsilon_c}-1)" if use_exponential else r"\varepsilon") + "$",
        fontsize=9)
    ax_formulas.text(0.05, 0.74, r"$T_{cw}=G_0\frac{(3\beta-1)}{\beta}\frac{2g(\varepsilon)}{1+\varepsilon}$", fontsize=9)
    ax_formulas.text(0.05, 0.63, r"$\sigma=T_{cw}+T_{pm}$", fontsize=9)
    ax_formulas.text(0.05, 0.46, f"K={slider_K.val:.2f} N/m  [{K_ref:.2f}]")
    ax_formulas.text(0.05, 0.37, f"G₀={slider_G0.val:.3f} N/m [{G0_ref:.3f}]")
    ax_formulas.text(0.05, 0.28, f"β={slider_beta.val:.1f}")
    ax_formulas.text(0.05, 0.19, f"Δε*={slider_delp.val:.2f}")
    if use_exponential:
        ax_formulas.text(0.05, 0.10, f"εc={slider_eps_c.val:.2f}")

update_formula_text(force=True)

# ────────────────── Dynamics ──────────────────
def get_dynamic_interval(g0_val):
    return 0.2 if g0_val > 0.1 else 0.15 if g0_val > 0.05 else 0.1

def update(_=None):
    global updating, pending_update, last_update_time, last_ylim_update
    global y_upper_override, x_upper_override

    now = time.time()
    if updating: pending_update = True; return
    if now - last_update_time < get_dynamic_interval(slider_G0.val):
        pending_update = True; return

    updating = True; pending_update = False; last_update_time = now

    # Sync small text boxes
    if not text_eps_c.active: text_eps_c.set_val(f"{slider_eps_c.val:.2f}")
    if not text_K.active:     text_K.set_val(f"{slider_K.val:.2f}")
    if not text_delp.active:  text_delp.set_val(f"{slider_delp.val:.2f}")
    if not text_G0.active:    text_G0.set_val(f"{slider_G0.val:.3f}")
    if not text_beta.active:  text_beta.set_val(f"{slider_beta.val:.1f}")

    # Compute mechanics
    T_cw, T_pm, comp = calculate_tensions(
        strain, slider_eps_c.val, slider_K.val,
        slider_G0.val, slider_delp.val, slider_beta.val, use_exponential
    )
    T_cw_d, T_pm_d, comp_d = [x*100 for x in (T_cw, T_pm, comp)]
    line_cw.set_ydata(T_cw_d); line_pm.set_ydata(T_pm_d); line_comp.set_ydata(comp_d)

    # Axis‑limit logic
    if y_upper_override is not None:
        ax.set_ylim(0, y_upper_override)
    else:
        if now - last_ylim_update > ylim_update_interval:
            ymax = max(T_cw_d.max(), T_pm_d.max(), comp_d.max())
            headroom = 1.2 if slider_G0.val < 0.05 else 1.5
            if ymax*headroom > ax.get_ylim()[1] or ymax*headroom < ax.get_ylim()[1]*0.5:
                ax.set_ylim(0, ymax*headroom); last_ylim_update = now
    ax.set_xlim(0, x_upper_override if x_upper_override is not None else 0.25)

    update_formula_text()
    fig.canvas.draw_idle()
    updating = False

    if pending_update:
        timer = fig.canvas.new_timer(interval=get_dynamic_interval(slider_G0.val)*1000)
        timer.add_callback(update); timer.start()

# ────────────────── Widget wiring ──────────────────
for s in (slider_eps_c, slider_K, slider_delp, slider_G0, slider_beta):
    s.on_changed(update)

def txt_handler(text, slider, vmin, vmax):
    try:
        v = float(text)
        if vmin <= v <= vmax: slider.set_val(v)
    except ValueError: pass

text_eps_c.on_submit(partial(txt_handler, slider=slider_eps_c, vmin=0.01, vmax=0.1))
text_K.on_submit    (partial(txt_handler, slider=slider_K,     vmin=0.1,  vmax=0.7))
text_delp.on_submit (partial(txt_handler, slider=slider_delp,  vmin=0,    vmax=0.2))
text_G0.on_submit   (partial(txt_handler, slider=slider_G0,    vmin=0.001,vmax=0.2))
text_beta.on_submit (partial(txt_handler, slider=slider_beta,  vmin=1.0,  vmax=5.0))

def model_select(label):
    global use_exponential
    use_exponential = (label == 'Exponential CW')
    update(); update_formula_text(force=True)
radio.on_clicked(model_select)

def toggle_visibility(label):
    mapping = {'T_cw':line_cw, 'T_pm':line_pm, 'Composite':line_comp}
    mapping[label].set_visible(not mapping[label].get_visible())
    fig.canvas.draw_idle()
check.on_clicked(toggle_visibility)

# NEW: axis‑limit callbacks
def set_ymax(text):
    global y_upper_override
    try:
        v = float(text); y_upper_override = v if v > 0 else None
    except ValueError:
        y_upper_override = None
    update()

def set_xmax(text):
    global x_upper_override
    try:
        v = float(text); x_upper_override = v if v > 0 else None
    except ValueError:
        x_upper_override = None
    update()

text_ymax.on_submit(set_ymax)
text_xmax.on_submit(set_xmax)

plt.show()
