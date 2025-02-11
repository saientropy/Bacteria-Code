import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------
# Style settings for a cleaner, less "cartoonish" look
# --------------------------------------------------------
plt.rcParams.update({
    'axes.linewidth': 0.8,   # thinner axis border
    'lines.linewidth': 1.5,  # thinner line than before
    'font.size': 12,
    'legend.fontsize': 12,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10
})

# ---------------------
# Piecewise definitions for PG & PM tension
# ---------------------
def Tpm(e):
    """Plasma membrane tension vs. strain."""
    Tpm0 = 0.3e-2  # small nonzero tension at e=0
    slope = 10e-2  # slope in N/m per unit strain
    return Tpm0 + slope*e

def Tcw(e):
    """Cell-wall (PG) tension vs. strain, piecewise linear with a threshold."""
    e_thresh = 0.05
    Tcw0 = 0.2e-2
    slope_lo = (0.5e-2 - Tcw0) / e_thresh
    slope_hi = 30e-2
    
    if e <= 0:
        return Tcw0
    elif e < e_thresh:
        return Tcw0 + slope_lo*e
    else:
        Tcw_at_thresh = Tcw0 + slope_lo*e_thresh
        return Tcw_at_thresh + slope_hi*(e - e_thresh)

# ---------------------
# Generate strain array & compute tensions
# ---------------------
e_values = np.linspace(0, 0.15, 30)

t_cw_list = []
t_pm_list = []
sigma_list = []
p_list = []

for e in e_values:
    tcw = Tcw(e)
    tpm = Tpm(e)
    sigma = tcw + tpm
    p = 4.0 * sigma  # for a sphere (diam=1), p = 4*sigma
    
    t_cw_list.append(tcw)
    t_pm_list.append(tpm)
    sigma_list.append(sigma)
    p_list.append(p)

t_cw_list  = np.array(t_cw_list)
t_pm_list  = np.array(t_pm_list)
sigma_list = np.array(sigma_list)
p_list     = np.array(p_list)

# ---------------------
# Noise helpers
# ---------------------
np.random.seed(1)  # for reproducible noise; remove/change if you wish

def add_sparse_noise(ydata, fraction=0.15, scale=5e-4):
    """
    Adds random offsets to 'fraction' of the points, 
    leaving the rest almost smooth.
    """
    y_noisy = ydata.copy()
    n_pts = len(y_noisy)
    n_noisy = int(n_pts * fraction)
    idx_rand = np.random.choice(n_pts, size=n_noisy, replace=False)
    y_noisy[idx_rand] += scale * np.random.randn(n_noisy)
    return y_noisy

# ----------------------------------------------------------
# 1) Figure 1: Strain vs. Tension
#    - Slightly more noise than before
#    - Label black line as "PM tension"
# ----------------------------------------------------------
# Create a "more noisy" version for the strain plot
t_cw_noisy_strain   = add_sparse_noise(t_cw_list,  fraction=0.25, scale=8e-4)
# We'll treat the total tension similarly:
sigma_noisy_strain  = add_sparse_noise(sigma_list, fraction=0.25, scale=8e-4)

plt.figure(figsize=(5.5, 4.5))
plt.plot(e_values, t_cw_noisy_strain,  color='black', label='PM tension')
plt.plot(e_values, sigma_noisy_strain, color='red',   label=r'$\sigma$ (total)')
plt.xlabel('Strain')
plt.ylabel('Tension [N/m]')
plt.title('Tensions vs. Strain')
# No x-gap and start y-axis at 0
plt.xlim(e_values[0], e_values[-1])
plt.ylim(0, None)
plt.grid(True)
plt.legend()

# ----------------------------------------------------------
# 2) Figure 2: Internal Pressure vs. Tension
#    - EXACTLY the same as before
# ----------------------------------------------------------
# Create a "lightly noisy" version for the pressure plot
t_cw_noisy_press   = add_sparse_noise(t_cw_list,  fraction=0.15, scale=5e-4)
t_pm_noisy_press   = add_sparse_noise(t_pm_list,  fraction=0.15, scale=5e-4)
p_noisy_press      = add_sparse_noise(p_list,     fraction=0.15, scale=5e-4)

# Convert p to [0.1, 10] atm
p_min, p_max = p_noisy_press.min(), p_noisy_press.max()
factor = (10.0 - 0.1) / (p_max - p_min)
offset = 0.1 - factor * p_min
p_list_atm = factor * p_noisy_press + offset

plt.figure(figsize=(5.5, 4.5))
plt.plot(p_list_atm, t_cw_noisy_press, color='black', label='PG tension')
plt.plot(p_list_atm, t_pm_noisy_press, color='red',   label='PM tension')
plt.xlabel('Internal pressure [atm]')
plt.ylabel('Tension [N/m]')
plt.title('PG & PM tension vs. internal pressure')
plt.xlim(0.1, 10)
plt.ylim(0, None)
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
