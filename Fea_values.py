import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
# 1) Define parameter sweeps (E. coli-inspired)
# ------------------------------------------------------------------------------
n_values      = [3, 4]          # Strain hardening exponents (dimensionless)
K_PG_values   = [10e6, 20e6]    # PG stiffness in Pa (10 MPa, 20 MPa)
K_PM_values   = [0.3e6, 0.5e6]  # PM stiffness in Pa (0.3 MPa, 0.5 MPa)
D_0_values    = [1.5e-6, 2.0e-6]# Initial diameter in meters (1.5 μm, 2.0 μm)

# Pressure range: 0.1 MPa to 0.5 MPa
P_values = np.linspace(0.1e6, 0.5e6, 5)  # [0.1, 0.2, 0.3, 0.4, 0.5] MPa in Pa

# ------------------------------------------------------------------------------
# 2) Define the equilibrium equation
# ------------------------------------------------------------------------------
def equilibrium_equation(epsilon, P, K_PG, K_PM, n, D_0):
    """
    K_PG * epsilon + K_PM * (epsilon^n) = (P * D_0 * sqrt(epsilon + 1)) / 4
    """
    return (K_PG * epsilon
            + K_PM * (epsilon**n)
            - (P * D_0 * np.sqrt(max(epsilon + 1, 1e-14)) / 4.0))

# ------------------------------------------------------------------------------
# 3) Dictionary to store all results
# ------------------------------------------------------------------------------
results = {}

# ------------------------------------------------------------------------------
# 4) Nested loops: sweep each parameter combination and solve with fsolve
# ------------------------------------------------------------------------------
for n in n_values:
    for K_PG in K_PG_values:
        for K_PM in K_PM_values:
            for D_0 in D_0_values:
                
                epsilon_list = []
                T_PG_list    = []
                T_PM_list    = []
                D_list       = []
                
                # Start with an initial guess for epsilon
                epsilon_guess = 0.0
                
                for P in P_values:
                    eps_solution = fsolve(
                        lambda eps: equilibrium_equation(eps, P, K_PG, K_PM, n, D_0),
                        epsilon_guess
                    )
                    eps_sol = eps_solution[0]
                    
                    # If we get an invalid solution (ε < -1), clamp
                    if eps_sol < -1:
                        print(f"[Warning] Invalid strain ε={eps_sol:.4f} < -1 "
                              f"at P={P/1e6:.2f} MPa. Clamping to -0.9999.")
                        eps_sol = -0.9999
                    
                    # Update guess for next iteration
                    epsilon_guess = eps_sol
                    
                    # Compute tensions (kN/m if we divide by 1e3 later)
                    T_pg = K_PG * eps_sol
                    T_pm = K_PM * (eps_sol ** n)
                    
                    # Compute new diameter
                    D_current = D_0 * np.sqrt(max(eps_sol + 1, 1e-14))
                    
                    epsilon_list.append(eps_sol)
                    T_PG_list.append(T_pg)
                    T_PM_list.append(T_pm)
                    D_list.append(D_current)
                
                # Store in dictionary
                key = (n, K_PG, K_PM, D_0)
                results[key] = {
                    'P': P_values,
                    'epsilon': np.array(epsilon_list),
                    'T_PG':    np.array(T_PG_list),
                    'T_PM':    np.array(T_PM_list),
                    'D':       np.array(D_list),
                }

# ------------------------------------------------------------------------------
# 5) Make one figure with four subplots in a 2x2 grid
# ------------------------------------------------------------------------------
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

ax_T_PG     = axes[0, 0]
ax_T_PM     = axes[0, 1]
ax_epsilon  = axes[1, 0]
ax_diameter = axes[1, 1]

# We will collect line handles and labels from *all* subplots,
# then make ONE combined legend at the end.
all_handles = []
all_labels  = []

# Some style cycles
colors        = ['blue', 'green', 'red', 'orange', 'purple', 'magenta', 'brown', 'gray']
linestyles    = ['-', '--', '-.', ':']
marker_styles = ['o', 's', '^', 'd', 'v', '<', '>', 'p']

combo_idx = 0
for (n, K_PG, K_PM, D_0), data in results.items():
    P_MPa = data['P'] / 1e6  # convert Pa -> MPa
    
    # Short label for this combination
    # (Removes the newline and is a bit shorter for a multi-col legend)
    label_str = (
        f"n={n}, Kpg={K_PG/1e6:.1f}MPa, "
        f"Kpm={K_PM/1e6:.1f}MPa, D0={D_0*1e6:.1f}µm"
    )
    
    c  = colors[combo_idx % len(colors)]
    ls = linestyles[combo_idx % len(linestyles)]
    mk = marker_styles[combo_idx % len(marker_styles)]
    
    # Plot Peptidoglycan tension (in kN/m)
    line_pg = ax_T_PG.plot(
        P_MPa, data['T_PG'] / 1e3,
        color=c, linestyle=ls, marker=mk, label=label_str
    )
    
    # Plot Plasma membrane tension (in kN/m)
    line_pm = ax_T_PM.plot(
        P_MPa, data['T_PM'] / 1e3,
        color=c, linestyle=ls, marker=mk, label=label_str
    )
    
    # Plot strain (dimensionless)
    line_eps = ax_epsilon.plot(
        P_MPa, data['epsilon'],
        color=c, linestyle=ls, marker=mk, label=label_str
    )
    
    # Plot diameter (in μm)
    line_d = ax_diameter.plot(
        P_MPa, data['D'] * 1e6,
        color=c, linestyle=ls, marker=mk, label=label_str
    )
    
    # Each 'plot(...)' returns a list of line handles (usually just 1).
    # We'll collect the handle + the label from the first line in each subplot.
    # But truly we only need one line per combo to represent the label.
    # Here, let's just grab the PG line handle for the overall legend, for instance.
    all_handles.append(line_pg[0])
    all_labels.append(label_str)
    
    combo_idx += 1

# ------------------------------------------------------------------------------
# 6) Configure each subplot
# ------------------------------------------------------------------------------
# Peptidoglycan Tension
ax_T_PG.set_xlabel('Pressure (MPa)')
ax_T_PG.set_ylabel('PG Tension (kN/m)')
ax_T_PG.set_title('Peptidoglycan Tension vs. Pressure')
ax_T_PG.grid(True)

# Plasma Membrane Tension
ax_T_PM.set_xlabel('Pressure (MPa)')
ax_T_PM.set_ylabel('PM Tension (kN/m)')
ax_T_PM.set_title('Plasma Membrane Tension vs. Pressure')
ax_T_PM.grid(True)

# Strain
ax_epsilon.set_xlabel('Pressure (MPa)')
ax_epsilon.set_ylabel('Strain (dimensionless)')
ax_epsilon.set_title('Strain vs. Pressure')
ax_epsilon.grid(True)

# Diameter
ax_diameter.set_xlabel('Pressure (MPa)')
ax_diameter.set_ylabel('Diameter (µm)')
ax_diameter.set_title('Diameter vs. Pressure')
ax_diameter.grid(True)

# ------------------------------------------------------------------------------
# 7) Create ONE combined legend for all lines
#    Place it outside the plot area; multiple columns so it doesn't get too tall.
# ------------------------------------------------------------------------------
fig.tight_layout()
# Move the right edge of the subplots left to leave room for the legend
fig.subplots_adjust(right=0.62)

fig.legend(
    all_handles, all_labels,
    loc='upper left',
    bbox_to_anchor=(0.65, 0.98),  # Adjust as needed
    borderaxespad=0.,
    fontsize=8,
    ncol=1  # Try 1, 2, or 3 columns
)

plt.show()

# ------------------------------------------------------------------------------
# 8) (Optional) Print or inspect data for a specific combination
# ------------------------------------------------------------------------------
example_key = (4, 20e6, 0.5e6, 2.0e-6)
if example_key in results:
    ex_data = results[example_key]
    P_MPa_ex = ex_data['P'] / 1e6
    print("\nExample data for", example_key, ":")
    for Pval, eps, Tpg, Tpm, Dval in zip(
        P_MPa_ex, ex_data['epsilon'], ex_data['T_PG'], ex_data['T_PM'], ex_data['D']
    ):
        print(f"  P={Pval:.2f} MPa | ε={eps:.4f} | "
              f"PG Tension={Tpg/1e3:.5f} kN/m | "
              f"PM Tension={Tpm/1e3:.5f} kN/m | "
              f"D={Dval*1e6:.4f} µm")