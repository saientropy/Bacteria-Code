import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

# ========== 1) MODEL PARAMETERS ==========
K_PG   = 1.0      # "Stiffness" for PG layer
K_PM   = 20.0     # "Stiffness" for PM (once engaged)
D0     = 1.0      # Reference diameter (arbitrary units)

# List of threshold strains to compare:
threshold_list = [0.02, 0.05, 0.10]  # 2%, 5%, 10%

# Pressure range (could be changed as desired)
P_values = np.linspace(0.01, 5.0, 100)

# ========== 2) HELPER FUNCTIONS ==========

def diameter(eps):
    """
    Diameter as a function of strain eps, given:
        eps = (D^2 / D0^2) - 1
        => D = D0 * sqrt(eps + 1)
    """
    return D0 * np.sqrt(eps + 1.0)

def T_PG(eps):
    """
    Linear tension in the PG layer.
    T_PG = K_PG * eps
    """
    return K_PG * eps

def T_PM(eps, threshold):
    """
    Piecewise tension in the PM layer, given a threshold strain.
      - Zero if eps < threshold
      - K_PM*(eps - threshold) if eps >= threshold
    """
    if eps < threshold:
        return 0.0
    else:
        return K_PM * (eps - threshold)

def residual(eps, P, threshold):
    """
    Tension balance for a spherical shell:
        T_PG(eps) + T_PM(eps, threshold) = (P * D(eps)) / 4
    We'll pass this to fsolve to find eps for a given P & threshold.
    """
    lhs = T_PG(eps) + T_PM(eps, threshold)
    rhs = (P * diameter(eps)) / 4.0
    return lhs - rhs

# ========== 3) LOOP OVER THRESHOLDS AND COLLECT SOLUTIONS ==========

# We'll store curves for each threshold to plot them all.
all_eps_data = {}      # dictionary: threshold -> array of strains
all_TPG_data = {}      # dictionary: threshold -> array of T_PG
all_TPM_data = {}      # dictionary: threshold -> array of T_PM

initial_guess = 0.0  # Starting guess for eps in fsolve

for thresh in threshold_list:
    eps_list = []
    TPG_list = []
    TPM_list = []

    for P in P_values:
        # Solve for eps:
        sol = fsolve(lambda e: residual(e, P, thresh), initial_guess)
        eps_sol = sol[0]
        # Update guess for next iteration (to speed convergence)
        initial_guess = eps_sol

        # Calculate tensions
        T_pg = T_PG(eps_sol)
        T_pm = T_PM(eps_sol, thresh)

        eps_list.append(eps_sol)
        TPG_list.append(T_pg)
        TPM_list.append(T_pm)

    all_eps_data[thresh] = np.array(eps_list)
    all_TPG_data[thresh] = np.array(TPG_list)
    all_TPM_data[thresh] = np.array(TPM_list)

# ========== 4) MAKE SUBPLOTS: T vs. P (LEFT), T vs. eps (RIGHT) ==========

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

ax_left  = axes[0]  # for T vs. P
ax_right = axes[1]  # for T vs. eps

# We'll store handles/labels for the legend
legend_handles = []

for thresh in threshold_list:
    eps_array = all_eps_data[thresh]
    TPG_array = all_TPG_data[thresh]
    TPM_array = all_TPM_data[thresh]

    # ----- Plot T_PG & T_PM vs P (left subplot) -----
    pg_line = ax_left.plot(
        P_values, TPG_array, 
        label=f"T_PG (threshold={int(thresh*100)}%)"
    )
    pm_line = ax_left.plot(
        P_values, TPM_array, 
        label=f"T_PM (threshold={int(thresh*100)}%)"
    )

    # We can keep track of line handles if we want a custom legend:
    # (But we'll rely on the standard approach of using the line labels)

    # ----- Plot T_PG & T_PM vs eps (right subplot) -----
    ax_right.plot(
        eps_array, TPG_array,
        label=f"T_PG (threshold={int(thresh*100)}%)"
    )
    ax_right.plot(
        eps_array, TPM_array,
        label=f"T_PM (threshold={int(thresh*100)}%)"
    )

# Add vertical lines at each threshold on the right subplot:
for thresh in threshold_list:
    ax_right.axvline(
        thresh, color="gray", linestyle="--", alpha=0.5,
        label=f"Threshold = {int(thresh*100)}%"
    )

# ========== 5) STYLING AND LEGENDS ==========

# Set axis labels, titles, etc.
ax_left.set_xlabel("Pressure P")
ax_left.set_ylabel("Tension")
ax_left.set_title("Tension vs. Pressure")

ax_right.set_xlabel("Strain (eps)")
ax_right.set_ylabel("Tension")
ax_right.set_title("Tension vs. Strain")

# Place a single legend on the right side of the figure (outside the subplots).
# We can gather labels from both subplots by doing:
handles, labels = axes[0].get_legend_handles_labels()
handles2, labels2 = axes[1].get_legend_handles_labels()

# Combine them (there will be duplicates, so we might remove them using a dict):
combined = dict(zip(labels+labels2, handles+handles2)) 
all_labels  = list(combined.keys())
all_handles = list(combined.values())

# Make room on the right for the legend
plt.tight_layout(rect=[0, 0, 0.78, 1])  # shrink main plot area

fig.legend(
    all_handles, all_labels,
    loc="center right",   # put legend to the right
    bbox_to_anchor=(0.98, 0.5),  # (x=0.98, y=0.5) near right edge
    title="Curves & Thresholds"
)

plt.show()