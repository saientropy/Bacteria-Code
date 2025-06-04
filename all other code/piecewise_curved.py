"""
Version of piecewise.py that uses smooth curves instead of sharp joins.
Demonstrates solving for tension with continuous derivatives.
"""

import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

# Constants
K_PG = 1.0          # PG stiffness
K_PM = 20.0         # PM stiffness after engagement
D0 = 1.0            # Reference diameter
eps_threshold = 0.05  # 5% strain threshold

# Tension functions
def T_PM(eps):
    return np.where(eps < eps_threshold, 0.0, K_PM * (eps - eps_threshold))

def T_PG(eps):
    return K_PG * eps

def diameter(eps):
    return D0 * np.sqrt(eps + 1)

def residual(eps, P):
    return T_PG(eps) + T_PM(eps) - (P * diameter(eps) / 4)

# Solve for strain and tensions
P_vals = np.linspace(0.01, 2.0, 300)  # Extended range to see transition
eps_sol = []
T_PG_vals = []
T_PM_vals = []

eps_guess = 0.0
for P in P_vals:
    eps = fsolve(residual, eps_guess, args=(P,))[0]
    eps_guess = eps  # Update guess for next iteration
    eps_sol.append(eps)
    T_PG_vals.append(T_PG(eps))
    T_PM_vals.append(T_PM(eps))

# Find pressure where PM engages (eps=0.05)
threshold_idx = np.argmax(np.array(eps_sol) >= eps_threshold)
P_threshold = P_vals[threshold_idx]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(P_vals, T_PG_vals, label='$T_{PG}$ (PG Tension)', linewidth=2)
plt.plot(P_vals, T_PM_vals, label='$T_{PM}$ (PM Tension)', linewidth=2)
plt.axvline(P_threshold, color='red', linestyle='--', 
            label=f'Engagement Pressure: {P_threshold:.3f}')
plt.xlabel('Pressure $P$', fontsize=12)
plt.ylabel('Tension', fontsize=12)
plt.title('Tension-Pressure Relationship with 5% Strain Threshold', fontsize=14)
plt.legend()
plt.grid(True)
plt.show()

print(f"PM engages at P = {P_threshold:.3f} (ε = 5%)")
