"""
Solves coupled strain-pressure equations with SciPy's fsolve. Plots the
resulting strain and tension curves.
"""

import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

# Given parameters (example values)
K_PG = 1.0
K_PM = 10.0
D_0 = 1
n = 4  # We have chosen n=2 as per discussion

def equation(epsilon, P, K_PG, K_PM, D_0):
    # Equation: K_PG*epsilon + K_PM*epsilon^2 - (P * D0 * sqrt(epsilon+1))/4 = 0
    return K_PG*epsilon + K_PM*(epsilon**n) - (P * D_0 * np.sqrt(epsilon + 1) / 4.0)

# Define a range of pressures:
P_values = np.linspace(0.1, 5.0, 100)  # from P=0.1 to 5.0 (adjust as needed)

epsilon_values = []
T_PG_values = []
T_PM_values = []

# Initial guess for epsilon (starting at low strain)
epsilon_guess = 0.0

for P in P_values:
    # Solve for epsilon:
    epsilon_sol = fsolve(lambda eps: equation(eps, P, K_PG, K_PM, D_0), epsilon_guess)
    epsilon_sol = epsilon_sol[0]
    
    # Update guess for continuity
    epsilon_guess = epsilon_sol
    
    # Calculate tensions:
    T_PG_val = K_PG*epsilon_sol
    T_PM_val = K_PM*(epsilon_sol**n)
    
    epsilon_values.append(epsilon_sol)
    T_PG_values.append(T_PG_val)
    T_PM_values.append(T_PM_val)

# Plot T_PG and T_PM vs P
plt.figure(figsize=(8,6))
plt.plot(P_values, T_PG_values, label='T_PG(P)')
plt.plot(P_values, T_PM_values, label='T_PM(P)')
plt.xlabel('Pressure (P)')
plt.ylabel('Tension')
plt.title('Tensions vs Pressure for n='+str(n))
plt.legend()
plt.grid(True)
plt.show()
