import numpy as np
import matplotlib.pyplot as plt

# Set up the figure
plt.figure(figsize=(8, 6))

# Generate strain values
strain = np.linspace(0.05, 0.15, 1000)

# Define functions - making T_cw more clearly parabolic
def tcw(x):
    # Pure parabolic curve for T_cw
    return 0.5 + 500 * (x - 0.05)**2  # Parabolic curve starting at 0.5

def tpm(x):
    # Zero until threshold, then steep linear increase
    threshold = 0.065
    return np.maximum(0, 28 * (x - threshold))

# Plot only the T_cw and T_pm curves (no sigma)
plt.plot(strain, tcw(strain), 'k-', linewidth=2, label='T_cw')
plt.plot(strain, tpm(strain), 'k--', linewidth=2, label='T_pm')

# Add only the black dot on T_cw
black_dot_x = 0.09
black_dot_y = tcw(black_dot_x)
plt.plot(black_dot_x, black_dot_y, 'ko', markersize=10)

# Configure the plot
plt.xlim(0.05, 0.15)
plt.ylim(0, 4.0)
plt.xlabel('strain', fontsize=12)
plt.ylabel('tension 10$^{-2}$ N/m', fontsize=12)
plt.grid(False)

# Add only the T_cw and T_pm labels
plt.text(0.125, 2.0, 'T_cw ——', fontsize=12)
plt.text(0.125, 1.7, 'T_pm - - - -', fontsize=12)

# Display the plot
plt.tight_layout()
plt.savefig('tension_strain_plot.png', dpi=300)
plt.show()