import numpy as np
import matplotlib.pyplot as plt

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(8, 6))

# Generate strain values
strain = np.linspace(0.05, 0.15, 1000)

# Define the functions: PG (parabolic) and PM (linear after threshold)
def pg(x):
    # Parabolic curve for PG starting at 0.5
    return 0.5 + 500 * (x - 0.05)**2

def pm(x):
    # Zero until threshold, then steep linear increase
    threshold = 0.065
    return np.maximum(0, 28 * (x - threshold))

# Plot the PG and PM curves with distinct colors and line styles
ax.plot(strain, pg(strain), color='darkgreen', linewidth=2, label='PG')
ax.plot(strain, pm(strain), color='darkred', linestyle='--', linewidth=2, label='PM')

# Add a black dot on the PG curve at strain = 0.09
black_dot_x = 0.09
black_dot_y = pg(black_dot_x)
ax.plot(black_dot_x, black_dot_y, 'ko', markersize=8)

# Configure axis limits and labels
ax.set_xlim(0.05, 0.15)
ax.set_ylim(0, 4.0)
ax.set_xlabel('strain', fontsize=14)
ax.set_ylabel('tension 10$^{-2}$ N/m', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=12)

# Place the legend in the upper left corner
ax.legend(loc='upper left', fontsize=12, frameon=False)

plt.tight_layout()
plt.savefig('tension_strain_plot.png', dpi=300)
plt.show()
