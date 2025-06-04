"""
Plots a single phosphatidylethanolamine (PM) tension curve. Illustrates the
shape of the PM contribution alone.
"""

import numpy as np
import matplotlib.pyplot as plt
import math

# User variables:
pm_angle_deg = 55         # PM curve's takeoff angle in degrees
noise_amplitude = 0.02     # Adjust this value for more or less "wobble"

pm_slope = math.tan(math.radians(pm_angle_deg))  # Slope from the specified angle

# Define the x-axis range (0 to 4 atm)
x_max = 4
x_values = np.linspace(0, x_max, 300)

# Define the PM curve:
# For x <= 2, y = 0; for x > 2, y = pm_slope*(x-2)
pm_y_values = np.where(x_values <= 2, 0, pm_slope * (x_values - 2))
# Add noise:
pm_y_values_noisy = pm_y_values + np.random.normal(0, noise_amplitude, pm_y_values.shape)

# Create the plot with thin line
plt.figure(figsize=(8,6))
plt.plot(x_values, pm_y_values_noisy, label="PM", linewidth=1)

# Update axis labels:
plt.xlabel("Internal Pressure [atm]")
plt.ylabel("Tension [10^2 N/m]")
plt.title("Single Shell PM")
plt.xlim(0, x_max)
plt.ylim(0, 10)
plt.legend()
plt.grid(True)
plt.show()
