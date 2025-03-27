import numpy as np
import matplotlib.pyplot as plt
import math

# User variables:
pm_angle_deg = 55      # PM curve's takeoff angle in degrees
noise_amplitude = 0.02  # Adjust this value for more or less "wobble"

pm_slope = math.tan(math.radians(pm_angle_deg))  # Slope from the specified angle

# Define the x-axis range (0 to 4 atm)
x_max = 4
x_values = np.linspace(0, x_max, 300)

# Define the PM curve:
# For x <= 2, y = 0; for x > 2, y = pm_slope*(x-2)
pm_y_values = np.where(x_values <= 2, 0, pm_slope * (x_values - 2))
# Add noise:
pm_y_values_noisy = pm_y_values + np.random.normal(0, noise_amplitude, pm_y_values.shape)

# Define the PG curve:
# For x <= 2, linearly from (0,0) to (2,5)
# For x > 2, extend with the same slope as PM
pg_y_values = np.where(x_values <= 2, (5/2)*x_values, 5 + pm_slope * (x_values - 2))
# Add noise:
pg_y_values_noisy = pg_y_values + np.random.normal(0, noise_amplitude, pg_y_values.shape)

# Define the Total curve:
# For x <= 2, linearly from (0,0) to (2,5.2)
# For x > 2, extend with the same slope (5.2/2 = 2.6)
total_slope = 5.2 / 2  # = 2.6
total_y_values = np.where(x_values <= 2, total_slope * x_values, 5.2 + total_slope * (x_values - 2))
# Add noise:
total_y_values_noisy = total_y_values + np.random.normal(0, noise_amplitude, total_y_values.shape)

# Create the plot with thin lines and specified colors:
plt.figure(figsize=(8,6))
plt.plot(x_values, pm_y_values_noisy, label="PM", color="blue", linewidth=1)
plt.plot(x_values, pg_y_values_noisy, label="PG", color="black", linewidth=1)
plt.plot(x_values, total_y_values_noisy, label="Total", color="red", linewidth=1)

# Add a horizontal line at y = 1.5 labeled as "rupture tension"
plt.axhline(y=1.5, color="green", linestyle="--", linewidth=1, label="rupture tension")

# Updated axis labels and title:
plt.xlabel("Interal Pressure [atm]")
plt.ylabel("Tension [ 10^2 N/m]")
plt.title("Experimental-like Curves with Noise and Rupture Tension")
plt.xlim(0, x_max)
plt.ylim(0, 10)
plt.legend()
plt.grid(True)
plt.show()
