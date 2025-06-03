import numpy as np
import matplotlib.pyplot as plt
import math

# User variables:
pm_angle_deg = 55 # PM curve's takeoff angle in degrees
noise_amplitude = 0.027 # Adjust this value for more or less "wobble"

pm_slope = math.tan(math.radians(pm_angle_deg))  # Slope from the specified angle

# Define the breakpoint for "take off"
x_break = 2.0

# Define the x-axis range (0 to 4 atm)
x_max = 4
x_values = np.linspace(0, x_max, 300)

# ---------------------
# PM Curve Definition:
# ---------------------
# For x <= x_break, y = 0; for x > x_break, y = pm_slope*(x - x_break)
pm_y_values = np.where(x_values <= x_break, 0, pm_slope * (x_values - x_break))
pm_y_values_noisy = pm_y_values + np.random.normal(0, noise_amplitude, pm_y_values.shape)

# ---------------------
# PG Curve Definition:
# ---------------------
# For x <= x_break, PG is linear from (0,0) to (x_break, pg_break_value)
pg_break_value = (5/2) * x_break  # For x_break=2.5, pg_break_value = 6.25
# For x > x_break, PG = pg_break_value + pm_slope*(x - x_break)
pg_y_values = np.where(x_values <= x_break, (5/2) * x_values, pg_break_value + pm_slope * (x_values - x_break))
pg_y_values_noisy = pg_y_values + np.random.normal(0, noise_amplitude, pg_y_values.shape)

# ---------------------
# Total Curve Definition:
# ---------------------
# For x <= x_break, Total is linear from (0,0) to (x_break, pg_break_value + 0.2)
total_break_value = pg_break_value + 0.2  # That is 6.25 + 0.2 = 6.45
# Slope for Total for x <= x_break:
total_slope = total_break_value / x_break  
# For x > x_break, Total = total_break_value + total_slope*(x - x_break)
total_y_values = np.where(x_values <= x_break, total_slope * x_values, total_break_value + total_slope * (x_values - x_break))
total_y_values_noisy = total_y_values + np.random.normal(0, noise_amplitude, total_y_values.shape)

# ---------------------
# Plotting:
# ---------------------
plt.figure(figsize=(8,6))
plt.plot(x_values, pm_y_values_noisy, label="PM", color="blue", linewidth=1)
plt.plot(x_values, pg_y_values_noisy, label="PG", color="black", linewidth=1)
plt.plot(x_values, total_y_values_noisy, label="Total", color="red", linewidth=1)

# Add a horizontal line at y = 1.5 labeled as "rupture tension"
plt.axhline(y=1.5, color="green", linestyle="--", linewidth=1, label="rupture tension")

# Updated axis labels and title:
plt.xlabel("Internal Pressure [atm]")
plt.ylabel("Tension [ 10^2 N/m]")
plt.title("Tension vs Pressure")
plt.xlim(0, x_max)
plt.ylim(0, 10)
plt.legend()
plt.grid(True)
plt.show()
