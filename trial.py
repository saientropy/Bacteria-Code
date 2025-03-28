import numpy as np
import matplotlib.pyplot as plt
import math

# Define common x-axis range (0 to 4 atm)
x_max = 4
x_values = np.linspace(0, x_max, 300)

# ---------------------
# Code 1: PG (linear) from Code 1
# ---------------------
noise_amplitude_code1 = 0.01
pg_slope_code1 = 5 / 2  # 2.25
pg_y_code1 = pg_slope_code1 * x_values
pg_y_code1_noisy = pg_y_code1 + np.random.normal(0, noise_amplitude_code1, pg_y_code1.shape)

# ---------------------
# Code 2: PM curve from Code 2
# ---------------------
pm_angle_deg_code2 = 55
noise_amplitude_code2 = 0.02
pm_slope_code2 = math.tan(math.radians(pm_angle_deg_code2))
pm_y_code2 = np.where(x_values <= 2, 0, pm_slope_code2 * (x_values - 2))
pm_y_code2_noisy = pm_y_code2 + np.random.normal(0, noise_amplitude_code2, pm_y_code2.shape)

# ---------------------
# Code 3: Combined curves (PM, PG, and Total) from Code 3
# ---------------------
pm_angle_deg_code3 = 55
noise_amplitude_code3 = 0.027
pm_slope_code3 = math.tan(math.radians(pm_angle_deg_code3))
x_break = 2.0

# PM curve (Code 3)
pm_y_code3 = np.where(x_values <= x_break, 0, pm_slope_code3 * (x_values - x_break))
pm_y_code3_noisy = pm_y_code3 + np.random.normal(0, noise_amplitude_code3, pm_y_code3.shape)

# PG curve (Code 3)
pg_break_value = (5/2) * x_break  # For x_break=2, pg_break_value = 5
pg_y_code3 = np.where(x_values <= x_break, (5/2) * x_values, pg_break_value + pm_slope_code3 * (x_values - x_break))
pg_y_code3_noisy = pg_y_code3 + np.random.normal(0, noise_amplitude_code3, pg_y_code3.shape)

# Total curve (Code 3)
total_break_value = pg_break_value + 0.2  # 5 + 0.2 = 5.2
total_slope = total_break_value / x_break  # 5.2 / 2 = 2.6
total_y_code3 = np.where(x_values <= x_break, total_slope * x_values, total_break_value + total_slope * (x_values - x_break))
total_y_code3_noisy = total_y_code3 + np.random.normal(0, noise_amplitude_code3, total_y_code3.shape)

# ---------------------
# Plotting all curves on a single figure
# ---------------------
plt.figure(figsize=(8, 6))

# Plot Code 1 PG (linear)
plt.plot(x_values, pg_y_code1_noisy, label="PG (linear) [Code 1]",
         linestyle="--", linewidth=1, color="magenta")

# Plot Code 2 PM curve
plt.plot(x_values, pm_y_code2_noisy, label="PM [Code 2]",
         linestyle=":", linewidth=1, color="orange")

# Plot Code 3 curves
plt.plot(x_values, pm_y_code3_noisy, label="PM [Code 3]",
         color="blue", linewidth=1)
plt.plot(x_values, pg_y_code3_noisy, label="PG [Code 3]",
         color="black", linewidth=1)
plt.plot(x_values, total_y_code3_noisy, label="Total [Code 3]",
         color="red", linewidth=1)

# Plot rupture tension horizontal line at y = 1.5
plt.axhline(y=1.5, color="green", linestyle="--", linewidth=1, label="rupture tension")

# Update axis labels and title
plt.xlabel("Internal Pressure [atm]")
plt.ylabel("Tension [10^2 N/m]")
plt.title("Superimposed Tension vs Pressure Curves")
plt.xlim(0, x_max)
plt.ylim(0, 10)
plt.legend()
plt.grid(True)
plt.show()
