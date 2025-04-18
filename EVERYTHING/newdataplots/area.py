import os
# Disable modules that may be causing compatibility issues (without terminal commands)
os.environ["NUMEXPR_DISABLE"] = "1"
os.environ["PANDAS_IGNORE_BOTTLENECK"] = "1"

import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd

# User variables:
pm_angle_deg = 80  # PM curve's takeoff angle in degrees
noise_amplitude = 0.036  # Adjust this value for more or less "wobble"
scale_factor = 10  # Scaling factor for PM curve

# Compute effective slope for PM (after x = 20 micron²)
pm_slope = math.tan(math.radians(pm_angle_deg)) / scale_factor

# Define the x-axis range (15 to 40 micron²)
x_min = 15
x_max = 40
x_values = np.linspace(x_min, x_max, 300)

# -------------------------
# PM Curve Definition:
# -------------------------
# For x < x_takeoff, PM = 0.
# For x >= x_takeoff, PM = pm_slope*(x - x_takeoff)
x_takeoff = 30
pm_y_values = np.where(x_values < x_takeoff, 0, pm_slope * (x_values - x_takeoff))
pm_y_noisy = pm_y_values + np.random.normal(0, noise_amplitude, pm_y_values.shape)
pm_y_noisy = np.clip(pm_y_noisy, 0, None)  # Remove negative values

# -------------------------
# PG Curve Definition:
# -------------------------
# PG is defined as a linear function from (15,0) to (40,pg_end)
pg_start = 0
pg_end = 9.5
pg_slope = (pg_end - pg_start) / (x_max - x_min)
pg_y_values = pg_start + pg_slope * (x_values - x_min)
pg_y_noisy = pg_y_values + np.random.normal(0, noise_amplitude, pg_y_values.shape)
pg_y_noisy = np.clip(pg_y_noisy, 0, None)  # Remove negative values

# -------------------------
# Total Curve Definition:
# -------------------------
# Total is the sum of the noisy PG and PM curves at each x
total_y_noisy = pg_y_noisy + pm_y_noisy
total_y_noisy = np.clip(total_y_noisy, 0, None)  # Remove negative values

# -------------------------
# Plotting:
# -------------------------
plt.figure(figsize=(8,6))
plt.plot(x_values, pg_y_noisy, label="PG", color="black", linewidth=1)
plt.plot(x_values, pm_y_noisy, label="PM", color="blue", linewidth=1)
plt.plot(x_values, total_y_noisy, label="Total", color="red", linewidth=1)
plt.axhline(y=1.5, color="green", linestyle="--", linewidth=1, label="rupture tension")
plt.xlabel("Internal Surface Area of the PG [µ²]")
plt.ylabel("Tension [ 10^2 N/m]")
plt.title("Tension vs Internal Surface Area")
plt.xlim(x_min, x_max)
plt.ylim(0, 10)
plt.legend()
plt.grid(True)
plt.show()

# -------------------------
# Saving data to CSV:
# -------------------------
data = pd.DataFrame({
    "Internal Surface Area [µ²]": x_values,
    "PG Tension [10^2 N/m]": pg_y_noisy,
    "PM Tension [10^2 N/m]": pm_y_noisy,
    "Total Tension [10^2 N/m]": total_y_noisy
})
csv_filename = "plot_data_second_1.csv"
data.to_csv(csv_filename, index=False)

# -------------------------
# Print the first 10 values:
# -------------------------
print("First 10 rows of", csv_filename)
print(data.head(10))
