import numpy as np
import matplotlib.pyplot as plt

# User variables:
noise_amplitude = 0.01     # Adjust this value for more or less "wobble"
pg_slope = 5/2            # Linear slope for PG across entire domain (0–4 atm)

# Define the x-axis range (0 to 4 atm)
x_max = 4
x_values = np.linspace(0, x_max, 300)

# Define the purely linear PG curve (y = slope * x)
pg_y_values = pg_slope * x_values

# Add noise
pg_y_values_noisy = pg_y_values + np.random.normal(0, noise_amplitude, pg_y_values.shape)

# Create the plot for PG only
plt.figure(figsize=(8,6))
plt.plot(x_values, pg_y_values_noisy, label="PG (linear)", linewidth=1)

# Update axis labels
plt.xlabel("Internal Pressure [atm]")
plt.ylabel("Tension [10^2 N/m]")
plt.title("Single Shell PG")
plt.xlim(0, x_max)
plt.ylim(0, 10)
plt.legend()
plt.grid(True)
plt.show()
