import matplotlib.pyplot as plt
import numpy as np
import csv

def add_wobble(x, y, amplitude=0.2, num_points=200):
    """
    Given arrays x and y defining a line, return new x and y arrays with many intermediate points
    and a slight, smoothed random perturbation added. The endpoints remain fixed.
    """
    t = np.linspace(0, 1, num_points)
    # Interpolate the original points to a finer resolution
    x_interp = np.interp(t, np.linspace(0, 1, len(x)), x)
    y_interp = np.interp(t, np.linspace(0, 1, len(y)), y)
    
    # Generate random noise and fix the endpoints to zero
    noise = np.random.normal(scale=amplitude, size=num_points)
    noise[0] = 0
    noise[-1] = 0
    
    # Smooth the noise using a moving average filter
    window_size = 5
    window = np.ones(window_size) / window_size
    noise = np.convolve(noise, window, mode='same')
    
    return x_interp, y_interp + noise

# Define the plot size
plt.figure(figsize=(5, 5))

# Define the common reduced angle
angle = 68  # Reduced angle for both curves

# Define the PM (Plasma Membrane) curve
pm_x = [0, 5, 20]  # X-axis (Pressure)
pm_y = [0, 0.5, 0.5 + np.tan(np.radians(angle)) * (20 - 5)]  # Y-axis (Tension)

# Define the PG (Peptidoglycan) curve
pg_x = [0, 5, 20]  # X-axis (Pressure)
pg_y = [0, 16, 10 + np.tan(np.radians(angle)) * (20 - 5)]  # Y-axis (Tension)
pg_y[2] = 10 + (20 - 5) * np.tan(np.radians(angle))  # Adjusting PG tilt correctly

# Define the additional linear extension ("Total")
ext_x = [0, 5, 20]  # X-axis (Pressure)
ext_y = [0, 18, 11 + (20 - 5) * ((17 - 0) / 5)]  # Linear extension based on slope

# Generate wobbly versions for all curves with amplitude=0.2
x_pm_w, y_pm_w = add_wobble(pm_x, pm_y, amplitude=0.3)
x_pg_w, y_pg_w = add_wobble(pg_x, pg_y, amplitude=0.3)
x_ext_w, y_ext_w = add_wobble(ext_x, ext_y, amplitude=0.1)

# Conversion factors: 
#   x: from 0–20 to 0–10 atm → factor_x = 10/20 = 0.5
#   y: from 0–30 to 0–3 N/m → factor_y = 3/30
factor_x = 0.5
factor_y = 3 / 30

# Scale the wobbly data accordingly
x_pm_w = x_pm_w * factor_x
y_pm_w = y_pm_w * factor_y

x_pg_w = x_pg_w * factor_x
y_pg_w = y_pg_w * factor_y

x_ext_w = x_ext_w * factor_x
y_ext_w = y_ext_w * factor_y

# Plot the curves
plt.plot(x_pm_w, y_pm_w, label='T_Plasma Membrane (PM)', linestyle='-', linewidth=2)
plt.plot(x_pg_w, y_pg_w, label='T_Peptidoglycan (PG)', linestyle='-', linewidth=2)
plt.plot(x_ext_w, y_ext_w, label='Total', linestyle='--', linewidth=2, color='red')

# Add a horizontal line at y = 1 for rupture tension
plt.axhline(y=1.5, color='black', linestyle='--', linewidth=1, label='Rupture Tension')

# Set new axis limits for the converted units
plt.xlim(0, 10)
plt.ylim(0, 3)

# Update labels with units
plt.xlabel("Pressure (atm)")
plt.ylabel("Tension in 10^-2 (N/m)")
plt.legend()
plt.grid()

# Write the data to a CSV file
with open('curves_data.csv', 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    # Write header
    csv_writer.writerow(["Pressure (atm)", "T_Plasma_Membrane", "T_Peptidoglycan", "T_Total"])
    # Assumes all arrays have the same length (num_points)
    for i in range(len(x_pm_w)):
        csv_writer.writerow([x_pm_w[i], y_pm_w[i], y_pg_w[i], y_ext_w[i]])

# Show plot
plt.show()
