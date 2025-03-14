import matplotlib.pyplot as plt
import numpy as np

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

plt.figure(figsize=(5, 5))

angle = 68  # Reduced angle for both curves

# PM (Plasma Membrane) curve
pm_x = [0, 9, 20]  
pm_y = [0, 0.5, 0.5 + np.tan(np.radians(angle)) * (20 - 9)]

# PG (Peptidoglycan) curve
pg_x = [0, 9, 20]  
pg_y = [0, 17, 10 + np.tan(np.radians(angle)) * (20 - 9)]
pg_y[2] = 10 + (20 - 9) * np.tan(np.radians(angle))  # Adjusting PG tilt

# "Total" line data (with its own definition)
ext_x = [0, 9, 20]  
ext_y = [0, 18, 11 + (23 - 3) * ((23 - 0) / 9)]

# Generate wobbly versions for all curves
x_pm_w, y_pm_w = add_wobble(pm_x, pm_y, amplitude=0.3)
x_pg_w, y_pg_w = add_wobble(pg_x, pg_y, amplitude=0.4)
x_ext_w, y_ext_w = add_wobble(ext_x, ext_y, amplitude=0.3)  # Wobble for Total line

# Scale tension values (y) from original 0–30 to 0–3 (in 10^-2 N/m units)
factor_y = 3 / 30
y_pm_w = y_pm_w * factor_y
y_pg_w = y_pg_w * factor_y
y_ext_w = y_ext_w * factor_y

# Remap the x values (original 0–20) to inner diameter in microns (2 to 4 µm)
x_pm_w = 2 + (x_pm_w / 20) * (4 - 2)
x_pg_w = 2 + (x_pg_w / 20) * (4 - 2)
x_ext_w = 2 + (x_ext_w / 20) * (4 - 2)

# Compute surface area from diameter (A = π * d^2)
A_pm = np.pi * x_pm_w**2
A_pg = np.pi * x_pg_w**2
A_ext = np.pi * x_ext_w**2

# Plot the curves against surface area
plt.plot(A_pm, y_pm_w, label='T_PM- thin shell', linestyle='-', linewidth=2)
plt.plot(A_pg, y_pg_w, label='T_PG (t = 6 nm)', linestyle='-', linewidth=2)
plt.plot(A_ext, y_ext_w, label='Total', linestyle='--', linewidth=2, color='red')

# Add a horizontal line at y = 1.5 for rupture tension
plt.axhline(y=1.5, color='black', linestyle='--', linewidth=1, label='Rupture Tension')


# Set axis limits: x from 4π to 12.25π µm²; y from 0 to 3 (tension in 10^-2 N/m)
plt.xlim(np.pi * 2**2, np.pi * 3.5**2)
plt.ylim(0, 3)

plt.xlabel("Inner Surface area of PG layer (µm²)")
plt.ylabel("Tension in 10^-2 (N/m)")
plt.legend()
plt.grid()

plt.show()
