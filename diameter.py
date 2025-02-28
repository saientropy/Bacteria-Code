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

angle = 50  # Reduced angle for both curves

# PM (Plasma Membrane) curve
pm_x = [0, 5, 20]  
pm_y = [0, 0.5, 0.5 + np.tan(np.radians(angle)) * (20 - 5)]

# PG (Peptidoglycan) curve
pg_x = [0, 5, 20]  
pg_y = [0, 10, 10 + np.tan(np.radians(angle)) * (20 - 5)]
pg_y[2] = 10 + (20 - 5) * np.tan(np.radians(angle))  # Adjusting PG tilt

# Generate wobbly versions
x_pm_w, y_pm_w = add_wobble(pm_x, pm_y, amplitude=0.2)
x_pg_w, y_pg_w = add_wobble(pg_x, pg_y, amplitude=0.2)

# Scale tension values (y) from original 0–30 to 0–3 (in 10^-2 N/m units)
factor_y = 3 / 30
y_pm_w = y_pm_w * factor_y
y_pg_w = y_pg_w * factor_y

# Remap the x values (original 0–20) to internal diameter in microns:
#   original 0   -> 2 µm  (if the bladder is very slack, the internal diameter is small)
#   original 20  -> 4 µm  (fully inflated, the ball reaches 4 µm in diameter)
x_pm_w = 2 + (x_pm_w / 20) * (4 - 2)
x_pg_w = 2 + (x_pg_w / 20) * (4 - 2)

# Plot the PM and PG curves
plt.plot(x_pm_w, y_pm_w, label='T_Plasma Membrane- thin shell', linestyle='-', linewidth=2)
plt.plot(x_pg_w, y_pg_w, label='T_Peptidoglycan (thickness 6 nm)', linestyle='-', linewidth=2)

# Add a horizontal line at y = 1 for rupture tension
plt.axhline(y=1, color='black', linestyle='--', linewidth=1, label='Rupture Tension ?')

# Set axis limits: x from 2 to 4 µm; y from 0 to 3 (tension in 10^-2 N/m)
plt.xlim(2, 4)
plt.ylim(0, 3)

plt.xlabel("Internal diameter of PG layer (µm)")
plt.ylabel("Tension in 10^-2 (N/m)")
plt.legend()
plt.grid()

plt.show()
