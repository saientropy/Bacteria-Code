import numpy as np
import matplotlib.pyplot as plt

def add_wobble(x, y, amplitude=0.2, num_points=200):
    """
    Given arrays x and y defining a line, return new x and y arrays with many intermediate
    points and a slight vertical noise added. The endpoints remain fixed.
    """
    t = np.linspace(0, 1, num_points)
    # Interpolate x and y values along a parameter t
    x_interp = np.interp(t, np.linspace(0, 1, len(x)), x)
    y_interp = np.interp(t, np.linspace(0, 1, len(y)), y)
    
    # Create random noise and force endpoints to zero so the overall line endpoints remain unchanged
    noise = np.random.normal(scale=amplitude, size=num_points)
    noise[0] = 0
    noise[-1] = 0
    
    # Smooth the noise using a simple moving average filter
    window_size = 5
    window = np.ones(window_size) / window_size
    noise = np.convolve(noise, window, mode='same')
    
    y_wobble = y_interp + noise
    return x_interp, y_wobble

# Original definitions (in original units)
x_pm = np.array([0, 5, 20])
y_pm = np.array([0, 0.5, 0.5 + (20 - 5) * np.tan(np.radians(55))])

x_mid = np.array([0, 20])
y_mid = np.array([0, (y_pm[-1] + 10 + (20 - 5) * np.tan(np.radians(55))) / 2])

x_pg = np.array([0, 5, 20])
y_pg = y_pm + np.interp(x_pg, x_mid, y_mid)

# Generate wobbly versions of each line with adjusted amplitude for the middle line
x_pm_w, y_pm_w = add_wobble(x_pm, y_pm, amplitude=0.2)
x_mid_w, y_mid_w = add_wobble(x_mid, y_mid, amplitude=0.05)  # Less wobble for middle line
x_pg_w, y_pg_w = add_wobble(x_pg, y_pg, amplitude=0.2)

# Scale the data to the new units:
# x: from 0-20 becomes 0-10 atm (multiply by 0.5)
# y: from 0-20 becomes 0-4 N/m² (multiply by 0.2)
x_scale = 0.5
y_scale = 0.2

x_pm_w = x_pm_w * x_scale
y_pm_w = y_pm_w * y_scale

x_mid_w = x_mid_w * x_scale
y_mid_w = y_mid_w * y_scale

x_pg_w = x_pg_w * x_scale
y_pg_w = y_pg_w * y_scale

# Create the plot with the new axis limits and labels
plt.figure(figsize=(5, 5))
plt.xlim(0, 10)
plt.ylim(0, 4)

# Plot the curves (visual appearance remains identical)
plt.plot(x_pm_w, y_pm_w, label='T_PM', linestyle='solid', linewidth=2)
plt.plot(x_pg_w, y_pg_w, label='Total', linestyle='solid', linewidth=2, color='red')
plt.plot(x_mid_w, y_mid_w, label='T_PG', linestyle='dashdot', linewidth=2, color='blue')

plt.grid(True, linestyle='--', linewidth=0.5)
plt.axhline(0, color='black', linewidth=1)
plt.axvline(0, color='black', linewidth=1)
plt.legend()

# Update axis labels
plt.xlabel("Pressure (atm)")
plt.ylabel("Tension 10^-2 (N/m²)")

plt.show()
