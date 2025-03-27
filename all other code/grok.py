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

# Define plot range
plt.figure(figsize=(5, 5))
plt.xlim(0, 20)
plt.ylim(0, 20)

# Define PM curve (piecewise linear)
x_pm = np.array([0, 5, 20])
y_pm = np.array([0, 0.5, 0.5 + (20 - 5) * np.tan(np.radians(55))])

# Define middle straight line
x_mid = np.array([0, 20])
y_mid = np.array([0, (y_pm[-1] + 10 + (20 - 5) * np.tan(np.radians(55))) / 2])

# Define PG curve as the sum of PM and middle lines
x_pg = np.array([0, 5, 20])
y_pg = y_pm + np.interp(x_pg, x_mid, y_mid)

# Generate wobbly versions of each line with adjusted amplitude for the middle line
x_pm_w, y_pm_w = add_wobble(x_pm, y_pm, amplitude=0.2)
x_mid_w, y_mid_w = add_wobble(x_mid, y_mid, amplitude=0.05)  # Less wobble for middle line
x_pg_w, y_pg_w = add_wobble(x_pg, y_pg, amplitude=0.2)

# Plot PM with wobble
plt.plot(x_pm_w, y_pm_w, label='T_PM', linestyle='solid', linewidth=2)
# Plot PG with wobble
plt.plot(x_pg_w, y_pg_w, label='Total', linestyle='dashed', linewidth=2, color='red')
# Plot middle line with less wobble
plt.plot(x_mid_w, y_mid_w, label='T_PG', linestyle='dashdot', linewidth=2, color='blue')

# Add grid and axes
plt.grid(True, linestyle='--', linewidth=0.5)
plt.axhline(0, color='black', linewidth=1)
plt.axvline(0, color='black', linewidth=1)
plt.legend()

# Show plot
plt.show()
