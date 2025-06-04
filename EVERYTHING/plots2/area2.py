"""
Another area plotting experiment with slightly different parameters. Helps
assess reproducibility of the wobble algorithm.
"""

import numpy as np
import matplotlib.pyplot as plt

def add_wobble(x, y, amplitude=0.2, num_points=200):
    """
    Create a wobbly version of a piecewise-linear curve.

    Parameters
    ----------
    x : array-like
        The x-coordinates of the control points.
    y : array-like
        The y-coordinates of the control points.
    amplitude : float, optional
        The standard deviation of the random noise, by default 0.2.
    num_points : int, optional
        How many total points to generate for the curve, by default 200.

    Returns
    -------
    x_wobble : np.ndarray
        New x-coordinates, evenly interpolated.
    y_wobble : np.ndarray
        Corresponding y-coordinates, with random "wobble."
    """
    # Parameterize t in [0,1] for interpolation
    t = np.linspace(0, 1, num_points)

    # Interpolate original control points onto this finer grid
    x_wobble = np.interp(t, np.linspace(0, 1, len(x)), x)
    y_wobble = np.interp(t, np.linspace(0, 1, len(y)), y)

    # Generate noise, force endpoints to 0 so we don't disturb endpoints
    noise = np.random.normal(loc=0.0, scale=amplitude, size=num_points)
    noise[0]  = 0
    noise[-1] = 0

    # Smooth the noise with a moving average for gentler wiggles
    window_size = 7
    window = np.ones(window_size) / window_size
    noise = np.convolve(noise, window, mode='same')

    # Add the noise to the y-values
    y_wobble += noise

    return x_wobble, y_wobble

# -------------------------
# USER PARAMETERS
# -------------------------

# ----------- PM line -----------
# Coordinates (start, mid, final)
pm_start_x = 15.0    # PM start X
pm_start_y = 0.0     # PM start Y
pm_mid_x   = 25.0    # where PM changes slope
pm_final_y = 4.0     # PM extends until Y=4

# Angles (in degrees from horizontal)
angle_pm1 = 0.0  # slope from (pm_start_x, pm_start_y) to (pm_mid_x, ???)
angle_pm2 = 20.0 # slope from (pm_mid_x, ???) until PM hits y=pm_final_y

# ----------- PG line -----------
# Coordinates (start, forced midpoint, final)
pg_start_x = 15.0
pg_start_y = 0.0
pg_mid_x   = 25.0  # forced midpoint X
pg_mid_y   = 3.0   # forced midpoint Y
pg_end_x   = 30.0  # final X for PG line

# Angles (in degrees from horizontal)
# We'll let the second segment of PG (x=20 to x=30) have its own angle:
angle_pg2 = 15.0

# ----------- Plot & Wobble -----------
# Wobble amplitude for both lines
wobble_amplitude = 0.05

# The horizontal "rupture tension" line
rupture_tension_value = 1.5

# X/Y axis display range
x_min_plot = 14
x_max_plot = 40
y_min_plot = 0
y_max_plot = 4

# -------------------------
# BUILD THE PM GEOMETRY
# -------------------------
#
# PM is piecewise:
#   1) from (pm_start_x, pm_start_y) to (pm_mid_x, pm_y1) 
#      with slope = tan(angle_pm1)
#   2) from (pm_mid_x, pm_y1) to final Y=pm_final_y 
#      with slope = tan(angle_pm2)
#

# Segment 1: compute the Y at pm_mid_x using angle_pm1
# angle_pm1 is measured from the horizontal => slope = tan(angle_pm1)
pm_y1 = pm_start_y + (pm_mid_x - pm_start_x)*np.tan(np.radians(angle_pm1))

# Segment 2: solve for the final X (pm_x2) where Y=pm_final_y
# slope = tan(angle_pm2) => (pm_final_y - pm_y1) = slope * (pm_x2 - pm_mid_x)
# => pm_x2 = pm_mid_x + (pm_final_y - pm_y1) / tan(angle_pm2)
pm_x2 = pm_mid_x + (pm_final_y - pm_y1)/np.tan(np.radians(angle_pm2))
pm_y2 = pm_final_y

# Gather PM control points
pm_x = [pm_start_x, pm_mid_x, pm_x2]
pm_y = [pm_start_y, pm_y1,    pm_y2]

# -------------------------
# BUILD THE PG GEOMETRY
# -------------------------
#
# PG is piecewise:
#   1) from (pg_start_x, pg_start_y) to (pg_mid_x, pg_mid_y) -- forced
#   2) from (pg_mid_x, pg_mid_y) to (pg_end_x, pg_y2)
#      with slope = tan(angle_pg2)

# For the second segment:
# slope = tan(angle_pg2), so (pg_y2 - pg_mid_y) = slope*(pg_end_x - pg_mid_x)
pg_y2 = pg_mid_y + (pg_end_x - pg_mid_x)*np.tan(np.radians(angle_pg2))

pg_x = [pg_start_x, pg_mid_x, pg_end_x]
pg_y = [pg_start_y, pg_mid_y, pg_y2]

# -------------------------
# WOBBLIFY (ADD NOISE)
# -------------------------
pm_x_w, pm_y_w = add_wobble(pm_x, pm_y, amplitude=wobble_amplitude, num_points=300)
pg_x_w, pg_y_w = add_wobble(pg_x, pg_y, amplitude=wobble_amplitude, num_points=300)

# -------------------------
# SUM (TOTAL) CURVE
# -------------------------
# We only sum over the region where PM and PG overlap.
# PG ends at x=pg_end_x; PM might extend beyond that or not.
# So the overlap is x in [15, min(pm_x2, pg_end_x)].
overlap_end = min(pm_x2, pg_end_x)
x_for_sum = np.linspace(pm_start_x, overlap_end, 300)

# Interpolate each wobbly line onto x_for_sum
pm_y_for_sum = np.interp(x_for_sum, pm_x_w, pm_y_w)
pg_y_for_sum = np.interp(x_for_sum, pg_x_w, pg_y_w)

# The total is the pointwise sum
total_y = pm_y_for_sum + pg_y_for_sum

# -------------------------
# PLOTTING
# -------------------------
plt.figure(figsize=(6, 5))

# Plot PM (blue)
plt.plot(pm_x_w, pm_y_w, label='T_PM- thin shell', linestyle='-', linewidth=2)

# Plot PG (green)
plt.plot(pg_x_w, pg_y_w, label='T_PG (t = 6 nm)', linestyle='-', linewidth=2)

# Plot total (red) on the overlap region
plt.plot(x_for_sum, total_y, label='Total', linestyle='--', linewidth=2, color='red')

# Rupture tension line at y=1.5
plt.axhline(y=rupture_tension_value, color='black', linestyle='--', linewidth=1, 
            label='Rupture Tension')

# Axis limits
plt.xlim(x_min_plot, x_max_plot)
plt.ylim(y_min_plot, y_max_plot)

# Labels
plt.xlabel("Inner Surface area of PG layer (µm²)")
plt.ylabel("Tension in 10^-2 (N/m)")

plt.legend()
plt.grid(True)

# If you wish, save to EPS:
# plt.savefig('myplot.eps', format='eps')

plt.show()
