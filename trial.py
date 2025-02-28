import numpy as np
import matplotlib.pyplot as plt

# Define plot range
plt.figure(figsize=(5, 5))
plt.xlim(0, 20)
plt.ylim(0, 20)

# Define PM curve
x_pm = np.array([0, 5, 20])
y_pm = np.array([0, 0.5, 0.5 + (20 - 5) * np.tan(np.radians(55))])

# Define middle straight line
x_mid = np.array([0, 20])
y_mid = np.array([0, (y_pm[-1] + 10 + (20 - 5) * np.tan(np.radians(55))) / 2])

# Define PG curve as the sum of PM and middle lines
x_pg = np.array([0, 5, 20])
y_pg = y_pm + np.interp(x_pg, x_mid, y_mid)

# Plot PM
plt.plot(x_pm, y_pm, label='Plasma Membrane (PM)', linestyle='solid', linewidth=2)

# Plot PG
plt.plot(x_pg, y_pg, label='Peptidoglycan (PG)', linestyle='dashed', linewidth=2, color='red')

# Plot middle straight line
plt.plot(x_mid, y_mid, label='Middle Line', linestyle='dashdot', linewidth=2, color='blue')

# Add grid, x and y axes
plt.grid(True, linestyle='--', linewidth=0.5)
plt.axhline(0, color='black', linewidth=1)
plt.axvline(0, color='black', linewidth=1)

# Show plot
plt.show()
