import numpy as np
import matplotlib.pyplot as plt

import os
if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    import matplotlib
    matplotlib.use('Agg')  # set the backend before importing pyplot

# Define the data points
x = np.array([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2])
z = np.array([-0.5, -0.4, -0.3, -0.2, 0, 0.5, 1, 1.2, 1.5])
# Define the orientation (for demonstration, I'll use an arbitrary angle array)
angles = np.linspace(0, 360, len(x))  # Assuming the orientation varies
# Create the figure and axis
fig, ax = plt.subplots()
# Plot each triangle at the points (x, z) with the correct orientation
for i in range(len(x)):
   triangle = plt.scatter(x[i], z[i], marker=(3, 0, angles[i]), color='r', s=400)

# Highlight the central triangle (optional, for reference)
ax.scatter(0, 0, marker=(3, 0, 0), color='g', s=800)  # Larger green triangle at origin
# Set axis labels
ax.set_xlabel('X [m]')
ax.set_ylabel('Z [m]')
ax.set_title('Matterport3d-s00015-39 frames')
# Adjust axis limits for better visualization
ax.set_xlim([-3, 3])
ax.set_ylim([-1.5, 2])
# Show the plot
plt.grid(True)
plt.savefig('/Titan/code/robohike_ws/src/pycpptools/src/python/utils_dataset/map_free_reloc/test.pdf')