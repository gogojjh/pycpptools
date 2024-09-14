import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.transforms as transforms

import os
if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    import matplotlib
    matplotlib.use('Agg')  # set the backend before importing pyplot

# Function to create a long triangle and rotate it by a given angle
def create_triangle(x, z, angle, size=0.5):
    triangle = np.array([[0, -size], [-size/4, size], [size/4, size]])  # Longer triangle shape
    # Rotate the triangle by 'angle'
    transform = transforms.Affine2D().rotate_deg(angle) + plt.gca().transData
    polygon = Polygon(triangle + [x, z], closed=True, color='r', transform=transform)
    plt.gca().add_patch(polygon)

# Define the data points
x = np.array([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2])
z = np.array([-0.5, -0.4, -0.3, -0.2, 0, 0.5, 1, 1.2, 1.5])

# Define the orientation (for demonstration, I'll use an arbitrary angle array)
angles = np.linspace(0, 360, len(x))  # Assuming the orientation varies

# Create the figure and axis
fig, ax = plt.subplots()

# Plot each long triangle at the points (x, z) with the correct orientation
for i in range(len(x)):
    create_triangle(x[i], z[i], angles[i])

# Highlight the central triangle (optional, for reference)
create_triangle(0, 0, 0, size=0.2)  # Larger green triangle at the origin

# Set axis labels
ax.set_xlabel('X [m]')
ax.set_ylabel('Z [m]')
ax.set_title('Matterport3d-s00015-39 frames')

# Adjust axis limits for better visualization
ax.set_xlim([-3, 3])
ax.set_ylim([-1.5, 2])

# Show the plot
plt.grid(True)
plt.savefig('/Titan/code/robohike_ws/src/pycpptools/src/python/utils_dataset/map_free_reloc/test_long_triangles.pdf')
# plt.show()