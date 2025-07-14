import numpy as np
import pyvista as pv

def setup_plotter(plotter):
    """Sets up the initial plotter configuration, including background, axes, and grid."""
    plotter.clear()
    plotter.add_axes()
    plotter.set_background(color='grey')

    # Add grid floor similar to Blender
    x = np.arange(-192, 193, 1)  # Ensure inclusive of boundaries
    y = np.arange(-108, 109, 1)
    x, y = np.meshgrid(x, y)
    z = np.zeros_like(x)
    grid = pv.StructuredGrid(x, y, z)
    plotter.add_mesh(grid, color='black', style='wireframe', line_width=0.5)

    # Create and add x-axis line
    x_axis = pv.Line([-192, 0, 0], [192, 0, 0])
    plotter.add_mesh(x_axis, color='red', line_width=5)

    # Create and add y-axis line
    y_axis = pv.Line([0, -108, 0], [0, 108, 0])
    plotter.add_mesh(y_axis, color='green', line_width=5)
