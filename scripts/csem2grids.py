"""
Read a list of Cartesian grid points, evaluate the CSEM, compute a Delauney
triangulation, and turn into a vtk file.
"""

import os
import numpy as np
from csemlib.csem.evaluate_csem import evaluate_csem
from csemlib.io.readers import read_from_grid
from csemlib.models.model import write_vtk, triangulate


depths = [100]

for depth in depths:

    # Read some grid points. ---------------------------------------------------
    x, y, z = read_from_grid('../../grids/OUTPUT/fib_'+str(depth)+'.dat')
    grid_data = evaluate_csem(x,y,z)

    # Generate output. ---------------------------------------------------------

    # Make vtk file.
    elements = triangulate(x, y, z)
    points = np.array((x, y, z)).T

    filename = os.path.join('./', str(depth)+'.vtk')
    write_vtk(filename, points, elements, grid_data.df['vsv'])
