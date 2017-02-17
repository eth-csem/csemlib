import os
import numpy as np
from csemlib.background.fibonacci_grid import FibonacciGrid
from csemlib.background.grid_data import GridData
from csemlib.models.model import triangulate, write_vtk
from csemlib.models.topography import Topography
VTK_DIR = os.path.join(os.path.split(__file__)[0], 'vtk')
DECIMAL_CLOSE = 3


def test_topography():
    """
    Regression test that checks whether the mean topography is still the same as at time of creation,
     also generates a vtk file which shows topography.
    :return:
    """

    # Generate Grid
    rad = 6371.0
    fib_grid = FibonacciGrid()
    radii = np.array([rad])
    resolution = np.ones_like(radii) * 500
    fib_grid.set_global_sphere(radii, resolution)
    grid_data = GridData(*fib_grid.get_coordinates())

    top = Topography()
    top.read()
    topo = top.eval(grid_data.df['c'], grid_data.df['l'], topo_smooth_factor=0.)

    x, y, z = grid_data.get_coordinates().T
    elements = triangulate(x, y, z)
    coords = np.array((x, y, z)).T

    # Test if topography does not change
    mean_topo_test = np.array([-2.46202810922])
    mean_topo = np.mean(topo)
    np.testing.assert_almost_equal(mean_topo, mean_topo_test, decimal=DECIMAL_CLOSE)

    write_vtk(os.path.join(VTK_DIR, 'topography.vtk'), coords, elements, topo, 'topography')
