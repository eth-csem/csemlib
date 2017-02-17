import os

import numpy as np
from csemlib.background.fibonacci_grid import FibonacciGrid
from csemlib.background.grid_data import GridData
from csemlib.models.crust import Crust
from csemlib.models.model import triangulate, write_vtk

VTK_DIR = os.path.join(os.path.split(__file__)[0], 'vtk')
DECIMAL_CLOSE = 3

def test_crust():
    """
    Regression test that checks whether the mean crustal depths and velocity is still the same as at time of creation,
     also generates vtk files
    :return:
    """

    # Generate Grid
    rad = 6351.0
    fib_grid = FibonacciGrid()
    radii = np.array(np.linspace(rad, rad, 1))
    resolution = np.ones_like(radii) * 500
    fib_grid.set_global_sphere(radii, resolution)
    grid_data = GridData(*fib_grid.get_coordinates())
    grid_data.add_one_d()

    crust = Crust()
    crust.read()
    crust_dep = crust.eval(grid_data.df['c'], grid_data.df['l'], param='crust_dep',
                           smooth_fac=crust.crust_dep_smooth_fac)
    crust_vs = crust.eval(grid_data.df['c'], grid_data.df['l'], param='crust_vs',
                          smooth_fac=crust.crust_dep_smooth_fac)

    #Test if mean crustal depth and velocity remain the same
    mean_crust_dep_test = np.array([18.9934922621])
    mean_crust_vs_test = np.array([3.42334914127])
    mean_crust_dep = np.mean(crust_dep)
    mean_crust_vs = np.mean(crust_vs)
    np.testing.assert_almost_equal(mean_crust_dep, mean_crust_dep_test, decimal=DECIMAL_CLOSE)
    np.testing.assert_almost_equal(mean_crust_vs, mean_crust_vs_test, decimal=DECIMAL_CLOSE)

    # Write vtk's
    x, y, z = grid_data.get_coordinates().T
    elements = triangulate(x, y, z)
    coords = np.array((x, y, z)).T
    write_vtk(os.path.join(VTK_DIR, 'crust_dep.vtk'), coords, elements, crust_dep, 'crust_dep')
    write_vtk(os.path.join(VTK_DIR, 'crust_vs.vtk'), coords, elements, crust_vs, 'crust_vs')
