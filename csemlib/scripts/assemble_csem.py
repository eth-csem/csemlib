import numpy as np

from csemlib.background.fibonacci_grid import FibonacciGrid
from csemlib.background.grid_data import GridData
from csemlib.models.crust import Crust
from csemlib.models.model import write_vtk, triangulate
from csemlib.models.s20rts import S20rts
from csemlib.models.ses3d import Ses3d
from csemlib.io.readers import read_from_grid

import os

csemlib_directory, _ = os.path.split(os.path.split(__file__)[0])
model_directory = os.path.join(csemlib_directory, '..', 'regional_models')


def assemble_csem(grid_data, **kwargs):
    # Default parameters
    params = dict(eval_crust=True, eval_s20=True, eval_japan=True,
                  eval_south_atlantic=True, eval_australia=True, eval_europe=True, eval_marmara=False)
    params.update(kwargs)

    # Initialise with CSEM 1D background model.
    grid_data.add_one_d()

    # Add s20rts
    if params['eval_s20']:
        s20 = S20rts()
        s20.eval_point_cloud_griddata(grid_data)

    # Models without crust that must be added before adding the crust. -------------------------------------------------

    # Add South Atlantic
    if params['eval_south_atlantic']:
        ses3d = Ses3d(os.path.join(model_directory, 'south_atlantic_2013'), grid_data.components)
        ses3d.eval_point_cloud_griddata(grid_data)

    # Add Australia
    if params['eval_australia']:
        ses3d = Ses3d(os.path.join(model_directory, 'australia_2010'), grid_data.components)
        ses3d.eval_point_cloud_griddata(grid_data)

    # Overwrite crustal values. ----------------------------------------------------------------------------------------

    # Add Crust
    if params['eval_crust']:
        cst = Crust()
        cst.eval_point_cloud_grid_data(grid_data)

    # Add 3D models with crustal component. ----------------------------------------------------------------------------

    # Add Japan
    if params['eval_japan']:
        ses3d = Ses3d(os.path.join(model_directory, 'japan_2016'), grid_data.components)
        ses3d.eval_point_cloud_griddata(grid_data)

    # Add Europe
    if params['eval_europe']:
        ses3d = Ses3d(os.path.join(model_directory, 'europe_2013'), grid_data.components)
        ses3d.eval_point_cloud_griddata(grid_data)

    # Add Marmara
    if params['eval_marmara']:
        ses3d = Ses3d(os.path.join(model_directory, 'marmara_2017'), grid_data.components)
        ses3d.eval_point_cloud_griddata(grid_data)

    return grid_data
    # Generate output. -------------------------------------------------------------------------------------------------


def depth_slice_to_vtk(depth, resolution, filename=None):

    # Initialize grid points
    fib_grid = FibonacciGrid()
    fib_grid.set_global_sphere(radii=[6371.0-depth], resolution=[resolution])
    grid_data = GridData(*fib_grid.get_coordinates())

    grid_data = assemble_csem(grid_data)

    x, y, z = grid_data.get_coordinates().T
    # Make vtk file.
    elements = triangulate(x, y, z)
    points = np.array((x, y, z)).T

    if filename is None:
        name = 'depth_' + str(depth) + "_res_" + str(resolution) + ".vtk"
        filename = os.path.join(".", name)
    print('Writing vtk to {}'.format(filename))
    write_vtk(filename, points, elements, grid_data.df['vsv'])
