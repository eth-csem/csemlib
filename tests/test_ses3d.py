import os

import numpy as np
import csemlib.background.skeleton as skl

import csemlib.models.ses3d as s3d
from csemlib.background.grid_data import GridData
from csemlib.background.fibonacci_grid import FibonacciGrid
from csemlib.models.crust import Crust
from csemlib.models.model import triangulate, write_vtk
from csemlib.models.one_dimensional import prem_eval_point_cloud
from csemlib.models.s20rts import S20rts
from csemlib.models.ses3d_rbf import Ses3d_rbf
from csemlib.utils import cart2sph

TEST_DATA_DIR = os.path.join(os.path.split(__file__)[0], 'test_data')
VTK_DIR = os.path.join(os.path.split(__file__)[0], 'vtk')
DECIMAL_CLOSE = 3


def test_ses3d():

    #  Generate visualisation grid
    fib_grid = FibonacciGrid()
    # Set global background grid
    radii = np.linspace(6271.0, 6271.0, 1)
    resolution = np.ones_like(radii) * (6371.0 / 15)
    fib_grid.set_global_sphere(radii, resolution)
    # refinement region coarse
    c_min = np.radians(35)
    c_max = np.radians(65)
    l_min = np.radians(125)
    l_max = np.radians(155)
    radii_regional = np.linspace(6271.0, 6271.0, 1)
    resolution_regional = np.ones_like(radii_regional) * 50
    fib_grid.add_refinement_region(c_min, c_max, l_min, l_max, radii_regional, resolution_regional)


    # Setup GridData
    grid_data = GridData(*fib_grid.get_coordinates())

    # Evaluate Prem
    rho, vpv, vsv, vsh = prem_eval_point_cloud(grid_data.df['r'])
    grid_data.set_component('vsv', np.ones(len(grid_data)))

    mod = Ses3d_rbf('japan', os.path.join(TEST_DATA_DIR, 'japan'),
                    components=grid_data.components, interp_method='nearest_neighbour')
    mod.eval_point_cloud_griddata(grid_data)

    # Write vtk
    x, y, z = grid_data.get_coordinates(coordinate_type='cartesian').T
    elements = triangulate(x, y, z)
    pts = np.array((x, y, z)).T
    write_vtk("ses3d_nearest_neighbour.vtk", pts, elements, grid_data.get_component('vsv'), 'ses3dvsv')


def test_ses3d_multi_region_read():
    """
    Test to ensure that a multi-region ses3d file gets read properly.
    We read in a dummy file with 3 regions and vsv defined in each regions.
    Check to make sure values from each region make sense.
    """

    mod = s3d.Ses3d('MultiRegion', os.path.join(os.path.join(TEST_DATA_DIR,
        'multi_region_ses3d')), components=['vsv'])
    mod.read()

    region = 0
    np.testing.assert_allclose(mod.data(region)['col'], [0.8735, 0.8753], rtol=1e-4,
            atol=0.0)
    np.testing.assert_allclose(mod.data(region)['lon'], [0.8735, 0.8753], rtol=1e-4,
            atol=0.0)
    np.testing.assert_allclose(mod.data(region)['rad'], [6.15e3, 6.25e3], rtol=1e-4,
            atol=0.0)

    region = 1
    np.testing.assert_allclose(mod.data(region)['col'], [0.8748], rtol=1e-4,
            atol=0.0)
    np.testing.assert_allclose(mod.data(region)['lon'], [0.8748], rtol=1e-4,
            atol=0.0)
    np.testing.assert_allclose(mod.data(region)['rad'], [6.05e3], rtol=1e-4,
            atol=0.0)
    
    region = 2
    np.testing.assert_allclose(mod.data(region)['col'], [0.8770, 0.8857, 0.8945], rtol=1e-4,
            atol=0.0)
    np.testing.assert_allclose(mod.data(region)['lon'], [0.8770, 0.8857, 0.8945], rtol=1e-4,
            atol=0.0)
    np.testing.assert_allclose(mod.data(region)['rad'], [5.5e3, 5.7e3, 5.9e3],
            rtol=1e-4, atol=0.0)


def test_ses3d():
    """
    Test to ensure that a ses3d model returns itself.
    """

    mod = s3d.Ses3d('japan', os.path.join(TEST_DATA_DIR, 'japan'),
                    components=['rho', 'vsv', 'vsh', 'vp'])
    mod.read()

    all_cols, all_lons, all_rads = np.meshgrid(
        mod.data.coords['col'].values,
        mod.data.coords['lon'].values,
        mod.data.coords['rad'].values)
    interp = mod.eval(mod.data['x'].values.ravel(), mod.data['y'].values.ravel(),
                      mod.data['z'].values.ravel(), param=['vsv', 'rho', 'vsh', 'vp'])
    # Setup true data.
    true = np.empty((len(all_cols.ravel()), 4))
    true[:, 0] = mod.data['vsv'].values.ravel()
    true[:, 1] = mod.data['rho'].values.ravel()
    true[:, 2] = mod.data['vsh'].values.ravel()
    true[:, 3] = mod.data['vp'].values.ravel()

    np.testing.assert_almost_equal(true, interp, decimal=DECIMAL_CLOSE)


