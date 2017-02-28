import os

import numpy as np

import csemlib.models.ses3d as s3d
from csemlib.background.grid_data import GridData
from csemlib.background.fibonacci_grid import FibonacciGrid
from csemlib.models.ses3d_rbf import Ses3d_rbf

TEST_DATA_DIR = os.path.join(os.path.split(__file__)[0], 'test_data')
VTK_DIR = os.path.join(os.path.split(__file__)[0], 'vtk')
DECIMAL_CLOSE = 3


def test_ses3d_griddata():

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
    grid_data.add_one_d()
    # Evaluate Prem
    grid_data.set_component('vsv', np.ones(len(grid_data)))
    print(grid_data.components)
    mod = Ses3d_rbf(os.path.join(TEST_DATA_DIR, 'japan_test'),
                    components=grid_data.components, interp_method='nearest_neighbour')
    mod.eval_point_cloud_griddata(grid_data)
    mod.eval_point_cloud_griddata(grid_data, interp_method='griddata_linear')
    mod.eval_point_cloud_griddata(grid_data, interp_method='radial_basis_func')

    mod.model_info['component_type'] = 'perturbation'
    mod.eval_point_cloud_griddata(grid_data, interp_method='griddata_linear')
    mod.eval_point_cloud_griddata(grid_data, interp_method='radial_basis_func')


def test_ses3d_multi_region_read():
    """
    Test to ensure that a multi-region ses3d file gets read properly.
    We read in a dummy file with 3 regions and vsv defined in each regions.
    Check to make sure values from each region make sense.
    """

    mod = s3d.Ses3d(os.path.join(os.path.join(TEST_DATA_DIR, 'multi_region_ses3d')), components=['vsv'])
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


def test_hdf5_writer():
    """
    Write an hdf5 file
    :return:
    """
    mod = s3d.Ses3d(os.path.join(TEST_DATA_DIR, 'japan_test'), components=['rho', 'vpv', 'vph', 'vsh', 'vsv'])
    mod.read()

    filename = os.path.join(TEST_DATA_DIR, 'japan_test.hdf5')
    mod.write_to_hdf5(filename)
    os.remove(filename)

def test_rotation():

    #  Generate visualisation grid
    fib_grid = FibonacciGrid()
    # Set global background grid
    radii = np.linspace(6271.0, 6271.0, 1)
    resolution = np.ones_like(radii) * 300
    fib_grid.set_global_sphere(radii, resolution)

    grid_data = GridData(*fib_grid.get_coordinates())
    grid_data.add_one_d()
    grid_data.set_component('vsv', np.zeros(len(grid_data)))

    mod = s3d.Ses3d(os.path.join(os.path.join(TEST_DATA_DIR, 'test_region')), components=['vsv'])
    mod.eval_point_cloud_griddata(grid_data)

    mod = Ses3d_rbf(os.path.join(os.path.join(TEST_DATA_DIR, 'test_region')), components=['vsv'])
    mod.eval_point_cloud_griddata(grid_data, interp_method='radial_basis_func')
    mod.eval_point_cloud_griddata(grid_data, interp_method='griddata_linear')
    mod.eval_point_cloud_griddata(grid_data)
    grid_data.del_one_d()
