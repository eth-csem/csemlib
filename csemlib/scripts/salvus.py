import numpy as np
import os

from ..io.exodus_reader import ExodusReader
from ..models.ses3d import Ses3d
from ..background.grid_data import GridData
from ..models.s20rts import S20rts
from ..models.crust import Crust
from ..models import one_dimensional
from csemlib.models.specfem import Specfem

csemlib_directory, _ = os.path.split(os.path.split(__file__)[0])
model_dir = os.path.join(csemlib_directory, '..', 'regional_models')


def _evaluate_csem_salvus(x, y, z, regions_dict, regions, regions_2=None):

    """
    :param x: np array of x-coordinates in kilometers
    :param y: np array of y-coordinates in kilometers
    :param z: np array of y-coordinates in kilometers
    :param regions: specifies the region where the point(x,y,z) falls into in the one dimensional Earth model.
    :param regions_2: secondary region used for averaging over discontinuities
    :return: a grid_data object with the required information contained on it.
    """

    grid_data = GridData(x, y, z, solver='salvus')

    if regions_2 is not None:
        grid_data.add_one_d_salvus_continuous(region_min_eps=regions, region_plus_eps=regions_2)
    else:
        grid_data.add_one_d_salvus_discontinuous(regions)

    # Add s20rts
    if regions_dict['eval_s20']:
        s20 = S20rts()
        s20.eval_point_cloud_griddata(grid_data)

    # Models without crust that must be added before adding the crust. -------------------------------------------------

    # Add South Atlantic
    if regions_dict['eval_south_atlantic']:
        ses3d = Ses3d(os.path.join(model_dir, 'south_atlantic_2013'), grid_data.components)
        ses3d.eval_point_cloud_griddata(grid_data)

    # Add Australia
    if regions_dict['eval_australia']:
        ses3d = Ses3d(os.path.join(model_dir, 'australia_2010'), grid_data.components)
        ses3d.eval_point_cloud_griddata(grid_data)

    # Overwrite crustal values. ----------------------------------------------------------------------------------------
    # Add Crust
    if regions_dict['eval_crust']:
        cst = Crust()
        cst.eval_point_cloud_grid_data(grid_data)

    # Add 3D models with crustal component. ----------------------------------------------------------------------------

    # Add Japan
    if regions_dict['eval_japan']:
        ses3d = Ses3d(os.path.join(model_dir, 'japan_2016'), grid_data.components)
        ses3d.eval_point_cloud_griddata(grid_data)

    # Add Europe
    if regions_dict['eval_europe']:
        ses3d = Ses3d(os.path.join(model_dir, 'europe_2013'), grid_data.components)
        ses3d.eval_point_cloud_griddata(grid_data)

    # Add Marmara
    if regions_dict['eval_marmara_2017']:
        ses3d = Ses3d(os.path.join(model_dir, 'marmara_2017'), grid_data.components)
        ses3d.eval_point_cloud_griddata(grid_data)

    # Add South-East Asia
    if regions_dict['eval_south_east_asia_2017']:
        ses3d = Ses3d(os.path.join(model_dir, 'south_east_asia_2017'), grid_data.components)
        ses3d.eval_point_cloud_griddata(grid_data)

    # Add Iberia 2015
    if regions_dict['eval_iberia_2015']:
        ses3d = Ses3d(os.path.join(model_dir, 'iberia_2015'), grid_data.components)
        ses3d.eval_point_cloud_griddata(grid_data)

    # Add Iberia 2017
    if regions_dict['eval_iberia_2017']:
        ses3d = Ses3d(os.path.join(model_dir, 'iberia_2017'), grid_data.components)
        ses3d.eval_point_cloud_griddata(grid_data)

    # Add North Atlantic 2013
    if regions_dict['eval_north_atlantic_2013']:
        ses3d = Ses3d(os.path.join(model_dir, 'north_atlantic_2013'), grid_data.components)
        ses3d.eval_point_cloud_griddata(grid_data)

    # Add North America 2017
    if regions_dict['eval_north_america']:
        ses3d = Ses3d(os.path.join(model_dir, 'north_america_2017'), grid_data.components)
        ses3d.eval_point_cloud_griddata(grid_data)

    # Add Mike's global update
    if regions_dict['eval_global_1']:
        mikes_update = Specfem(interp_method="trilinear_interpolation")
        mikes_update.eval_point_cloud_griddata(grid_data)

    return grid_data


def add_csem_to_continuous_exodus(filename, regions_dict, with_topography=False):
    """
    Adds CSEM to a continuous Salvus mesh
    :param filename: salvus mesh file
    :param kwargs:
    :return:
    """
    # Default parameters
    salvus_mesh = ExodusReader(filename, mode='a')

    # 2D case
    if salvus_mesh.ndim == 2:
        x, y = salvus_mesh.points.T / 1000.0
        z = np.zeros_like(x)
    # 3D case
    elif salvus_mesh.ndim == 3:
        x, y, z = salvus_mesh.points.T / 1000.0
    else:
        raise ValueError('Incorrect amount of dimensions in Salvus mesh file')

    # compute radius
    rad = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    if with_topography:
        # get 1D_radius (normalized from 0 to 1)
        print("Accounting for topography")
        rad_1D = salvus_mesh.get_nodal_field("radius_1D")
        r_earth = 6371.0
        x[rad > 0] = x[rad > 0] * r_earth * rad_1D[rad > 0] / rad[rad > 0]
        y[rad > 0] = y[rad > 0] * r_earth * rad_1D[rad > 0] / rad[rad > 0]
        z[rad > 0] = z[rad > 0] * r_earth * rad_1D[rad > 0] / rad[rad > 0]
        rad = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    # get regions +/- eps to average velocities on the discontinuities
    epsilon = 0.5
    rad_plus_eps = rad + epsilon
    rad_minus_eps = rad - epsilon
    regions_plus_eps = one_dimensional.get_region(rad_plus_eps)
    regions_min_eps = one_dimensional.get_region(rad_minus_eps)

    grid_data = _evaluate_csem_salvus(x, y, z, regions_dict, regions=regions_plus_eps, regions_2=regions_min_eps)
    dimensionless_components = ["eta", "QKappa", "Qmu"]
    for component in grid_data.components:
        if component in dimensionless_components:
            vals = grid_data.get_component(component)
        else:
            vals = grid_data.get_component(component) * 1000.0
        salvus_mesh.attach_field('%s' % component.upper(), vals)
    salvus_mesh.close()

def add_s20_to_isotropic_exodus(filename, regions_dict, with_topography=False):
    """
    Adds s2- to a continuous Salvus mesh
    :param filename: salvus mesh file
    :param kwargs:
    :return:
    """
    # Default parameters
    salvus_mesh = ExodusReader(filename, mode='a')

    # 2D case
    if salvus_mesh.ndim == 2:
        x, y = salvus_mesh.points.T / 1000.0
        z = np.zeros_like(x)
    # 3D case
    elif salvus_mesh.ndim == 3:
        x, y, z = salvus_mesh.points.T / 1000.0
    else:
        raise ValueError('Incorrect amount of dimensions in Salvus mesh file')

    vp = salvus_mesh.get_nodal_field("VP")
    vs = salvus_mesh.get_nodal_field("VS")

    grid_data = GridData(x,y,z)
    grid_data.set_component("vp", vp)
    grid_data.set_component("vs", vs)

    s20 = S20rts()
    s20.eval_point_cloud_griddata(grid_data)

    for component in grid_data.components:

        vals = grid_data.get_component(component)
        salvus_mesh.attach_field('%s' % component.upper(), vals)

    salvus_mesh.close()


def add_csem_to_discontinuous_exodus(filename, regions_dict):
    """
    Adds CSEM to a discontinuous Salvus mesh
    :param filename: salvus mesh file
    :param kwargs:
    :return:
    """

    salvus_mesh = ExodusReader(filename, mode='a')

    # Get element centroids
    if salvus_mesh.ndim == 2:
        x_c, y_c = salvus_mesh.get_element_centroid().T
        z_c = np.zeros_like(x_c)
    elif salvus_mesh.ndim == 3:
        x_c, y_c, z_c = salvus_mesh.get_element_centroid().T
    else:
        raise ValueError('Incorrect amount of dimensions in Salvus mesh file')

    # Get element centroid in km
    rad_c = np.sqrt(x_c ** 2 + y_c ** 2 + z_c ** 2) / 1000.0
    # Get region corresponding to element centroid
    regions = one_dimensional.get_region(rad_c)

    for i in np.arange(salvus_mesh.nodes_per_element):
        print('Adding CSEM to node {} out of {}'.format(i+1, salvus_mesh.nodes_per_element))
        if salvus_mesh.ndim == 2:
            x, y = salvus_mesh.points[salvus_mesh.connectivity[:, i]].T / 1000.0
            z = np.zeros_like(x)
        else:
            x, y, z = salvus_mesh.points[salvus_mesh.connectivity[:, i]].T / 1000.0

        grid_data = _evaluate_csem_salvus(x, y, z, regions_dict, regions)
        dimensionless_components = ["eta", "QKappa", "Qmu"]
        for component in grid_data.components:
            # Convert to m/s
            if component in dimensionless_components:
                vals = grid_data.get_component(component)
            else:
                vals = grid_data.get_component(component) * 1000
            salvus_mesh.attach_field('%s_%d' % (component.upper(), i), vals)

    # Attach fluid field
    salvus_mesh.attach_field('fluid', np.array(regions == 1, dtype=int))
    salvus_mesh.close()



