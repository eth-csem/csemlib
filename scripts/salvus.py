import numpy as np
import os

from csemlib.background.grid_data import GridData
from csemlib.io.exodus_reader import ExodusReader
from csemlib.models import one_dimensional
from csemlib.models.s20rts import S20rts
from scripts.evaluate_csem import evaluate_csem


def add_csem_to_continuous_exodus(filename,
                                  with_topography=False):
    """
    Adds CSEM to a continuous Salvus mesh
    :param with_topography: If mesh contains topography and/or ellipticity
    and it contains the parameter radius_1D, setting this parameter to True
    will ensure correct interpolation.
    :param filename: salvus mesh file
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

    grid_data = evaluate_csem(x, y, z, regions=regions_plus_eps,
                              regions_2=regions_min_eps)
    dimensionless_components = ["eta", "QKappa", "Qmu"]

    for component in grid_data.components:
        print(component)
        if component in dimensionless_components:
            vals = grid_data.get_component(component)
        else:
            vals = grid_data.get_component(component) * 1000.0
        salvus_mesh.attach_field('%s' % component.upper(), vals)
    salvus_mesh.close()


def add_s20_to_isotropic_exodus(filename):
    """
    Adds s2- to a continuous Salvus mesh
    :param filename: salvus mesh file
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

    grid_data = GridData(x, y, z)
    grid_data.set_component("vp", vp)
    grid_data.set_component("vs", vs)

    s20 = S20rts()
    s20.eval_point_cloud_griddata(grid_data)

    for component in grid_data.components:
        vals = grid_data.get_component(component)
        salvus_mesh.attach_field('%s' % component.upper(), vals)

    salvus_mesh.close()


def add_csem_to_salvusv2(filename):
    """
    This currently still assumes TTI and might not deal correctly with the CMB
    and topography.

    :param filename:
    """
    import h5py

    gll = h5py.File(filename, 'r+')
    gll_coords = gll["MODEL/coordinates"]
    nelem = gll_coords.shape[0]
    ngll_pelem = gll_coords.shape[1]
    ndim = 3

    # reshape to list of coordinates
    gll_coords = np.reshape(gll_coords, (ngll_pelem * nelem, ndim))
    x, y, z = gll_coords.T / 1000.0

    # compute radius
    rad = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    # get regions +/- eps to average velocities on the discontinuities
    epsilon = 0.5
    rad_plus_eps = rad + epsilon
    rad_minus_eps = rad - epsilon
    regions_plus_eps = one_dimensional.get_region(rad_plus_eps)
    regions_min_eps = one_dimensional.get_region(rad_minus_eps)

    grid_data = evaluate_csem(x, y, z, regions=regions_plus_eps,
                              regions_2=regions_min_eps)
    dimensionless_components = ["Eta", "Qkappa", "Qmu"]

    data = gll["MODEL"]["data"]
    params_str = data.attrs["DIMENSION_LABELS"][1].decode()

    # data shape (nelem, ndata, ngll)
    param_list = [x.strip(' ') for x in
                  params_str.lstrip("[").rstrip("]").split("|")]
    data_vals = data.value
    for idx, component in enumerate(param_list):
        component = component.lower()
        if component == "qmu":
            component = "Qmu"
        elif component == "qkappa":
            component = "QKappa"
        if component in dimensionless_components:
            vals = grid_data.get_component(component)
        else:
            vals = grid_data.get_component(component) * 1000.0

        # reshape to (nelem, ngll_pelem)
        vals_reshaped = np.reshape(vals, (nelem, ngll_pelem))
        data_vals[:, idx, :] = vals_reshaped

    data.write_direct(data_vals)


def add_csem_to_discontinuous_exodus(filename):
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
        print('Adding CSEM to node {} out of {}'.format(i + 1,
                                                        salvus_mesh.nodes_per_element))
        if salvus_mesh.ndim == 2:
            x, y = salvus_mesh.points[
                       salvus_mesh.connectivity[:, i]].T / 1000.0
            z = np.zeros_like(x)
        else:
            x, y, z = salvus_mesh.points[
                          salvus_mesh.connectivity[:, i]].T / 1000.0

        grid_data = evaluate_csem(x, y, z, regions)
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

def put_csem(mesh):

    """
    Adds CSEM to a mesh object
    :param mesh: salvus mesh object
    :param kwargs:
    :return:
    """

    nodes_per_element = 27
    # convert x,y,z to spherical shhape


    for i in np.arange(nodes_per_element):
        print('Adding CSEM to node {} out of {}'.format(i + 1,nodes_per_element))

        x, y, z = mesh.points[mesh.connectivity[:, i]].T / 1000.0
        rad = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        # get 1D_radius (normalized from 0 to 1)
        print("Accounting for topography")
        rad_1D = mesh.element_nodal_fields["z_node_1D"][:, i]
        r_earth = 6371.0
        x[rad > 0] = x[rad > 0] * r_earth * rad_1D[rad > 0] / rad[rad > 0]
        y[rad > 0] = y[rad > 0] * r_earth * rad_1D[rad > 0] / rad[rad > 0]
        z[rad > 0] = z[rad > 0] * r_earth * rad_1D[rad > 0] / rad[rad > 0]
        rad = np.sqrt(x ** 2 + y ** 2 + z ** 2)

        epsilon = 0.5
        rad_plus_eps = rad + epsilon
        rad_minus_eps = rad - epsilon
        regions_plus_eps = one_dimensional.get_region(rad_plus_eps)
        regions_min_eps = one_dimensional.get_region(rad_minus_eps)

        grid_data = evaluate_csem(x, y, z, regions=regions_plus_eps,
                                  regions_2=regions_min_eps)
        dimensionless_components = ["eta", "QKappa", "Qmu"]
        for component in grid_data.components:
            # Convert to m/s
            if component in dimensionless_components:
                vals = grid_data.get_component(component)
            else:
                vals = grid_data.get_component(component) * 1000
            mesh.element_nodal_fields['%s' % (component.upper())][:, i] = vals

    return mesh


def put_csem_nodal(mesh):
    """
    Adds CSEM to a mesh object
    :param mesh: salvus mesh object
    :param kwargs:
    :return:
    """

    nodes_per_element = 27
    # convert x,y,z to spherical shhape

    x, y, z = mesh.points.T / 1000.0
    rad = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    # get 1D_radius (normalized from 0 to 1)
    print("Accounting for topography")

    _, i = np.unique(mesh.connectivity, return_index=True)
    rad_1D = mesh.element_nodal_fields["z_node_1D"].flatten()[i]
    r_earth = 6371.0
    x[rad > 0] = x[rad > 0] * r_earth * rad_1D[rad > 0] / rad[rad > 0]
    y[rad > 0] = y[rad > 0] * r_earth * rad_1D[rad > 0] / rad[rad > 0]
    z[rad > 0] = z[rad > 0] * r_earth * rad_1D[rad > 0] / rad[rad > 0]
    rad = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    epsilon = 0.5
    rad_plus_eps = rad + epsilon
    rad_minus_eps = rad - epsilon
    regions_plus_eps = one_dimensional.get_region(rad_plus_eps)
    regions_min_eps = one_dimensional.get_region(rad_minus_eps)

    grid_data = evaluate_csem(x, y, z, regions=regions_plus_eps,
                              regions_2=regions_min_eps)
    dimensionless_components = ["eta", "QKappa", "Qmu"]
    for component in grid_data.components:
        # Convert to m/s
        if component in dimensionless_components:
            vals = grid_data.get_component(component)
        else:
            vals = grid_data.get_component(component) * 1000

        mesh.element_nodal_fields['%s' % (component.upper())][:] = \
            vals[mesh.connectivity]

    return mesh
