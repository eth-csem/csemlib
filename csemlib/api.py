import numpy as np

from csemlib.background.fibonacci_grid import FibonacciGrid
from csemlib.models.model import write_vtk, triangulate
from csemlib.background.grid_data import GridData
from csemlib.io.exodus_reader import ExodusReader
from csemlib.models import one_dimensional
from csemlib.models.s20rts import S20rts
from csemlib.csem.evaluate_csem import evaluate_csem


def depth_slice_to_vtk(depth, resolution, parameter="vsv", filename=None):
    """
    Writes a spherical slice to the VTK format for visualization with e.g.
    Paraview.

    :param depth: Depth of the slice in km
    :param resolution: Distance between grid points in km
    :param parameter: Name of parameter to be plotted. Choose from: vsv, vsh,
    rho, vpv, vph, eta. Defaults to vsv.
    :param filename: Name of the vtk depth slice, if none is given
    an automatic name will be chosen.
    """

    # Initialize grid points
    fib_grid = FibonacciGrid()
    fib_grid.set_global_sphere(radii=[6371.0-depth], resolution=[resolution])
    grid_data = evaluate_csem(*fib_grid.get_coordinates())

    x, y, z = grid_data.get_coordinates().T

    # Make vtk file.
    elements = triangulate(x, y, z)
    points = np.array((x, y, z)).T

    if filename is None:
        filename = f"depth_{str(depth)}_res_{str(resolution)}_" \
               f"{parameter}.vtk"

        # filename = os.path.join(".", name)
    print('Writing vtk to {}'.format(filename))
    write_vtk(filename, points, elements, grid_data.df[parameter],
              name=parameter)


def write_csem2emc(parameter_file):
    """
    This uses the former csem2emc script and writes a csem extraction
    into the emc format specifified by the paramter file.
    :param parameter_file: Name of the parameter file.
    """
    from csemlib.csem.csem2emc import csem2emc
    csem2emc(parameter_file)


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


def csem2salvus_mesh(mesh):
    """
    Adds CSEM to a Salvus mesh object. Acts on the obj
    :param mesh: salvus.mesh.UnstructuredMesh object
    :return:
    """

    # Map points to the sphere to easy point finding
    x, y, z = mesh.points.T / 1000.0
    rad = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    _, i = np.unique(mesh.connectivity, return_index=True)
    rad_1d = mesh.element_nodal_fields["z_node_1D"].flatten()[i]
    r_earth = 6371.0
    x[rad > 0] = x[rad > 0] * r_earth * rad_1d[rad > 0] / rad[rad > 0]
    y[rad > 0] = y[rad > 0] * r_earth * rad_1d[rad > 0] / rad[rad > 0]
    z[rad > 0] = z[rad > 0] * r_earth * rad_1d[rad > 0] / rad[rad > 0]
    rad = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    # Average on the discontinuities
    epsilon = 0.5
    rad_plus_eps = rad + epsilon
    rad_minus_eps = rad - epsilon
    regions_plus_eps = one_dimensional.get_region(rad_plus_eps)
    regions_min_eps = one_dimensional.get_region(rad_minus_eps)

    grid_data = evaluate_csem(x, y, z, regions=regions_plus_eps,
                              regions_2=regions_min_eps)

    dimensionless_components = ["eta", "QKappa", "Qmu"]
    for component in grid_data.components:
        if component in dimensionless_components:
            vals = grid_data.get_component(component)
        else:
            # Convert to m/s
            vals = grid_data.get_component(component) * 1000

        mesh.element_nodal_fields['%s' % (component.upper())][:] = \
            vals[mesh.connectivity]


def csem_to_csv(lats, lons, depths, filename="csem.csv"):
    """
    Writea a CSEM extraction to CSV file for a grid of lats, lons, depths.
    Provide the coordinates only, this function automatically creates
    the grid

    :param lats: latitude array in degrees. [-90,90].
    E.g.: np.linspace(45.0, 60.0, 16)
    :param lons: Longitude array in degrees [-180,180].

    :param depths: Depth in km.
    :param filename: Name of the CSV file

    """

    from csemlib.tools.utils import sph2cart, lat2colat
    import pandas as pd

    rads = 6371.0 - depths
    all_lats, all_lons, all_rads = np.meshgrid(lats, lons, rads)
    all_lats, all_lons, all_rads = np.array(
            (all_lats.ravel(), all_lons.ravel(), all_rads.ravel())
        )

    # Convert to CSEM coordinates
    all_colats = lat2colat(all_lats)
    all_colats_rad = np.deg2rad(all_colats)
    all_lons_rad = np.deg2rad(all_lons)
    x, y, z = sph2cart(all_colats_rad, all_lons_rad, all_rads)

    # Evaluate grid points
    grid_data = evaluate_csem(x,y,z)

    # Write gridpoints to CSV
    df = pd.DataFrame({'lats': all_lats, 'lons': all_lons, "depths": 6371.0 - all_rads,
                "VSV":grid_data.df["vsv"].values,
                "VSH":grid_data.df["vsh"].values,
                "VPV":grid_data.df["vpv"].values,
                "VPH":grid_data.df["vph"].values,
                "RHO":grid_data.df["rho"].values,
                "ETA":grid_data.df["eta"].values,
                  })
    df.to_csv(filename, index=False)



# def put_csem(mesh):
#
#     """
#     Deprecated function that is slower, but less memory intensive.
#     It places csem on a salvus grid and does it for the element's gll points
#     one by one. This might be useful at some point.
#     :param mesh: salvus mesh object
#     :param kwargs:
#     :return:
#     """
#
#     nodes_per_element = 27
#     # convert x,y,z to spherical shhape
#
#     for i in np.arange(nodes_per_element):
#         print('Adding CSEM to node {} out of {}'.format(i + 1,nodes_per_element))
#
#         x, y, z = mesh.points[mesh.connectivity[:, i]].T / 1000.0
#         rad = np.sqrt(x ** 2 + y ** 2 + z ** 2)
#         # get 1D_radius (normalized from 0 to 1)
#         print("Accounting for topography")
#         rad_1D = mesh.element_nodal_fields["z_node_1D"][:, i]
#         r_earth = 6371.0
#         x[rad > 0] = x[rad > 0] * r_earth * rad_1D[rad > 0] / rad[rad > 0]
#         y[rad > 0] = y[rad > 0] * r_earth * rad_1D[rad > 0] / rad[rad > 0]
#         z[rad > 0] = z[rad > 0] * r_earth * rad_1D[rad > 0] / rad[rad > 0]
#         rad = np.sqrt(x ** 2 + y ** 2 + z ** 2)
#
#         epsilon = 0.5
#         rad_plus_eps = rad + epsilon
#         rad_minus_eps = rad - epsilon
#         regions_plus_eps = one_dimensional.get_region(rad_plus_eps)
#         regions_min_eps = one_dimensional.get_region(rad_minus_eps)
#
#         grid_data = evaluate_csem(x, y, z, regions=regions_plus_eps,
#                                   regions_2=regions_min_eps)
#         dimensionless_components = ["eta", "QKappa", "Qmu"]
#         for component in grid_data.components:
#             # Convert to m/s
#             if component in dimensionless_components:
#                 vals = grid_data.get_component(component)
#             else:
#                 vals = grid_data.get_component(component) * 1000
#             mesh.element_nodal_fields['%s' % (component.upper())][:, i] = vals
#
#     return mesh
