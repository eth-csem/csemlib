import numpy as np

from ..background.grid_data import GridData
from ..models.s20rts import S20rts
from ..models.crust import Crust
from ..models import one_dimensional
from ..models.ses3d_rbf import Ses3d_rbf


def _evaluate_csem_salvus(x, y, z, regions, **kwargs):
    # Default parameters
    params = dict(eval_crust=False, eval_s20=False)
    params.update(kwargs)

    grid_data = GridData(x, y, z, solver='salvus')
    grid_data.add_one_d_salvus(regions)

    # Add s20
    if params['eval_s20']:
        s20 = S20rts()
        s20.eval_point_cloud_griddata(grid_data)

    # if params['eval_japan']:
    #     mod = Ses3d_rbf('/home/sed/CSEM/csemlib/ses3d_models/japan', grid_data.components)
    #     mod.eval_point_cloud_griddata(grid_data)

    # # Evaluate regional ses3d models. Requires models to be present still hardcoded replace this with smtng better
    # if params['eval_regional']:
    #     mod = Ses3d_rbf('/home/sed/CSEM/csemlib/ses3d_models/Australia', grid_data.components)
    #     mod.eval_point_cloud_griddata(grid_data)
    #
    #     mod = Ses3d_rbf('/home/sed/CSEM/csemlib/ses3d_models/south_atlantic', grid_data.components)
    #     mod.eval_point_cloud_griddata(grid_data)

    # Add crust
    if params['eval_crust']:
        cst = Crust()
        cst.eval_point_cloud_grid_data(grid_data)

    # # Add europe model on top of crust
    # if params['eval_regional']:
    #     mod = Ses3d_rbf('/home/sed/CSEM/csemlib/ses3d_models/europe_1s', grid_data.components)
    #     mod.eval_point_cloud_griddata(grid_data)

    return grid_data


def add_csem_discontinuous(salvus_mesh, **kwargs):
    # Default parameters
    params = dict(eval_crust=False, eval_s20=False)
    params.update(kwargs)

    # Get element centroids
    if salvus_mesh.ndim == 2:
        x_c, y_c = salvus_mesh.get_element_centroid().T
        z_c = np.zeros_like(x_c)
    else:
        x_c, y_c, z_c = salvus_mesh.get_element_centroid().T

    # Get region corresponding to element centroid
    rad_c = np.sqrt(x_c ** 2 + y_c ** 2 + z_c ** 2) * 6371.0 / salvus_mesh.scale
    regions = one_dimensional.get_region(rad_c)

    for i in np.arange(salvus_mesh.nodes_per_element):
        print('Adding CSEM to node {} out of {}'.format(i+1, salvus_mesh.nodes_per_element))
        if salvus_mesh.ndim == 2:
            x, y = salvus_mesh.points[salvus_mesh.connectivity[:, i]].T * 6371.0 / salvus_mesh.scale
            z = np.zeros_like(x)
        else:
            x, y, z = salvus_mesh.points[salvus_mesh.connectivity[:, i]].T * 6371.0 / salvus_mesh.scale
        grid_data = _evaluate_csem_salvus(x, y, z, regions, **params)

        for component in grid_data.components:
            # Convert to m/s
            vals = grid_data.get_component(component) * 1000
            salvus_mesh.attach_field('%s_%d' % (component.upper(), i), vals)

    # Attach fluid field
    salvus_mesh.attach_field('fluid', np.array(regions == 1, dtype=int))

    return salvus_mesh


def add_csem_continuous(salvus_mesh, **kwargs):
    # Default parameters
    params = dict(eval_crust=False, eval_s20=False)
    params.update(kwargs)

    # 2D case
    if salvus_mesh.ndim == 2:
        x, y = salvus_mesh.points.T * 6371.0 / salvus_mesh.scale
        z = np.zeros_like(x)
    # 3D
    else:
        x, y, z = salvus_mesh.points.T * 6371.0 / salvus_mesh.scale
    rad = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    # Get region corresponding to element centroid
    regions = one_dimensional.get_region(rad)
    grid_data = _evaluate_csem_salvus(x, y, z, regions, **params)

    for component in grid_data.components:
        vals = grid_data.get_component(component) * 1000
        salvus_mesh.attach_field('%s' % component.upper(), vals)

    # Attach fluid field, (meshes generated with this function normally don't have fluids
    #salvus_mesh.attach_field('fluid', np.array(regions == 1, dtype=int))

    return salvus_mesh


