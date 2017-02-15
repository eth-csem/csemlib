import numpy as np

from ..background.grid_data import GridData
from ..models.S20RTS.s20rts_f2py import S20rts_f2py
from ..models.crust import Crust
from ..models import one_dimensional
from ..models.ses3d_rbf import Ses3d_rbf


def _evaluate_csem_salvus(x, y, z, regions, **kwargs):
    # Default parameters
    params = dict(eval_crust=True, eval_s20=True, eval_regional=False)
    params.update(kwargs)

    grid_data = GridData(x, y, z, solver='salvus')
    grid_data.add_one_d_salvus(regions)

    # Add s20
    if params['eval_s20']:
        s20 = S20rts_f2py()
        s20.eval_point_cloud_griddata(grid_data)

    # Evaluate regional ses3d models. Requires models to be present still hardcoded replace this with smtng better
    if params['eval_regional']:
        mod = Ses3d_rbf('/home/sed/CSEM/csemlib/ses3d_models/japan', grid_data.components)
        mod.eval_point_cloud_griddata(grid_data)

        mod = Ses3d_rbf('/home/sed/CSEM/csemlib/ses3d_models/Australia', grid_data.components)
        mod.eval_point_cloud_griddata(grid_data)

        mod = Ses3d_rbf('/home/sed/CSEM/csemlib/ses3d_models/south_atlantic', grid_data.components)
        mod.eval_point_cloud_griddata(grid_data)

    # Add crust
    if params['eval_crust']:
        cst = Crust()
        cst.eval_point_cloud_grid_data(grid_data)

    # Add europe model on top of crust
    if params['eval_regional']:
        mod = Ses3d_rbf('/home/sed/CSEM/csemlib/ses3d_models/europe_1s', grid_data.components)
        mod.eval_point_cloud_griddata(grid_data)

    return grid_data


def add_csem_to_salvus_mesh(salvus_mesh, **kwargs):
    # Default parameters
    params = dict(eval_crust=True, eval_s20=True, eval_regional=False)
    params.update(kwargs)

    # Get element centroids
    x_c, y_c, z_c = salvus_mesh.get_element_centroid().T
    rad_c = np.sqrt(x_c ** 2 + y_c ** 2 + z_c ** 2) * 6371.0 / salvus_mesh.scale

    # Get region corresponding to element centroid
    regions = one_dimensional.get_region(rad_c)

    for i in np.arange(salvus_mesh.nodes_per_element):
        x, y, z = salvus_mesh.points[salvus_mesh.connectivity[:, i]].T * 6371.0 / salvus_mesh.scale
        grid_data = _evaluate_csem_salvus(x, y, z, regions)

        for component in grid_data.components:
            salvus_mesh.attach_field('%s_%d' % (component.upper(), i), grid_data.get_component(component))

    # Attach fluid field
    salvus_mesh.attach_field('fluid', np.array(regions == 1, dtype=int))

    return salvus_mesh

