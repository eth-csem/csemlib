import os

from csemlib.background.grid_data import GridData
from csemlib.models.crust import Crust
from csemlib.models.s20rts import S20rts
from csemlib.models.ses3d import Ses3d
from csemlib.models.specfem import Specfem


def evaluate_csem(x, y, z, regions=None, regions_2=None):

    """
    Evaluate the CSEM on Cartesian grid points x, y, z, provided as input. Return a GridData object.

    :param x: np array of x-coordinates in kilometers
    :param y: np array of y-coordinates in kilometers
    :param z: np array of y-coordinates in kilometers
    :param regions: specifies the region where the point(x,y,z) falls into in the one dimensional Earth model.
    :param regions_2: secondary region used for averaging over discontinuities

    :return: a grid_data object with the required information contained on it.
    """

    csemlib_directory, _ = os.path.split(os.path.split(__file__)[0])
    model_dir = os.path.join(csemlib_directory,'csemlib','data','refinements')

    print(model_dir)

    grid_data = GridData(x, y, z)

    # Base Model. ------------------------------------------------------------------------------------------------------

    # Add 1D background model to a mesh with or without explicit discontinuities.
    if (regions is not None) and (regions_2 is not None):
        grid_data.add_one_d_continuous(region_min_eps=regions, region_plus_eps=regions_2)
    elif regions is not None:
        grid_data.add_one_d_discontinuous(regions)
    else:
        grid_data.add_one_d()


    # Add S20RTS 
    s20 = S20rts()
    s20.eval_point_cloud_griddata(grid_data)


    # Regional refinements and crustal model. --------------------------------------------------------------------------

    # Models without crust that must be added before adding the crust.

    # Add South Atlantic
    ses3d = Ses3d(os.path.join(model_dir, 'south_atlantic_2013'), grid_data.components, interp_method='nearest_neighbour')
    ses3d.eval_point_cloud_griddata(grid_data)

    # Add Australia
    ses3d = Ses3d(os.path.join(model_dir, 'australia_2010'), grid_data.components, interp_method='nearest_neighbour')
    ses3d.eval_point_cloud_griddata(grid_data)


    # Overwrite crustal values.
    cst = Crust()
    cst.eval_point_cloud_grid_data(grid_data)


    # # Add 3D models with crustal component.
    
    # Add Japan
    ses3d = Ses3d(os.path.join(model_dir, 'japan_2016'), grid_data.components, interp_method='nearest_neighbour')
    ses3d.eval_point_cloud_griddata(grid_data)

    # Add Europe
    ses3d = Ses3d(os.path.join(model_dir, 'europe_2013'), grid_data.components, interp_method='nearest_neighbour')
    ses3d.eval_point_cloud_griddata(grid_data)

    # Add Marmara
    ses3d = Ses3d(os.path.join(model_dir, 'marmara_2017'), grid_data.components, interp_method='nearest_neighbour')
    ses3d.eval_point_cloud_griddata(grid_data)
    
    # Add South-East Asia
    ses3d = Ses3d(os.path.join(model_dir, 'south_east_asia_2017'), grid_data.components, interp_method='nearest_neighbour')
    ses3d.eval_point_cloud_griddata(grid_data)

    # Add Iberia 2015
    ses3d = Ses3d(os.path.join(model_dir, 'iberia_2015'), grid_data.components, interp_method='nearest_neighbour')
    ses3d.eval_point_cloud_griddata(grid_data)
    
    # Add Iberia 2017
    ses3d = Ses3d(os.path.join(model_dir, 'iberia_2017'), grid_data.components, interp_method='nearest_neighbour')
    ses3d.eval_point_cloud_griddata(grid_data)
    
    # Add North Atlantic 2013
    ses3d = Ses3d(os.path.join(model_dir, 'north_atlantic_2013'), grid_data.components, interp_method='nearest_neighbour')
    ses3d.eval_point_cloud_griddata(grid_data)

    # Add North America 2017
    ses3d = Ses3d(os.path.join(model_dir, 'north_america_2017'), grid_data.components, interp_method='nearest_neighbour')
    ses3d.eval_point_cloud_griddata(grid_data)

    # Add Japan 2017
    ses3d = Ses3d(os.path.join(model_dir, 'japan_2017'), grid_data.components, interp_method='nearest_neighbour')
    ses3d.eval_point_cloud_griddata(grid_data)

    # Add Eastern Mediterranean 2019
    ses3d = Ses3d(os.path.join(model_dir, 'eastern_mediterranean_2019'), grid_data.components, interp_method='nearest_neighbour')
    ses3d.eval_point_cloud_griddata(grid_data)

    # Add Michael Afanasiev's global update
    global1 = Specfem(interp_method="trilinear_interpolation")
    global1.eval_point_cloud_griddata(grid_data)


    return grid_data
