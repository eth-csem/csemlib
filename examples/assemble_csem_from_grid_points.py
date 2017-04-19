import numpy as np

from csemlib.background.grid_data import GridData
from csemlib.models.crust import Crust
from csemlib.models.model import write_vtk, triangulate
from csemlib.models.s20rts import S20rts
from csemlib.models.ses3d import Ses3d
from csemlib.io.readers import read_from_grid

import os


csemlib_directory, _ = os.path.split(os.path.split(__file__)[0])
model_directory = os.path.join(csemlib_directory, 'regional_models')

depths = [413]

for depth in depths:

    # Read some grid points. -------------------------------------------------------------------------------------------
    x, y, z = read_from_grid('../grid/OUTPUT/fib_'+str(depth)+'.dat')
    grid_data = GridData(x, y, z)

    # 1D background plus S20RTS. ---------------------------------------------------------------------------------------

    # Initialise with CSEM 1D background model.
    grid_data.add_one_d()
    grid_data.set_component('vsv', grid_data.df['one_d_vsv'])

    # Add s20rts
    s20 = S20rts()
    s20.eval_point_cloud_griddata(grid_data)

    # Models without crust that must be added before adding the crust. -------------------------------------------------

    # Add South Atlantic
    ses3d = Ses3d(os.path.join(model_directory, 'south_atlantic_2013'), grid_data.components)
    ses3d.eval_point_cloud_griddata(grid_data)

    # Add Australia
    ses3d = Ses3d(os.path.join(model_directory, 'australia_2010'), grid_data.components)
    ses3d.eval_point_cloud_griddata(grid_data)

    # Overwrite crustal values. ----------------------------------------------------------------------------------------

    # Add Crust
    cst = Crust()
    cst.eval_point_cloud_grid_data(grid_data)

    # Add 3D models with crustal component. ----------------------------------------------------------------------------

    # Add Japan
    ses3d = Ses3d(os.path.join(model_directory, 'japan_2016'), grid_data.components)
    ses3d.eval_point_cloud_griddata(grid_data)

    # Add Europe
    ses3d = Ses3d(os.path.join(model_directory, 'europe_2013'), grid_data.components)
    ses3d.eval_point_cloud_griddata(grid_data)

    # Add Marmara
    ses3d = Ses3d(os.path.join(model_directory, 'marmara_2017'), grid_data.components)
    ses3d.eval_point_cloud_griddata(grid_data)
    
    # Add South-East Asia
    ses3d = Ses3d(os.path.join(model_directory, 'south_east_asia_2017'), grid_data.components)
    ses3d.eval_point_cloud_griddata(grid_data)

    # Add Iberia 2015
    ses3d = Ses3d(os.path.join(model_directory, 'iberia_2015'), grid_data.components)
    ses3d.eval_point_cloud_griddata(grid_data)
    
    # Add Iberia 2017
    ses3d = Ses3d(os.path.join(model_directory, 'iberia_2017'), grid_data.components)
    ses3d.eval_point_cloud_griddata(grid_data)
    
    # Add North Atlantic 2013
    ses3d = Ses3d(os.path.join(model_directory, 'north_atlantic_2013'), grid_data.components)
    ses3d.eval_point_cloud_griddata(grid_data)

    # Add North America 2017
    ses3d = Ses3d(os.path.join(model_directory, 'north_america_2017'), grid_data.components)
    ses3d.eval_point_cloud_griddata(grid_data)

    # Generate output. -------------------------------------------------------------------------------------------------

    # Make vtk file.
    elements = triangulate(x, y, z)
    points = np.array((x, y, z)).T

    filename = os.path.join(csemlib_directory, str(depth)+'.vtk')
    write_vtk(filename, points, elements, grid_data.df['vsv'])
