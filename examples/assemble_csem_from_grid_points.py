import numpy as np
from csemlib.background.grid_data import GridData
from csemlib.models.crust import Crust
from csemlib.models.model import write_vtk, triangulate
from csemlib.models.s20rts import S20rts
from csemlib.models.ses3d import Ses3d
from csemlib.io.readers import read_from_grid


depths=[13,40]

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
    ses3d = Ses3d('/Users/Andreas/CSEM/csemlib/regional_models/south_atlantic', grid_data.components)
    ses3d.eval_point_cloud_griddata(grid_data)

    # Add Australia
    ses3d = Ses3d('/Users/Andreas/CSEM/csemlib/regional_models/australia', grid_data.components)
    ses3d.eval_point_cloud_griddata(grid_data)

    # Overwrite crustal values. ----------------------------------------------------------------------------------------

    # Add Crust
    cst = Crust()
    cst.eval_point_cloud_grid_data(grid_data)

    # Add 3D models with crustal component. ----------------------------------------------------------------------------

    # Add Japan
    ses3d = Ses3d('/Users/Andreas/CSEM/csemlib/regional_models/japan', grid_data.components)
    ses3d.eval_point_cloud_griddata(grid_data)

    # Add Europe
    ses3d = Ses3d('/Users/Andreas/CSEM/csemlib/regional_models/europe_1s', grid_data.components)
    ses3d.eval_point_cloud_griddata(grid_data)

    # Generate output. -------------------------------------------------------------------------------------------------

    #- Make vtk file.
    elements = triangulate(x, y, z)
    points = np.array((x, y, z)).T

    filename = '/Users/Andreas/CSEM/csemlib/'+str(depth)+'.vtk'
    write_vtk(filename, points, elements, grid_data.df['vsv'])
