.. highlight:: rst

=======
csemlib
=======

-----------------------------------------------
Python package that enables extracting the CSEM
-----------------------------------------------

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Installation on Linux and MAC OS X
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* Install the latest version of `Anaconda <https://www.continuum.io/downloads>`_ for Python 3.x
* Clone this repository by typing the following commands to the terminal::

     git clone https://github.com/eth-csem/csemlib.git

* To ensure correct installation create a conda environment by typing::

     conda create --name csemlib-env python=3 numpy numba
     source activate csemlib-env

* Change directory to csemlib and install by typing::

     cd csemlib
     pip install -v -e .

* Test for correct installation by running the following command::

    py.test

* If installation was successful all tests should be completed succesfully. If not, this could be related to additional missing dependencies, look at the *travis.yml* for inspiration. This file describes an installation with the additional system dependencies.


^^^^^^^
Example
^^^^^^^

.. code-block:: python
    :linenos:

   import numpy as np
   from csemlib.background.fibonacci_grid import FibonacciGrid
   from csemlib.background.grid_data import GridData\
   from csemlib.models.crust import Crust
   from csemlib.models.model import triangulate, write_vtk
   from csemlib.models.S20RTS.s20rts_f2py import S20rts_f2py
   
   # Generate Grid
   rad = 6250.0
   fib_grid = FibonacciGrid()
   radii = np.array([rad])
   resolution = np.ones_like(radii) * 200
   fib_grid.set_global_sphere(radii, resolution)
   grid_data = GridData(*fib_grid.get_coordinates())
   
   # Add 1d background model and initalize a 'vsv' array with values
   grid_data.add_one_d()
   grid_data.set_component('vsv', grid_data.df['one_d_vsv'])
   
   # Evaluate S20RTS
   s20 = S20rts_f2py()
   s20.eval_point_cloud_griddata(grid_data)
   
   # Evaluate Crust
   cst = Crust()
   cst.eval_point_cloud_grid_data(grid_data)

   # Write to a vtk file for easy visualisation in Paraview
   x, y, z = grid_data.get_coordinates().T
   elements = triangulate(x, y, z)
   coords = np.array((x, y, z)).T

   write_vtk('prem_s20_crust_vsv.vtk'), coords, elements, grid_data.get_component('vsv'), 'vsv')

