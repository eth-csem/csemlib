.. highlight:: rst

=======
csemlib
=======

-----------------------------------------------
Python package that enables extracting the CSEM
-----------------------------------------------

Note that CSEM extractions are only possible, if you have access to the regional submodels.
In addition, you gcc, should be installed before installation. (This is currently still a legacy from the
fact that we historically used Fortran to evaluate the S20RTS spherical harmonics).

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Installation on Linux and Mac OS X
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* Install the latest version of `Anaconda <https://www.continuum.io/downloads>`_ for Python 3.x
* Clone this repository by typing the following commands to the terminal::

     git clone https://github.com/eth-csem/csemlib.git

* csemlib maybe installed into an existing Python 3.x environment. To ensure correct installation create a conda environment by typing::

     conda create --name csemlib-env python=3 numpy scipy pytest cython numba xarray pandas matplotlib PyYAML
     source activate csemlib-env
     
* Alternatively, you may skip the above step.

* Change directory to csemlib and install by typing::

     cd csemlib
     pip install -v -e .

* Test for correct installation by running the following command::

    py.test

* If installation was successful all tests should be completed succesfully. If not, this could be related to additional missing dependencies, look at the *travis.yml* for inspiration. This file describes an installation with the additional system dependencies.


^^^^^^^^^^^^^
Example usage
^^^^^^^^^^^^^

The code block below shows an example where the CSEM is extracted onto a spherical depth slice at 200 km depth.
This writes a VTK file that can be visualized with Paraview, for example.


.. code-block:: python

   from csenlib.api import csem2vtk
   csem2vtk(depth=200, resolution=200, parameter="vsv", filename="extraction.vtk")


Below is an example of an extraction of the CSEM on to a grid, that is then written
to the CSV file format.

.. code-block:: python

   import numpy as np

   latitudes = np.linspace(30, 60, 31)
   longitudes = np.linspace(40, 90, 51)
   depths = np.linspace(0, 600, 31)

   from csenlib.api import csem2csv
   csem2csv(latitudes, longitudes, depths, filename="csem_extraction.csv")

In the following example, we make an extraction into the IRIS EMC file format.
This works with a parameters.yml file of which an example can be found in csemlib/scripts/parameters.yml.

.. code-block:: python

    from csemlib.api import csem2emc
    csem2emc("parameters.yml")

In the next example, we add the CSEM model onto a salvus mesh object. See www.mondaic.com for more information
on salvus meshes. Here the salvus.mesh.UnstructuredMesh object to be defined of course and it should use the
same parameterization as the CSEM. Ellipticity and topography are automatically taken into account upon
extraction.

.. code-block:: python

    from csemlib.api import csem2salvus_mesh
    csem2salvus_mesh(mesh)


