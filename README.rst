.. highlight:: rst

=======
csemlib
=======

-----------------------------------------------
Python package that enables extracting the CSEM
-----------------------------------------------

Note that CSEM extractions are only possible, if you have access to the regional submodels. 

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


^^^^^^^
Example
^^^^^^^

The code block below shows an example where the CSEM is extracted onto a spherical depth slice at 200 km depth.


.. code-block:: python

   from csenlib.api import depth_slice_to_vtk
   depth_slice_to_vtk(depth=200, resolution=200, parameter="vsv", filename="extraction.vtk")

This writes a VTK file that can be visualized with Paraview, for example.
