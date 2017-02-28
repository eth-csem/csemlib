from pyexodus import exodus
import numpy as np
from ..lib.helpers import load_lib

lib = load_lib()

class ExodusReader:
    """
    This class reads variables from an exodus file into memory, currently only supports
    one element block
    """
    def __init__(self, filename, mode='r'):
        self._filename = filename
        self.mode = mode
        self.connectivity = None
        self.nelem = None
        self.nodes_per_element = None
        self.ndim = None
        self.x = None
        self.y = None
        self.z = None
        self.elem_var_names = None
        self.points = None
        # Read File
        self._read()

    def _read(self):
        with exodus(self._filename, self.mode) as e:
            self.ndim = e.num_dims
            assert self.ndim in [2, 3], "Only '2D', '3D' exodus files are supported."
            self.connectivity, self.nelem, self.nodes_per_element = e.get_elem_connectivity(id=1)
            self.connectivity = np.array(self.connectivity, dtype='int64', ) - 1
            self.elem_var_names = e.get_element_variable_names()
            self.x, self.y, self.z = e.get_coords()
            e.close()

        self.points = np.array((self.x, self.y, self.z)).T

    def get_element_centroid(self):
        '''
        Compute the centroids of all elements on the fly from the nodes of the
        mesh. Usefull to determine which domain in a layered medium an element
        belongs to or to compute elemtal properties from the model.
        '''
        centroid = np.zeros((self.nelem, self.ndim))
        lib.centroid(self.ndim, self.nelem, self.nodes_per_element,
                     self.connectivity, np.ascontiguousarray(self.points), centroid)
        return centroid


