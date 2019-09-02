from pyexodus import exodus
import numpy as np
from ..helpers import load_lib

lib = load_lib()


class ExodusReader(object):
    """
    This class reads variables from an exodus file into memory, currently only supports
    one element block
    """
    def __init__(self, filename, mode='r'):
        self._filename = filename
        assert mode in ['a', 'r'], "Only mode 'a', 'r' is supported"
        self.mode = mode
        self.connectivity = None
        self.nodes_per_element = None
        self.ndim = None
        self.nelem = None
        self.x = None
        self.y = None
        self.z = None
        self.elem_var_names = None
        self.points = None
        self.e = exodus(self._filename, self.mode)

        # Read File
        self._read()

    def _read(self):
        self.ndim = self.e.num_dims
        assert self.ndim in [2, 3], "Only '2D', '3D' exodus files are supported."
        self.connectivity, self.nelem, self.nodes_per_element = self.e.get_elem_connectivity(id=1)
        self.connectivity = np.array(self.connectivity, dtype='int64', ) - 1
        self.elem_var_names = self.e.get_element_variable_names()
        self.x, self.y, self.z = self.e.get_coords()
        self.points = np.array((self.x, self.y, self.z)).T

    def get_element_centroid(self):
        """
        Compute the centroids of all elements on the fly from the nodes of the
        mesh. Useful to determine which domain in a layered medium an element
        belongs to or to compute elemental properties from the model.
        """
        centroid = np.zeros((self.nelem, self.ndim))
        lib.centroid(self.ndim, self.nelem, self.nodes_per_element,
                     self.connectivity, np.ascontiguousarray(self.points), centroid)
        return centroid

    def attach_field(self, name, values):
        """
        Write values with name to exodus file
        :param name: name of the variable to be written
        :param values: numpy array of values to be written
        :return:
        """

        assert self.mode in ['a'], "Attach field option only available in mode 'a'"

        if values.size == self.nelem:
            self.e.put_element_variable_values(blockId=1, name=name, step=1, values=values)

        elif values.size == self.npoint:
            self.e.put_node_variable_values(name, 1, values)

        else:
            raise ValueError('Shape matches neither the nodes nor the '
                             'elements')

    def get_element_field(self, name):
        """
        Get values from elemental field.
        :param name: name of the variable to be retrieved
        :return element field values:
        """

        assert self.mode in ['r', 'a'], "Attach field option only available in mode 'r' or 'a'"
        assert name in self.elem_var_names, "Could not find the requested field"

        values = self.e.get_element_variable_values(blockId=1, name=name, step=1)
        return values

    def get_nodal_field(self, name):
        """
        Get values from nodal field.
        :param name: name of the variable to be retrieved
        :return nodal field values:
        """

        assert self.mode in ['r', 'a'], "Attach field option only available in mode 'r' or 'a'"
        assert name in self.e.get_node_variable_names(), "Could not find the requested field"

        values = self.e.get_node_variable_values(name=name, step=1)
        return values


    @property
    def npoint(self):
         """
         Number of points / nodes in the mesh
         """
         return self.points.shape[0]


    def close(self):
        self.e.close()
