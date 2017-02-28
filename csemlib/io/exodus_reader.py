from pyexodus import exodus

class ExodusReader:
    """
    This class reads variables from an exodus file into memory, currently only supports
    one element block
    """
    def __init__(self, filename, mode='r'):
        self._filename = filename
        self.mode = mode
        self.connectivity = None
        self.num_elements = None
        self.num_nodes_per_elem = None
        self.num_dims = None
        self.x = None
        self.y = None
        self.z = None
        self.elem_var_names = None

    def read(self):
        with exodus(self._filename, self.mode) as e:
            self.num_dims = e.num_dims
            assert self.num_dims in [2, 3], "Only '2D', '3D' exodus files are supported."
            self.connectivity, self.num_elements, self.num_nodes_per_elem = e.get_elem_connectivity(id=1)
            self.elem_var_names = e.get_element_variable_names()
            self.x, self.y, self.z = e.get_coords()
            e.close()


