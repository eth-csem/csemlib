import numpy as np
import pyexodus
import yaml
import os
import io

from scipy import spatial

from csemlib.utils import cart2sph
from ..helpers import load_lib

lib = load_lib()


class Salvus_v1(object):
    """
    This Class handles reading and evaluating Salvus V1 model files.
    It requires the initial and final model as well as the radius_1D field
    to correct applied topography or ellipticity.
    """

    def __init__(self, directory):
        """
        :param directory: Directory where the final and initial model are
        stored. The files should be called final.e and initial.e respectively.
        """
        self.directory = directory
        self.perturbations = {}
        self.points = None
        self.radius_1d = None
        self.nelem = None
        self.ndim = 3
        self.nodes_per_element = None
        self.connectivity = None
        self.centroids = None
        self.params = ["VSV", "VSH", "VPV", "VPH"]
        self.read()

    def read(self):
        """ Reads the original exodus model files."""
        final_file = os.path.join(self.directory, "final.e")
        initial_file = os.path.join(self.directory, "initial.e")

        # Read yaml file containing information on the ses3d submodel.
        with io.open(os.path.join(self.directory, 'modelinfo.yml'), 'rt') as fh:
            try:
                self.model_info = yaml.load(fh, Loader=yaml.FullLoader)
                print('Evaluating Salvus 1 model: {}'.format(self.model_info['model']))
            except yaml.YAMLError as exc:
                print(exc)

        # Read perturbations
        for param in self.params:
            with pyexodus.exodus(final_file) as e_final:
                val = e_final.get_node_variable_values(param, step=1)
            with pyexodus.exodus(initial_file) as e_init:
                val -= e_init.get_node_variable_values(param, step=1)
                self.perturbations[param] = val

        # read points
        with pyexodus.exodus(initial_file) as e_init:
            # get coords and convert to km
            x, y, z = e_init.get_coords()
            self.points = np.array((x, y, z)).T / 1000.0

            if "radius_1D" in e_init.get_node_variable_names():
                self.radius_1d = e_init.get_node_variable_values("radius_1D",
                                                                 step=1)

            rads_3d = np.sqrt(self.points[:, 0] ** 2 + self.points[:, 1] ** 2 +
                              self.points[:, 2] ** 2)
            # convert coordinates to CSEM reference frame
            self.points[:, 0] = self.points[:, 0] * self.radius_1d * \
                                6371.0 / rads_3d
            self.points[:, 1] = self.points[:, 1] * self.radius_1d * \
                                6371.0 / rads_3d
            self.points[:, 2] = self.points[:, 2] * self.radius_1d * \
                                6371.0 / rads_3d

            self.connectivity, self.nelem, self.nodes_per_element = \
                e_init.get_elem_connectivity(id=1)
            # subtract 1, because exodus uses one based numbering
            self.connectivity -= 1
            self.connectivity = self.connectivity.astype("int64")

    def get_element_centroid(self):
        """
        Compute the centroids of all elements on the fly from the nodes of the
        mesh. Useful to determine which domain in a layered medium an element
        belongs to or to compute elemental properties from the model.
        """
        self.centroids = np.ascontiguousarray(
            np.zeros((self.nelem, self.ndim)))

        lib.centroid(self.ndim, self.nelem, self.nodes_per_element,
                     np.ascontiguousarray(self.connectivity),
                     np.ascontiguousarray(self.points),
                     self.centroids)
        # for element in self.connectivity

    def extract_domain(self, GridData):
        """ This function extracts the chunk of the point cloud that lie
        near to the salvus_v1 model."""

        salvus_dmn = GridData.copy()
        c, l, r = cart2sph(self.points[:, 0], self.points[:, 1],
                           self.points[:, 2])
        c_min = np.min(c)
        c_max = np.max(c)

        l_min = np.min(l)
        l_max = np.max(l)
        r_min = np.min(r)

        # Extract rough domain
        salvus_dmn.df = salvus_dmn.df[salvus_dmn.df['c'] > c_min]
        salvus_dmn.df = salvus_dmn.df[salvus_dmn.df['c'] < c_max]
        salvus_dmn.df = salvus_dmn.df[salvus_dmn.df['l'] > l_min]
        salvus_dmn.df = salvus_dmn.df[salvus_dmn.df['l'] < l_max]
        salvus_dmn.df = salvus_dmn.df[salvus_dmn.df['r'] > r_min]
        return salvus_dmn

    def eval_point_cloud_griddata(self, GridData):
        """
        :param GridData: GridData object
        """
        salvus_dmn = self.extract_domain(GridData)
        self.trilinear_interpolation_in_c(salvus_dmn, GridData)

    def trilinear_interpolation_in_c(self, salvus_dmn, GridData,
                                     nelem_to_search=10):
        """

        :param salvus_dmn: Subset of GridData that falls into that specific
        salvus_dmn subdomain
        :param GridData: Master GridData
        :param nelem_to_search: number of surrounding elements that are
        searched to find the
        enclosing element.
        """
        # Generate KD-Tree from centroids
        self.get_element_centroid()
        centroid_tree = spatial.cKDTree(self.centroids, balanced_tree=False)

        # Get list of tuples (dist, index) sorted on distance
        points = salvus_dmn.get_coordinates(coordinate_type='cartesian')
        _, nearest_element_indices = centroid_tree.query(points,
                                                         k=nelem_to_search)

        # reorder connectivity array to match ordering of interpolation routine
        permutation = [0, 3, 2, 1, 4, 5, 6, 7]
        i = np.argsort(permutation)
        connectivity_reordered = self.connectivity[:, i]

        # connectivity_reordered = self.connectivity
        nelem = len(connectivity_reordered)
        npoints = len(salvus_dmn)
        npoints_mesh = len(self.points)
        enclosing_elem_indices = np.zeros((npoints, 8), dtype=np.int64)

        weights = np.zeros((npoints, 8))

        n_not_found = \
            lib.triLinearInterpolator(nelem, nelem_to_search, npoints,
                                      npoints_mesh,
                                      np.ascontiguousarray(
                                          nearest_element_indices),
                                      np.ascontiguousarray(
                                          connectivity_reordered),
                                      enclosing_elem_indices,
                                      np.ascontiguousarray(self.points),
                                      np.ascontiguousarray(weights),
                                      np.ascontiguousarray(points))

        # Divide by 1000 because csemlib expects velocities in km/s
        for param in self.params:
            salvus_dmn.df[:][param.lower()] += np.sum(
                self.perturbations[param][enclosing_elem_indices] *
                weights, axis=1) / 1000.0

        GridData.df.update(salvus_dmn.df)
