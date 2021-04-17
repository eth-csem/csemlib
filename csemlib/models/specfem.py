import h5py
import numpy as np
import scipy.spatial as spatial
import os
import sys
import warnings

from csemlib.tools.helpers import load_lib
lib = load_lib()


class Specfem(object):
    """
    This Class handles reading and evaluating Specfen files
    """

    def __init__(self, interp_method='nearest_neighbour'):
        self.nodes = None
        self.connectivity = None
        self.betav = None
        self.betah = None
        self.r_earth = 6371.0
        self.r_CMB = 3480.0
        self.tolerance = -5.0  # tolerance for extracting the domain at the surface
        self.step_length = 0.066  # computed from a maximum change to the model of 1.4%
        self.ndim = 3
        self.nelem = None
        self.nodes_per_element = 8
        self.centroids = None
        self.interp_method = interp_method

        directory, _ = os.path.split(os.path.split(__file__)[0])
        self.directory = os.path.join(directory, 'data','refinements','global_2016')

        #self.read()
        self.read_from_hdf5()



    def read(self, write_to_h5=True):
        """
        Reads the kernels and mesh information, this is quite slow therefore the preferred option is
        to write this data to hdf5 and use this file for reading.
        :param write_to_h5: True or False
        """
        self.nodes = np.genfromtxt(os.path.join(self.directory, 'node_coordinates.txt')) * self.r_earth
        self.betav = np.genfromtxt(os.path.join(self.directory, 'reg_1_bulk_betav_kernel_smooth_values.txt'))
        self.betah = np.genfromtxt(os.path.join(self.directory, 'reg_1_bulk_betah_kernel_smooth_values.txt'))
        self.connectivity = np.genfromtxt(os.path.join(self.directory, 'node_connectivity.txt'), dtype='int64')
        self.nelem = len(self.connectivity)

        if write_to_h5:
            self.write_to_hdf5()

    def read_from_hdf5(self):
        """
        Reads kernel information from hdf5 file
        """
        # Open HDF5 file containing the ses3d model.
        filename = os.path.join(self.directory, "global_update.hdf5")
        f = h5py.File(filename, "r")

        # Get parameters
        self.connectivity = f['connectivity'][:]
        self.betav = f['betav'][:]
        self.betah = f['betah'][:]
        self.nodes = f['nodes'][:]
        f.close()

        self.nelem = len(self.connectivity)

    def write_to_hdf5(self):
        """
        Writes an hdf5 file with a default filename
        """
        filename = os.path.join(self.directory, "global_update.hdf5")
        f = h5py.File(filename, "w")

        f.create_dataset('connectivity', data=self.connectivity, dtype='int64')
        f.create_dataset('betav', data=self.betav, dtype='d')
        f.create_dataset('betah', data=self.betav, dtype='d')
        f.create_dataset('nodes', data=self.nodes, dtype='d')
        f.close()

    def eval_point_cloud_griddata(self, GridData):
        """
        This function interpolates the kernel onto the GridData structure,
        depending on how the Specfem object is initialized it will use
        nearest neighbour or trilinear interpolation.

        :param GridData: GridData object which contains
        :return: No return. GridData is updated internally.
        """
        specfem_dmn = self.extract_specfem_dmn(GridData)

        print('Evaluating 1st global update')

        if self.interp_method == 'nearest_neighbour':
            print('Performing nearest neighbour interpolation')

            # Generate KDTrees, needed later for interpolation.
            pnt_tree_orig = spatial.cKDTree(self.nodes, balanced_tree=False)
            self.nearest_neighbour_interpolation(pnt_tree_orig, specfem_dmn, GridData)

        elif self.interp_method == 'trilinear_interpolation':
            print('Performing trilinear interpolation')
            self.trilinear_interpolation_in_c(specfem_dmn, GridData)

    def extract_specfem_dmn(self, GridData):
        """
        Extract those points from the current collection of grid points that fall inside the specfem subdomain.
        :param GridData: GridData structure with collection of current grid points and their properties.
        :return: Subset of GridData that falls into that specific specfem subdomain.
        """

        specfem_dmn = GridData.copy()
        specfem_dmn.df = specfem_dmn.df[specfem_dmn.df['r'] >= self.r_CMB]
        specfem_dmn.df = specfem_dmn.df[specfem_dmn.df['r'] <= self.r_earth + self.tolerance]

        return specfem_dmn

    def nearest_neighbour_interpolation(self, pnt_tree_orig, specfem_dmn, GridData):
        """
        Implement nearest-neighbor interpolation.
        :param pnt_tree_orig: KDTree of the grid coordinates in the ses3d model.
        :param specfem_dmn: Subset of the GridData structure that falls into the specfem domain.
        :param GridData: Master GridData structure.
        :return: No return. GridData is updated internally.
        """

        # Get indices of the nearest neighbours
        _, indices = pnt_tree_orig.query(specfem_dmn.get_coordinates(coordinate_type='cartesian'), k=1)

        # update vsv
        specfem_dmn.df[:]['vsv'] += self.betav[indices] * self.step_length
        # update vsh
        specfem_dmn.df[:]['vsh'] += self.betah[indices] * self.step_length

        GridData.df.update(specfem_dmn.df)

    def get_element_centroid(self):
        """
        Compute the centroids of all elements on the fly from the nodes of the
        mesh. Useful to determine which domain in a layered medium an element
        belongs to or to compute elemental properties from the model.
        """
        self.centroids = np.zeros((self.nelem, self.ndim))
        lib.centroid(self.ndim, self.nelem, self.nodes_per_element,
                     self.connectivity, np.ascontiguousarray(self.nodes), self.centroids)

    def trilinear_interpolation(self, specfem_dmn, GridData, nelem_to_search=50):
        """

        :param specfem_dmn: Subset of GridData that falls into that specific specfem subdomain
        :param GridData: Master GridData
        :param nelem_to_search: number of surrounding elements that are searched to find the
        enclosing element.
        :return: No return. GridData is updated internally.
        """
        # Generate KD-Tree from centroids
        self.get_element_centroid()
        centroid_tree = spatial.cKDTree(self.centroids, balanced_tree=False)

        # Get list of tuples (dist, index) sorted on distance
        points = specfem_dmn.get_coordinates(coordinate_type='cartesian')
        _, element_indices = centroid_tree.query(points, k=nelem_to_search)

        # reorder connectivity array to match ordering of interpolation routine
        permutation = [0, 3, 2, 1, 4, 5, 6, 7]
        i = np.argsort(permutation)
        connectivity_reordered = self.connectivity[:, i]

        tlp = TriLinearInterpolator()
        for idx in range(len(points))[:]:
            element_indices_for_point = element_indices[idx, :]

            for ii in range(nelem_to_search):
                vtx = self.nodes[connectivity_reordered[element_indices_for_point[ii], :]]
                pnt = points[idx]

                solution = tlp.check_hull(pnt=pnt, vtx=vtx)

                # if point in element, do the interpolation
                if solution[0]:
                    weights = tlp.interpolate_at_point(solution[1])
                    vsv = self.betav[connectivity_reordered[element_indices_for_point[ii], :]]
                    vsv_at_point = np.sum(vsv*weights)
                    specfem_dmn.df['vsv'].loc[idx] += vsv_at_point * self.step_length

                    vsh = self.betah[connectivity_reordered[element_indices_for_point[ii], :]]
                    vsh_at_point = np.sum(vsh*weights)
                    specfem_dmn.df['vsh'].loc[idx] += vsh_at_point * self.step_length

                    break
                elif ii == nelem_to_search - 1:
                    warnings.warn('no element found for ' + str(pnt))

            if idx % 500 == 0:
                ind = float(idx)
                percent = ind / len(points) * 100.0
                sys.stdout.write("\rProgress: %.1f%% " % percent)
                sys.stdout.flush()
        sys.stdout.write("\r")

        # Update master GridData structure.
        GridData.df.update(specfem_dmn.df)

    def trilinear_interpolation_in_c(self, specfem_dmn, GridData, nelem_to_search=25):
        """

        :param specfem_dmn: Subset of GridData that falls into that specific specfem subdomain
        :param GridData: Master GridData
        :param nelem_to_search: number of surrounding elements that are searched to find the
        enclosing element.
        :return: No return. GridData is updated internally.
        """
        # Generate KD-Tree from centroids
        self.get_element_centroid()
        centroid_tree = spatial.cKDTree(self.centroids, balanced_tree=False)

        # Get list of tuples (dist, index) sorted on distance
        points = specfem_dmn.get_coordinates(coordinate_type='cartesian')
        _, element_indices = centroid_tree.query(points, k=nelem_to_search)

        # reorder connectivity array to match ordering of interpolation routine
        permutation = [0, 3, 2, 1, 4, 5, 6, 7]
        i = np.argsort(permutation)
        connectivity_reordered = self.connectivity[:, i]

        nelem = len(connectivity_reordered)
        nelem_to_search = nelem_to_search
        npoints = len(specfem_dmn)
        npoints_mesh = len(self.nodes)
        nearest_element_indices = element_indices
        enclosing_elem_indices = np.zeros((npoints, 8), dtype=np.int64)

        weights = np.zeros((npoints, 8))

        nfailed = lib.triLinearInterpolator(nelem, nelem_to_search, npoints, npoints_mesh,
                                  nearest_element_indices, np.ascontiguousarray(connectivity_reordered), enclosing_elem_indices,
                                  np.ascontiguousarray(self.nodes), weights, np.ascontiguousarray(points))

        if nfailed > 0:
            warning_string = '%d points were not interpolated in trilinear interpolation routine, ' \
                             'increase nelem_to_search and/or tolerances ' %nfailed
            warnings.warn(warning_string)

        if 'vsv' in specfem_dmn.components:
            specfem_dmn.df[:]['vsv'] += np.sum(self.betav[enclosing_elem_indices] * weights, axis=1) * self.step_length

        if 'vsh' in specfem_dmn.components:
            specfem_dmn.df[:]['vsh'] += np.sum(self.betah[enclosing_elem_indices] * weights, axis=1) * self.step_length

        GridData.df.update(specfem_dmn.df)



class TriLinearInterpolator:
    """
    This class handles trilinear interpolation on hexahedral element
    (translated from Salvus code)

     reference hex, with r,s,t=[-1,1]x[-1,1]x[-1,1]
   *
   *                                      (v7)                    (v6)
   *                                        /--------------d------/+
   *                                     /--|                  /-- |
   *                                 /---   |              /---    |
   *                              /--       |       f   /--        |
   *                           /--          |         --           |
   *                    (v4) +---------------b--------+ (v5)       | \gamma
   *                         |              |         |            |          ^
   *                         |              |         |            |          |
   *                         |              |       g |            |          |
   *    ^                    |              |         |            |          |
   *    | (t)                |              |         |            |          |
   *    |                    |            --+-------- |----c-------+ (v2)     |
   *    |                    |        ---/ (v1)       |         /--          ---
   *    |                    |     --/              e |     /--- /> \beta
   *    |         (s)        | ---/                   |  /--                 /-
   *    |         /-         +/--------------a--------+--                  --
   *    |     /---          (v0)                    (v3)
   *    | /---               |--------------> \alpha
   *    +-----------------> (r)

    """


    def __init__(self):
        self.mNodesR = np.array([-1, -1, +1, +1, -1, +1, +1, -1])
        self.mNodesS = np.array([-1, +1, +1, -1, -1, -1, +1, +1])
        self.mNodesT = np.array([-1, -1, -1, -1, +1, +1, +1, +1])
        self.ndim = 3
        self.nodes_per_element = 8

    def check_hull(self, pnt, vtx):
        """
        :param pnt: point to be tested
        :param vtx: hexahedral element nodes
        :return: True/False and coordinates of the reference element
        """
        reference_coordinates = self.inverse_coordinate_transform(pnt, vtx)
        return (max(abs(reference_coordinates)) <= 1. + 0.01), reference_coordinates


    def inverse_jacobian_at_point(self, pnt, vtx):
        R = pnt[0]
        S = pnt[1]
        T = pnt[2]

        Dn = np.zeros((self.ndim, self.nodes_per_element))
        for J in range(self.nodes_per_element):
            for I in range(self.ndim):
                if I == 0:
                    Dn[I, J] = self.dNdR(J, S, T)
                elif I == 1:
                    Dn[I, J] = self.dNdS(J, R, T)
                elif I == 2:
                    Dn[I, J] = self.dNdT(J, R, S)

        jac = np.dot(Dn, vtx)
        det_jac = np.linalg.det(jac)
        inv_jac = np.linalg.inv(jac)
        return det_jac, inv_jac

    def dNdR(self, N, S, T):
        return 0.125 * self.mNodesR[N] * (S * self.mNodesS[N] + 1) * (T * self.mNodesT[N] + 1)

    def dNdS(self, N, R, T):
        return 0.125 * self.mNodesS[N] * (R * self.mNodesR[N] + 1) * (T * self.mNodesT[N] + 1)

    def dNdT(self, N, R, S):
         return 0.125 * self.mNodesT[N] * (R * self.mNodesR[N] + 1) * (S * self.mNodesS[N] + 1)

    def inverse_coordinate_transform(self, pnt, vtx):
        """
        Compute the minimum distance between a point and all the vertices of the element

        :param pnt: pnt Real-space coordinates of test point
        :param vtx: vtx Matrix containing element vertices
        :return: The minimum distance to the vertices
        """
        scalexy = max(abs(vtx[1, 0] - vtx[0, 0]), abs(vtx[1, 1] - vtx[0, 1]))
        scale = max(abs(vtx[1, 2] - vtx[0, 2]), scalexy)
        tol = 1e-8 * scale

        num_iter = 0
        solution = np.array([0.0, 0.0, 0.0])
        while True:
            T = self.coordinate_transform(solution, vtx)
            objective_function = np.array([pnt[0] - T[0], pnt[1] - T[1], pnt[2] - T[2]])
            if (np.abs(objective_function) < tol).all():
                return solution

            else:
                detJ, jacobian_inverse_t = self.inverse_jacobian_at_point(solution, vtx)
                solution += np.dot(jacobian_inverse_t.T, objective_function)

            if num_iter > 10:
                raise Exception('inverseCoordinateTransform in Specfem failed to converge after 10 iterations.')
            num_iter += 1

    @staticmethod
    def reference_to_element_mapping(v0, v1, v2, v3, v4, v5, v6, v7, r, s, t):
        return v0 + 0.5 * (r + 1.0) * (-v0 + v3)  + 0.5 * (s + 1.0) * (-v0 + v1 - 0.5 * (r + 1.0) * (-v0 + v3) +
                    0.5 * (r + 1.0) * (-v1 + v2)) + 0.5 * (t + 1.0) * (-v0 + v4 - 0.5 * (r + 1.0) * (-v0 + v3) +
                    0.5 * (r + 1.0) * (-v4 + v5)  - 0.5 * (s + 1.0) * (-v0 + v1 - 0.5 * (r + 1.0) * (-v0 + v3) +
                    0.5 * (r + 1.0) * (-v1 + v2)) + 0.5 * (s + 1.0) * (-v4 + v7 - 0.5 * (r + 1.0) * (-v4 + v5) +
                    0.5 * (r + 1.0) * (v6 - v7)))

    def coordinate_transform(self, pnt, vtx):
        solution = np.zeros(3)
        solution[0] = self.reference_to_element_mapping(vtx[0, 0], vtx[1, 0], vtx[2, 0], vtx[3, 0], vtx[4, 0],
                                                        vtx[5, 0], vtx[6, 0], vtx[7, 0], pnt[0], pnt[1], pnt[2])

        solution[1] = self.reference_to_element_mapping(vtx[0, 1], vtx[1, 1], vtx[2, 1], vtx[3, 1], vtx[4, 1],
                                                        vtx[5, 1], vtx[6, 1], vtx[7, 1], pnt[0], pnt[1], pnt[2])

        solution[2] = self.reference_to_element_mapping(vtx[0, 2], vtx[1, 2], vtx[2, 2], vtx[3, 2], vtx[4, 2],
                                                        vtx[5, 2], vtx[6, 2], vtx[7, 2], pnt[0], pnt[1], pnt[2])
        return solution

    @staticmethod
    def interpolate_at_point(pnt):
        """
        :param pnt: reference element coordinates
        :return: weights of the element nodes
        """
        r = pnt[0]
        s = pnt[1]
        t = pnt[2]

        interpolator = np.zeros(8)
        interpolator[0] = -0.125 * r * s * t + 0.125 * r * s + 0.125 * r * t - \
                          0.125 * r + 0.125 * s * t - 0.125 * s - 0.125 * t + 0.125
        interpolator[1] = +0.125 * r * s * t - 0.125 * r * s + 0.125 * r * t - 0.125 * r - \
                          0.125 * s * t + 0.125 * s - 0.125 * t + 0.125
        interpolator[2] = -0.125 * r * s * t + 0.125 * r * s - 0.125 * r * t + 0.125 * r - \
                          0.125 * s * t + 0.125 * s - 0.125 * t + 0.125
        interpolator[3] = +0.125 * r * s * t - 0.125 * r * s - 0.125 * r * t + 0.125 * r + \
                          0.125 * s * t - 0.125 * s - 0.125 * t + 0.125
        interpolator[4] = +0.125 * r * s * t + 0.125 * r * s - 0.125 * r * t - 0.125 * r - \
                          0.125 * s * t - 0.125 * s + 0.125 * t + 0.125
        interpolator[5] = -0.125 * r * s * t - 0.125 * r * s + 0.125 * r * t + 0.125 * r - \
                          0.125 * s * t - 0.125 * s + 0.125 * t + 0.125
        interpolator[6] = +0.125 * r * s * t + 0.125 * r * s + 0.125 * r * t + 0.125 * r + \
                          0.125 * s * t + 0.125 * s + 0.125 * t + 0.125
        interpolator[7] = -0.125 * r * s * t - 0.125 * r * s - 0.125 * r * t - 0.125 * r + \
                          0.125 * s * t + 0.125 * s + 0.125 * t + 0.125
        return interpolator
