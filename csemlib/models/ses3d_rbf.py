import os

import sys

import h5py
from csemlib.background.grid_data import GridData
from csemlib.models.ses3d import Ses3d

import numpy as np
import scipy.spatial as spatial
from scipy.interpolate import Rbf
from scipy.interpolate import griddata


class Ses3d_rbf(Ses3d):
    """
    Class built open Ses3D which adds extra interpolation methods
    """

    def __init__(self, directory, components, interp_method='nearest_neighbour'):
        """
        Initialisation of the Ses3d_rbf class.
        :param directory: Directory where the ses3d model is located.
        :param components: Components to be extracted, e.g. 'rho', 'vsh', ... .
        :param interp_method: Interpolation method to go from the ses3d grid to the new grid points.
        """

        super(Ses3d_rbf, self).__init__(directory, components)
        self.grid_data_ses3d = None
        self.interp_method = interp_method


    def init_grid_data_hdf5(self, region=0):
        """
        Read ses3d model from an HDF5 file that was generated before with Ses3d.write_hdf5.
        :param region: Subregion index of the ses3d model.
        :return: No return. Fill self.grid_data_ses3d, the GridData structure containing the material properties of the subregion.
        """

        # Open HDF5 file containing the ses3d model.
        filename = os.path.join(self.directory, "{}.hdf5".format(self.model_info['model']))
        f = h5py.File(filename, "r")

        # Extract Cartesian x, y, z coordinates and assign them to a GridData structure.
        x = f['region_{}'.format(region)]['x'][:]
        y = f['region_{}'.format(region)]['y'][:]
        z = f['region_{}'.format(region)]['z'][:]

        self.grid_data_ses3d = GridData(x, y, z, components=self.components)

        # March through all components and assign values to the GridData structure. Include the taper as a component
        # if the taper exists.
        if self.model_info['taper']:
            components = self.components + ['taper']
        else:
            components = self.components

        for component in components:
            self.grid_data_ses3d.set_component(component, f['region_{}'.format(region)][component][:])

        f.close()


    def eval_point_cloud_griddata(self, GridData, interp_method=None):
        """
        ATTENTION: This function assumes that the ses3d model is available in the form of an HDF5 file, written
        before with Ses3d.write_hdf5.
        :param GridData: Pre-existing GridData structure that will be assigned material properties of the ses3d model.
        :param interp_method: Interpolation method to go from the ses3d block model to the grid points in GridData.
        :return: No return. GridData is manipulated internally.
        """

        print('Evaluating SES3D model:', self.model_info['model'])
        interp_method = interp_method or self.interp_method

        # Loop through all the ses3d subdomains.
        for region in range(self.model_info['region_info']['num_regions']):

            # Extract points that lie within that specific subdomain.
            ses3d_dmn = self.extract_ses3d_dmn(GridData, region)
            if len(ses3d_dmn) == 0:
                continue

            # Fill the GridData structure self.grid_data_ses3d. This is the GridData structure with the grid and the values of the ses3d model.
            self.init_grid_data_hdf5(region)

            # Get the Cartesian coordinates of the ses3d grid, for later use in interpolation.
            grid_coords = self.grid_data_ses3d.get_coordinates(coordinate_type='cartesian')

            # Generate KDTrees, needed later for interpolation.
            pnt_tree_orig = spatial.cKDTree(grid_coords, balanced_tree=False)

            # Do nearest neighbour unless specified otherwise.
            if interp_method == 'nearest_neighbour':
                self.nearest_neighbour_interpolation(pnt_tree_orig, ses3d_dmn, GridData)
            else:
                self.grid_and_rbf_interpolation(pnt_tree_orig, ses3d_dmn, interp_method, grid_coords, GridData)



    def nearest_neighbour_interpolation(self, pnt_tree_orig, ses3d_dmn, GridData):

        # Get indices of the ses3d sub-GridData structure ses3d_dmn that are nearest neighbors to the ses3d model points.
        _, indices = pnt_tree_orig.query(ses3d_dmn.get_coordinates(coordinate_type='cartesian'), k=1)

        # March through the components (material properties) of this ses3d model.
        for component in self.components:

            # Interpolation for the case where properties are perturbations to the 1D background model.
            if self.model_info['component_type'] == 'perturbation_to_1D':

                # If a taper is present, add perturbations with the taper applied to it.
                if self.model_info['taper']:
                    taper = self.grid_data_ses3d.df['taper'][indices].values
                    one_d = ses3d_dmn.df[:]['one_d_{}'.format(component)]
                    ses3d_dmn.df[:][component] = ((one_d + self.grid_data_ses3d.df[component][indices].values) * taper) + (1.0 - taper) * ses3d_dmn.df[:][component]

                # Otherwise, if there is no taper, apply the plain perturbations.
                else:
                    ses3d_dmn.df[:][component] = one_d + self.grid_data_ses3d.df[component][indices].values

            # Interpolation for the case where properties are perturbations to the 3D heterogeneous model.
            elif self.model_info['component_type'] == 'perturbation_to_3D':

                # If a taper is present, add perturbations with the taper applied to it.
                if self.model_info['taper']:
                    taper = self.grid_data_ses3d.df['taper'][indices].values
                    one_d = ses3d_dmn.df[:]['one_d_{}'.format(component)]
                    ses3d_dmn.df[:][component] = ((ses3d_dmn.df[:][component] + self.grid_data_ses3d.df[component][indices].values) * taper) + (1.0 - taper) * ses3d_dmn.df[:][component]

                # Otherwise, if there is no taper, apply the plain perturbations.
                else:
                    ses3d_dmn.df[:][component] += self.grid_data_ses3d.df[component][indices].values

            # Interpolation for the case where properties are absolute values.
            elif self.model_info['component_type'] == 'absolute':
                if self.model_info['taper']:
                    taper = self.grid_data_ses3d.df['taper'][indices].values
                    ses3d_dmn.df[:][component] = taper * self.grid_data_ses3d.df[component][indices].values + (1 - taper) * ses3d_dmn.df[:][component]
                else:
                    ses3d_dmn.df[:][component] = self.grid_data_ses3d.df[component][indices].values

            # No valid component_type.
            else:
                print 'No valid component_type. Must be perturbation_to_1D, perturbation_to_3D or absolute'

        # Update that master GridData structure.
        GridData.df.update(ses3d_dmn.df)




    def grid_and_rbf_interpolation(self, pnt_tree_orig, ses3d_dmn, interp_method, grid_coords, GridData):
        # Use 30 nearest points
        _, all_neighbours = pnt_tree_orig.query(ses3d_dmn.get_coordinates(coordinate_type='cartesian'), k=100)

        # Interpolate ses3d value for each grid point
        i = 0
        if self.model_info['taper']:
            components = ['taper'] + self.components
            ses3d_dmn.set_component('taper', np.zeros(len(ses3d_dmn)))
        else:
            components = self.components

        for neighbours in all_neighbours:
            x_c_orig, y_c_orig, z_c_orig = grid_coords[neighbours].T
            for component in components:
                dat_orig = self.grid_data_ses3d.df[component][neighbours].values
                coords_new = ses3d_dmn.get_coordinates(coordinate_type='cartesian').T
                x_c_new, y_c_new, z_c_new = coords_new.T[i]

                if interp_method == 'griddata_linear':
                    pts_local = np.array((x_c_orig, y_c_orig, z_c_orig)).T
                    xi = np.array((x_c_new, y_c_new, z_c_new))
                    if self.model_info['component_type'] == 'absolute':
                        val = griddata(pts_local, dat_orig, xi, method='linear',
                                       fill_value=ses3d_dmn.df[component].values[i])
                    elif self.model_info['component_type'] == 'perturbation':
                        val = griddata(pts_local, dat_orig, xi, method='linear', fill_value=0.0)

                elif interp_method == 'radial_basis_func':
                    rbfi = Rbf(x_c_orig, y_c_orig, z_c_orig, dat_orig, function='linear')
                    val = rbfi(x_c_new, y_c_new, z_c_new)

                if self.model_info['component_type'] == 'perturbation':
                    if self.model_info['taper'] and component != 'taper':
                        taper = ses3d_dmn.df['taper'].values[i]
                        one_d = ses3d_dmn.df['one_d_{}'.format(component)].values[i]
                        ses3d_dmn.df[component].values[i] += (taper * val)
                        ses3d_dmn.df[component].values[i] = (one_d + val) * taper + \
                                                            (1 - taper) * ses3d_dmn.df[component].values[i]
                    else:
                        ses3d_dmn.df[component].values[i] += val
                elif self.model_info['component_type'] == 'absolute':
                    if self.model_info['taper'] and component != 'taper':
                        taper = ses3d_dmn.df['taper'].values[i]
                        ses3d_dmn.df[component].values[i] = taper * val + (1-taper) * ses3d_dmn.df[component].values[i]
                    else:
                        ses3d_dmn.df[component].values[i] = val
            i += 1

            if i % 100 == 0:
                ind = float(i)
                percent = ind / len(all_neighbours) * 100.0
                sys.stdout.write("\rProgress: %.1f%% " % percent)
                sys.stdout.flush()
        sys.stdout.write("\r")
        if self.model_info['taper']:
            del ses3d_dmn.df['taper']

        GridData.df.update(ses3d_dmn.df)
