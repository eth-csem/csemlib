import datetime
import io
import yaml
import xarray
import os
import sys
import h5py
from csemlib.background.grid_data import GridData

import numpy as np
import scipy.spatial as spatial
from scipy.interpolate import Rbf
from scipy.interpolate import griddata

from ..utils import sph2cart, rotate, get_rot_matrix

#=======================================================================================================================
# Little helper functions.
#=======================================================================================================================

def _read_multi_region_file(data):
    """
    Transform ses3d block_* files into a list of number with the starting points and lengths of the subdomains.
    :param data: Array containing the numbers listed in a block file.
    :return: List with numbers of starting indices and lengths of the ses3d subdomains.
    """

    regions = []
    num_region, region_start = None, None
    num_regions = int(data[0])
    for i in range(num_regions):
        region_start = int(region_start + num_region + 1) if region_start else int(2)
        num_region = int(data[region_start - 1]) if num_region else int(data[1])
        regions.append(data[region_start:region_start + num_region])
    return regions

#=======================================================================================================================
# SES3D object.
#=======================================================================================================================


class Ses3d(object):
    """
    Class handling file-IO for a model in SES3D format.
    """

    # Initialisation. ==================================================================================================

    def __init__(self, directory, components, interp_method='nearest_neighbour'):
        super(Ses3d, self).__init__()
        self.rot_mat = None
        self._disc = []
        self._data = []
        self.directory = directory

        self.grid_data_ses3d = None
        self.interp_method = interp_method

        # Read yaml file containing information on the ses3d submodel.
        with io.open(os.path.join(self.directory, 'modelinfo.yml'), 'rt') as fh:
            try:
                self.model_info = yaml.load(fh)
            except yaml.YAMLError as exc:
                print(exc)

        # Assign components (e.g. vs, vp, ...), geometric setup and rotation of the submodel.
        self.components = list(set(self.model_info['components']).intersection(components))
        self.geometry = self.model_info['geometry']
        self.rot_vec = np.array([self.geometry['rot_x'], self.geometry['rot_y'], self.geometry['rot_z']])
        self.rot_angle = self.geometry['rot_angle']


    # Read original ses3d model. =======================================================================================

    def read(self):
        """
        Read ses3d model in the original definition with block files etc.
        :return: No return values. Fills self._data with an xarray containing the model.
        """
        files = set(os.listdir(self.directory))
        if self.components:
            if not set(self.components).issubset(files):
                raise IOError('Model directory does not have all components ' + ', '.join(self.components))

        # Read the block_* files containing the coordinate lines. Make lists of indices characterising the subdomains.
        with io.open(os.path.join(self.directory, 'block_x'), 'rt') as fh:
            data = np.asarray(fh.readlines(), dtype=float)
            col_regions = _read_multi_region_file(data)
        with io.open(os.path.join(self.directory, 'block_y'), 'rt') as fh:
            data = np.asarray(fh.readlines(), dtype=float)
            lon_regions = _read_multi_region_file(data)
        with io.open(os.path.join(self.directory, 'block_z'), 'rt') as fh:
            data = np.asanyarray(fh.readlines(), dtype=float)
            rad_regions = _read_multi_region_file(data)

        # Get centers of boxes.
        for i, _ in enumerate(col_regions):
            discretizations = {
                    'col': (col_regions[i][1] - col_regions[i][0]) / 2.0,
                    'lon': (lon_regions[i][1] - lon_regions[i][0]) / 2.0,
                    'rad': (rad_regions[i][1] - rad_regions[i][0]) / 2.0}
            self._disc.append(discretizations)
            col_regions[i] = 0.5 * (col_regions[i][1:] + col_regions[i][:-1])
            lon_regions[i] = 0.5 * (lon_regions[i][1:] + lon_regions[i][:-1])
            rad_regions[i] = 0.5 * (rad_regions[i][1:] + rad_regions[i][:-1])

        # If a taper is present, add it to the components of the ses3d model.
        if self.model_info['taper']:
            components = self.components + ['taper']
        else:
            components = self.components

        # Walk through the components and read their values.
        for p in components:

            with io.open(os.path.join(self.directory, p), 'rt') as fh:
                data = np.asarray(fh.readlines(), dtype=float)
                val_regions = _read_multi_region_file(data)

            for i, _ in enumerate(val_regions):
                val_regions[i] = val_regions[i].reshape((len(col_regions[i]), len(lon_regions[i]), len(rad_regions[i])), order='C')
                if not self._data:
                    self._data = [xarray.Dataset() for j in range(len(val_regions))]
                self._data[i][p] = (('col', 'lon', 'rad'), val_regions[i])
                if 'rho' in p:
                    self._data[i][p].attrs['units'] = 'g/cm3'
                else:
                    self._data[i][p].attrs['units'] = 'km/s'

        # Add coordinates.
        for i, _ in enumerate(val_regions):
            s_col, s_lon, s_rad = len(col_regions[i]), len(lon_regions[i]), len(rad_regions[i])
            self._data[i].coords['col'] = np.radians(col_regions[i])
            self._data[i].coords['lon'] = np.radians(lon_regions[i])
            self._data[i].coords['rad'] = rad_regions[i]

            cols, lons, rads = np.meshgrid(self._data[i].coords['col'].values,
                                           self._data[i].coords['lon'].values,
                                           self._data[i].coords['rad'].values, indexing='ij')

            # Cartesian coordinates and rotation.
            x, y, z = sph2cart(cols.ravel(), lons.ravel(), rads.ravel())
            if self.model_info['geometry']['rotation']:
                if len(self.rot_vec) is not 3:
                    raise ValueError("Rotation matrix must be a 3-vector.")
                self.rot_mat = get_rot_matrix(np.radians(self.geometry['rot_angle']), *self.rot_vec)
                x, y, z = rotate(x, y, z, self.rot_mat)

            self._data[i]['x'] = (('col', 'lon', 'rad'), x.reshape((s_col, s_lon, s_rad), order='C'))
            self._data[i]['y'] = (('col', 'lon', 'rad'), y.reshape((s_col, s_lon, s_rad), order='C'))
            self._data[i]['z'] = (('col', 'lon', 'rad'), z.reshape((s_col, s_lon, s_rad), order='C'))

            # Add units.
            self._data[i].coords['col'].attrs['units'] = 'radians'
            self._data[i].coords['lon'].attrs['units'] = 'radians'
            self._data[i].coords['rad'].attrs['units'] = 'km'

            # Add Ses3d attributes.
            self._data[i].attrs['solver'] = 'ses3d'
            self._data[i].attrs['coordinate_system'] = 'spherical'
            self._data[i].attrs['date'] = datetime.datetime.now().__str__()


    # Write original ses3d model to hdf5 file. =========================================================================
    def write_to_hdf5(self, filename=None):
        self.read()
        filename = filename or os.path.join(self.directory, "{}.hdf5".format(self.model_info['model']))
        f = h5py.File(filename, "w")

        parameters = ['x', 'y', 'z'] + self.model_info['components']
        if self.model_info['taper']:
            parameters += ['taper']

        for region in range(self.model_info['region_info']['num_regions']):
            region_grp = f.create_group('region_{}'.format(region))
            for param in parameters:
                region_grp.create_dataset(param, data=self.data(region)[param].values.ravel(), dtype='d')
        f.close()


    # Read ses3d model from hdf5 into a GridData structure. ============================================================

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


    # Ascribe material properties to GridData points. ==================================================================

    def eval_point_cloud_griddata(self, GridData, interp_method=None):
        """
        Ascribe material properties to the those GridData points that fall into the ses3d domain.
        ATTENTION: This function assumes that the ses3d model is available in the form of an HDF5 file, written
        before with Ses3d.write_hdf5.
        :param GridData: Pre-existing GridData structure that will be assigned material properties of the ses3d model.
        :param interp_method: Interpolation method to go from the ses3d block model to the grid points in GridData.
        :return: No return. GridData is manipulated internally.
        """

        interp_method = interp_method or self.interp_method

        # Loop through all the ses3d subdomains.
        for region in range(self.model_info['region_info']['num_regions']):

            # Extract points that lie within that specific subdomain.
            ses3d_dmn = self.extract_ses3d_dmn(GridData, region)
            if len(ses3d_dmn) == 0:
                continue

            if region == 0:
                print('Evaluating SES3D model: {}'.format(self.model_info['model']))

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

    # Extract grid points within ses3d domain. =========================================================================

    def extract_ses3d_dmn(self, GridData, region=0):
        """
        Extract those points from the current collection of grid points that fall inside a ses3d subdomain.
        :param GridData: GridData structure with collection of current grid points and their properties.
        :param region: Index of the ses3d subregion, starting with 0.
        :return: Subset of GridData that falls into that specific ses3d subdomain.
        """

        # Transscribe geometric info and make deep copy of the current data structure.
        geometry = self.model_info['geometry']
        ses3d_dmn = GridData.copy()

        # move points near boundary into region above if multiregion model
        region_info = self.model_info['region_info']
        num_regions = region_info['num_regions']

        # if the ses3d model has multiple regions, slightly shift near boundaries to the upper region.
        if num_regions > 1:
            eps = 0.05
            for reg in range(num_regions)[1:]:
                top = region_info['region_{}_top'.format(reg)]
                relative_shift = (top + 2 * eps) / top
                nodes_to_be_shifted = (np.abs(ses3d_dmn.df['r'] - top) < eps)
                ses3d_dmn.df['r'][nodes_to_be_shifted] *= relative_shift
                ses3d_dmn.df['x'][nodes_to_be_shifted] *= relative_shift
                ses3d_dmn.df['y'][nodes_to_be_shifted] *= relative_shift
                ses3d_dmn.df['z'][nodes_to_be_shifted] *= relative_shift

        # Rotate all current points into the rotated ses3d coordinate system, in case there is a rotation.
        if geometry['rotation'] is True:
            ses3d_dmn.rotate(-np.radians(geometry['rot_angle']), geometry['rot_x'], geometry['rot_y'], geometry['rot_z'])
        # Extract points from the current data structure in colatitude direction.
        ses3d_dmn.df = ses3d_dmn.df[ses3d_dmn.df['c'] >= np.deg2rad(geometry['cmin'])]
        ses3d_dmn.df = ses3d_dmn.df[ses3d_dmn.df['c'] <= np.deg2rad(geometry['cmax'])]

        # Extract points from the current data structure in longitude direction.
        l_min = geometry['lmin']
        l_max = geometry['lmax']

        if l_min > 180.0:
            l_min -= 360.0

        if l_max > 180.0:
            l_max -= 360.0

        if l_max >= l_min:
            ses3d_dmn.df = ses3d_dmn.df[(ses3d_dmn.df["l"] >= np.deg2rad(l_min)) & (ses3d_dmn.df["l"] <= np.deg2rad(l_max))]
        elif l_max < l_min:
            ses3d_dmn.df = ses3d_dmn.df[(ses3d_dmn.df["l"] <= np.deg2rad(l_max)) | (ses3d_dmn.df["l"] >= np.deg2rad(l_min))]

        # Extract points from the current data structure in radial direction.
        bottom = 'region_{}_bottom'.format(region)
        top = 'region_{}_top'.format(region)

        if region_info[top] >= 6371.0:
            tolerance = 0.1   # A small tolerance in km to make sure no points are missed near Earth's surface.
        else:
            tolerance = 0.0

        ses3d_dmn.df = ses3d_dmn.df[ses3d_dmn.df['r'] > region_info[bottom]]
        ses3d_dmn.df = ses3d_dmn.df[ses3d_dmn.df['r'] <= region_info[top] + tolerance]

        # get radial layers for each region
        with io.open(os.path.join(self.directory, 'block_z'), 'rt') as fh:
            data = np.asanyarray(fh.readlines(), dtype=float)
            rad_regions = _read_multi_region_file(data)

        # slightly shift nodes at layer boundaries such that spherical slices through a ses3d layer boundary look clean
        eps = 0.05
        for rad in rad_regions[region]:
            relative_shift = (rad + eps) / rad
            nodes_to_be_shifted = (np.abs(ses3d_dmn.df['r'] - rad) < eps)
            ses3d_dmn.df['r'][nodes_to_be_shifted] *= relative_shift
            ses3d_dmn.df['x'][nodes_to_be_shifted] *= relative_shift
            ses3d_dmn.df['y'][nodes_to_be_shifted] *= relative_shift
            ses3d_dmn.df['z'][nodes_to_be_shifted] *= relative_shift

        # Rotate back to the actual physical domain.
        if geometry['rotation'] is True:
            ses3d_dmn.rotate(np.radians(geometry['rot_angle']), geometry['rot_x'], geometry['rot_y'], geometry['rot_z'])
        return ses3d_dmn

    def data(self, region=0):
        return self._data[region]

    # Nearest-neighbor interpolation. ==================================================================================
    def nearest_neighbour_interpolation(self, pnt_tree_orig, ses3d_dmn, GridData):
        """
        Implement nearest-neighbor interpolation.
        :param pnt_tree_orig: KDTree of the grid coordinates in the ses3d model.
        :param ses3d_dmn: Subset of the GridData structure that falls into the ses3d domain.
        :param GridData: Master GridData structure.
        :return: No return. GridData is updated internally.
        """

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
                    one_d = ses3d_dmn.df[:]['one_d_{}'.format(component)]
                    ses3d_dmn.df[:][component] = one_d + self.grid_data_ses3d.df[component][indices].values

            # Interpolation for the case where properties are perturbations to the 3D heterogeneous model.
            elif self.model_info['component_type'] == 'perturbation_to_3D':

                # If a taper is present, add perturbations with the taper applied to it.
                if self.model_info['taper']:
                    taper = self.grid_data_ses3d.df['taper'][indices].values
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
                print('No valid component_type. Must be perturbation_to_1D, perturbation_to_3D or absolute')

        # Update that master GridData structure.
        GridData.df.update(ses3d_dmn.df)


    # Spline interpolation. Not yet fully functional. ==================================================================

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
