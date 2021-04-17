import datetime
import io
import yaml
import xarray
import os
from tqdm import tqdm
import h5py
from csemlib.background.grid_data import GridData

import numpy as np
import scipy.spatial as spatial

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

    def __init__(self, directory, components, interp_method='trilinear'):
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
                self.model_info = yaml.load(fh, Loader=yaml.FullLoader)
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
            grid_coords = self.grid_data_ses3d.get_coordinates(
                coordinate_type='cartesian')

            if self.interp_method == "trilinear":
                self.trilinear_interpolation_parallel(ses3d_dmn, GridData, region)

            elif interp_method == 'nearest_neighbour':
                # Generate KDTrees, needed later for interpolation.
                pnt_tree_orig = spatial.cKDTree(grid_coords,
                                                balanced_tree=False)
                self.nearest_neighbour_interpolation(pnt_tree_orig, ses3d_dmn, GridData)
            else:
                raise Exception(f"Interpolation method {interp_method} is not "
                                f"implemented.")

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
            tolerance = 0.5   # A small tolerance in km to make sure no points are missed near Earth's surface.
        else:
            tolerance = 0.0

        # get radial layers for each region
        with io.open(os.path.join(self.directory, 'block_z'), 'rt') as fh:
            data = np.asanyarray(fh.readlines(), dtype=float)
            rad_regions = _read_multi_region_file(data)

        # slightly shift nodes at layer boundaries such that spherical slices through a ses3d layer boundary look clean
        eps = 0.1
        for rad in rad_regions[region]:
            relative_shift = (rad + 2*eps) / rad
            nodes_to_be_shifted = (np.abs(ses3d_dmn.df['r'] - rad) < eps)
            ses3d_dmn.df['r'][nodes_to_be_shifted] *= relative_shift
            ses3d_dmn.df['x'][nodes_to_be_shifted] *= relative_shift
            ses3d_dmn.df['y'][nodes_to_be_shifted] *= relative_shift
            ses3d_dmn.df['z'][nodes_to_be_shifted] *= relative_shift

        ses3d_dmn.df = ses3d_dmn.df[ses3d_dmn.df['r'] > region_info[bottom]]
        ses3d_dmn.df = ses3d_dmn.df[ses3d_dmn.df['r'] <= region_info[top] + tolerance]

        # Rotate back to the actual physical domain.
        if geometry['rotation'] is True and self.interp_method != "trilinear":
            ses3d_dmn.rotate(np.radians(geometry['rot_angle']), geometry['rot_x'], geometry['rot_y'], geometry['rot_z'])
        return ses3d_dmn

    def data(self, region=0):
        return self._data[region]

    def trilinear_interpolation(self, ses3d_dmn, GridData, region):
        """
        Essentially what is done here is loop through the points in the ses3d_dmn,
        find the nearest indices and perform a trilinear interpolation.

        :param ses3d_dmn: Subset of the GridData structure that falls into the ses3d domain
        :param GridData: Master GridData structure
        :return:
        """

        # Read the block_* files containing the coordinate lines. Make lists of indices characterising the subdomains.
        with io.open(os.path.join(self.directory, 'block_x'), 'rt') as fh:
            data = np.asarray(fh.readlines(), dtype=float)
            unique_colats = _read_multi_region_file(data)
        with io.open(os.path.join(self.directory, 'block_y'), 'rt') as fh:
            data = np.asarray(fh.readlines(), dtype=float)
            unique_lons = _read_multi_region_file(data)
        with io.open(os.path.join(self.directory, 'block_z'), 'rt') as fh:
            data = np.asanyarray(fh.readlines(), dtype=float)
            unique_rads = _read_multi_region_file(data)

        # Get centers of boxes.
        for i, _ in enumerate(unique_colats):
            unique_colats[i] = 0.5 * (unique_colats[i][1:] + unique_colats[i][:-1])
            unique_lons[i] = 0.5 * (unique_lons[i][1:] + unique_lons[i][:-1])
            unique_rads[i] = 0.5 * (unique_rads[i][1:] + unique_rads[i][:-1])

        unique_colats_region = unique_colats[region]
        unique_lons_region = unique_lons[region]
        unique_rads_region = unique_rads[region]

        num_colats = len(unique_colats_region)
        num_lons = len(unique_lons_region)
        num_depths = len(unique_rads_region)
        tolerance = 0.0001

        print("Performing trilinear interpolation for SES3D model... This can be a little slow")
        for idx in range(len(ses3d_dmn.df["r"])):
            if idx % 5000 == 0:
                print(np.round(idx/len(ses3d_dmn.df["r"])*100.0, 2), "%", end="\r", flush=True)
            colat = np.rad2deg(ses3d_dmn.df['c'].values[idx])
            lon = np.rad2deg(ses3d_dmn.df['l'].values[idx])
            rad = ses3d_dmn.df['r'].values[idx]

            if rad > np.max(unique_rads_region):
                rad = np.max(unique_rads_region)

            if rad <= np.min(unique_rads_region):
                rad = np.min(unique_rads_region) + 0.01 # added this line, seems to skip the bottom otherwise
                # continue

            # simply skip edges
            if colat > np.max(unique_colats_region) or colat <= np.min(unique_colats_region):
                continue

            # handle edge case australia (quick fix)
            if np.max(unique_lons_region) > 180.0 and lon < 0.0:
                lon += 360

            if lon > np.max(unique_lons_region) or lon <= np.min(unique_lons_region):
                continue

            # Find surrounding vertices
            colat_max = np.where(unique_colats_region - colat >= 0.0)[0][0]
            colat_min = np.where(unique_colats_region - colat < 0.0)[0][-1]

            lon_max = np.where(unique_lons_region - lon >= 0.0)[0][0]
            lon_min = np.where(unique_lons_region - lon < 0.0)[0][-1]

            rmin = np.where(unique_rads_region - rad < 0.0)[0][-1]
            rmax = np.where(unique_rads_region - rad >= 0.0)[0][0]

            max_colat = unique_colats_region[colat_max]
            min_colat = unique_colats_region[colat_min]

            max_lon = unique_lons_region[lon_max]
            min_lon = unique_lons_region[lon_min]

            min_dep = unique_rads_region[rmin]  # top
            max_dep = unique_rads_region[rmax]  # bottom

            # bi-linear interpolation bottom
            # compute row position in arrays
            # min colat, min lon
            row_idx_min_min = colat_min * (
                num_depths * num_lons) + lon_min * num_depths + rmin
            # max colat, min_lon
            row_idx_max_min = colat_max * (
                num_depths * num_lons) + lon_min * num_depths + rmin
            # min colat, max lon
            row_idx_min_max = colat_min * (
                num_depths * num_lons) + lon_max * num_depths + rmin
            # max colat, max lon
            row_idx_max_max = colat_max* (
                num_depths * num_lons) + lon_max * num_depths + rmin

            if self.model_info['taper']:
                component = "taper"
                Q11 = self.grid_data_ses3d.df[component].values[
                    row_idx_min_min]
                Q12 = self.grid_data_ses3d.df[component].values[
                    row_idx_max_min]
                Q21 = self.grid_data_ses3d.df[component].values[
                    row_idx_min_max]
                Q22 = self.grid_data_ses3d.df[component].values[
                    row_idx_max_max]

                r = np.array([[max_colat - colat], [colat - min_colat]])
                m = np.array([[Q11, Q12], [Q21, Q22]])
                l = np.array([max_lon - lon, lon - min_lon])
                val_rmin = l @ m @ r / (
                (max_lon - min_lon) * (max_colat - min_colat))

                # bi-linear interpolation top
                # compute row position in arrays
                # min colat, min lon
                row_idx_min_min = colat_min * (
                    num_depths * num_lons) + lon_min * num_depths + rmax
                # max colat, min_lon
                row_idx_max_min = colat_max * (
                    num_depths * num_lons) + lon_min * num_depths + rmax
                # min colat, max lon
                row_idx_min_max = colat_min * (
                    num_depths * num_lons) + lon_max * num_depths + rmax
                # max colat, max lon
                row_idx_max_max = colat_max * (
                    num_depths * num_lons) + lon_max * num_depths + rmax

                Q11 = self.grid_data_ses3d.df[component].values[
                    row_idx_min_min]
                Q12 = self.grid_data_ses3d.df[component].values[
                    row_idx_max_min]
                Q21 = self.grid_data_ses3d.df[component].values[
                    row_idx_min_max]
                Q22 = self.grid_data_ses3d.df[component].values[
                    row_idx_max_max]

                m = np.array([[Q11, Q12], [Q21, Q22]])
                val_rmax = l @ m @ r / (
                    (max_lon - min_lon) * (max_colat - min_colat))

                # linear interpolation top and bottom
                taper = (val_rmax * (min_dep - rad) + val_rmin * (
                    rad - max_dep)) / (min_dep - max_dep)
            else:
                taper = 1.0

            for component in self.components:
                Q11 = self.grid_data_ses3d.df[component].values[row_idx_min_min]
                Q12 = self.grid_data_ses3d.df[component].values[row_idx_max_min]
                Q21 = self.grid_data_ses3d.df[component].values[row_idx_min_max]
                Q22 = self.grid_data_ses3d.df[component].values[row_idx_max_max]

                r = np.array([[max_colat - colat], [colat - min_colat]])
                m = np.array([[Q11, Q12], [Q21, Q22]])
                l = np.array([max_lon - lon, lon - min_lon])
                val_rmin = l @ m @ r / ((max_lon - min_lon) * (max_colat - min_colat))

                # bi-linear interpolation top
                # compute row position in arrays
                # min colat, min lon
                row_idx_min_min = colat_min * (
                    num_depths * num_lons) + lon_min * num_depths + rmax
                # max colat, min_lon
                row_idx_max_min = colat_max * (
                    num_depths * num_lons) + lon_min * num_depths + rmax
                # min colat, max lon
                row_idx_min_max = colat_min * (
                    num_depths * num_lons) + lon_max * num_depths + rmax
                # max colat, max lon
                row_idx_max_max = colat_max * (
                    num_depths * num_lons) + lon_max * num_depths + rmax

                Q11 = self.grid_data_ses3d.df[component].values[row_idx_min_min]
                Q12 = self.grid_data_ses3d.df[component].values[row_idx_max_min]
                Q21 = self.grid_data_ses3d.df[component].values[row_idx_min_max]
                Q22 = self.grid_data_ses3d.df[component].values[row_idx_max_max]

                m = np.array([[Q11, Q12], [Q21, Q22]])
                val_rmax = l @ m @ r / (
                    (max_lon - min_lon) * (max_colat - min_colat))

                # linear interpolation top and bottom
                val = (val_rmax * (min_dep - rad) + val_rmin * (rad - max_dep)) / (min_dep - max_dep)

                # ses3d_dmn.df["vsv"].values[idx] = np.rad2deg(val) / 1000.0
                if self.model_info['component_type'] == 'perturbation_to_1D':
                    one_d = ses3d_dmn.df['one_d_{}'.format(component)].values[idx]
                    ses3d_dmn.df[component].values[idx] = ((one_d + val) * taper) + (1.0 - taper) * ses3d_dmn.df[component].values[idx]
                elif self.model_info['component_type'] == 'perturbation_to_3D':
                    ses3d_dmn.df[component].values[idx] = (ses3d_dmn.df[component].values[idx] + val) * taper + (1.0 - taper) * ses3d_dmn.df[component].values[idx]
                elif self.model_info['component_type'] == 'absolute':
                    ses3d_dmn.df[component].values[idx] = taper * val + (1.0 - taper) * ses3d_dmn.df[component].values[idx]
                else:
                    print(
                        'No valid component_type. Must be perturbation_to_1D, perturbation_to_3D or absolute')

        # Rotate back to the actual physical domain.
        geometry = self.model_info['geometry']
        if geometry['rotation'] is True and self.interp_method == "trilinear":
            ses3d_dmn.rotate(np.radians(geometry['rot_angle']), geometry['rot_x'], geometry['rot_y'], geometry['rot_z'])

        # Finally update GridData object
        GridData.df.update(ses3d_dmn.df)


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
                    ses3d_dmn.df[:][component] = taper * self.grid_data_ses3d.df[component][indices].values + (1.0 - taper) * ses3d_dmn.df[:][component]
                else:
                    ses3d_dmn.df[:][component] = self.grid_data_ses3d.df[component][indices].values

            # No valid component_type.
            else:
                print('No valid component_type. Must be perturbation_to_1D, perturbation_to_3D or absolute')

        # Update that master GridData structure.
        GridData.df.update(ses3d_dmn.df)

    def trilinear_interpolation_parallel(self, ses3d_dmn, GridData, region):
        """
        Essentially what is done here is loop through the points in the ses3d_dmn,
        find the nearest indices and perform a trilinear interpolation.

        :param ses3d_dmn: Subset of the GridData structure that falls into the ses3d domain
        :param GridData: Master GridData structure
        :return:
        """

        # Read the block_* files containing the coordinate lines. Make lists of indices characterising the subdomains.
        with io.open(os.path.join(self.directory, 'block_x'), 'rt') as fh:
            data = np.asarray(fh.readlines(), dtype=float)
            unique_colats = _read_multi_region_file(data)
        with io.open(os.path.join(self.directory, 'block_y'), 'rt') as fh:
            data = np.asarray(fh.readlines(), dtype=float)
            unique_lons = _read_multi_region_file(data)
        with io.open(os.path.join(self.directory, 'block_z'), 'rt') as fh:
            data = np.asanyarray(fh.readlines(), dtype=float)
            unique_rads = _read_multi_region_file(data)

        # Get centers of boxes.
        for i, _ in enumerate(unique_colats):
            unique_colats[i] = 0.5 * (unique_colats[i][1:] + unique_colats[i][:-1])
            unique_lons[i] = 0.5 * (unique_lons[i][1:] + unique_lons[i][:-1])
            unique_rads[i] = 0.5 * (unique_rads[i][1:] + unique_rads[i][:-1])

        unique_colats_region = unique_colats[region]
        unique_lons_region = unique_lons[region]
        unique_rads_region = unique_rads[region]

        num_lons = len(unique_lons_region)
        num_depths = len(unique_rads_region)

        print("Performing trilinear interpolation for SES3D model...")

        global _process

        def _process(point_indices):
            def set_val(idx):

                # fill valls array with current values
                vals = np.zeros(len(self.components))
                for _i, component in enumerate(self.components):
                    vals[_i] = ses3d_dmn.df[component].values[idx]

                colat = np.rad2deg(ses3d_dmn.df['c'].values[idx])
                lon = np.rad2deg(ses3d_dmn.df['l'].values[idx])
                rad = ses3d_dmn.df['r'].values[idx]

                if rad > np.max(unique_rads_region):
                    rad = np.max(unique_rads_region)

                if rad <= np.min(unique_rads_region):
                    rad = np.min(unique_rads_region) + 0.01
                    # added the above line after issue at 50, 200 km with europe
                    # return vals

                # simply skip edges
                if colat > np.max(unique_colats_region) or\
                        colat <= np.min(unique_colats_region):
                    return vals

                # handle edge case australia (quick fix)
                if np.max(unique_lons_region) > 180.0 and lon < 0.0:
                    lon += 360

                if lon > np.max(unique_lons_region) or\
                        lon <= np.min(unique_lons_region):
                    return vals

                # Find surrounding vertices
                colat_max = np.where(unique_colats_region - colat >= 0.0)[0][0]
                colat_min = np.where(unique_colats_region - colat < 0.0)[0][-1]

                lon_max = np.where(unique_lons_region - lon >= 0.0)[0][0]
                lon_min = np.where(unique_lons_region - lon < 0.0)[0][-1]

                rmin = np.where(unique_rads_region - rad < 0.0)[0][-1]
                rmax = np.where(unique_rads_region - rad >= 0.0)[0][0]

                max_colat = unique_colats_region[colat_max]
                min_colat = unique_colats_region[colat_min]

                max_lon = unique_lons_region[lon_max]
                min_lon = unique_lons_region[lon_min]

                min_dep = unique_rads_region[rmin]  # top
                max_dep = unique_rads_region[rmax]  # bottom

                r = np.array([[max_colat - colat], [colat - min_colat]])
                l = np.array([max_lon - lon, lon - min_lon])

                # bi-linear interpolation bottom
                # compute row position bottom in arrays
                # min colat, min lon
                row_idx_min_min_min = colat_min * (
                    num_depths * num_lons) + lon_min * num_depths + rmin
                # max colat, min_lon
                row_idx_max_min_min = colat_max * (
                    num_depths * num_lons) + lon_min * num_depths + rmin
                # min colat, max lon
                row_idx_min_max_min = colat_min * (
                    num_depths * num_lons) + lon_max * num_depths + rmin
                # max colat, max lon
                row_idx_max_max_min = colat_max* (
                    num_depths * num_lons) + lon_max * num_depths + rmin

                # bi-linear interpolation top
                # compute row position top in arrays
                # min colat, min lon
                row_idx_min_min_max = colat_min * (
                        num_depths * num_lons) + lon_min * num_depths + rmax
                # max colat, min_lon
                row_idx_max_min_max = colat_max * (
                        num_depths * num_lons) + lon_min * num_depths + rmax
                # min colat, max lon
                row_idx_min_max_max = colat_min * (
                        num_depths * num_lons) + lon_max * num_depths + rmax
                # max colat, max lon
                row_idx_max_max_max = colat_max * (
                        num_depths * num_lons) + lon_max * num_depths + rmax

                # If taper, compute interpolate the taper value too.
                if self.model_info['taper']:
                    component = "taper"
                    Q11 = self.grid_data_ses3d.df[component].values[
                        row_idx_min_min_min]
                    Q12 = self.grid_data_ses3d.df[component].values[
                        row_idx_max_min_min]
                    Q21 = self.grid_data_ses3d.df[component].values[
                        row_idx_min_max_min]
                    Q22 = self.grid_data_ses3d.df[component].values[
                        row_idx_max_max_min]

                    m = np.array([[Q11, Q12], [Q21, Q22]])
                    val_rmin = l @ m @ r / (
                    (max_lon - min_lon) * (max_colat - min_colat))

                    Q11 = self.grid_data_ses3d.df[component].values[
                        row_idx_min_min_max]
                    Q12 = self.grid_data_ses3d.df[component].values[
                        row_idx_max_min_max]
                    Q21 = self.grid_data_ses3d.df[component].values[
                        row_idx_min_max_max]
                    Q22 = self.grid_data_ses3d.df[component].values[
                        row_idx_max_max_max]

                    m = np.array([[Q11, Q12], [Q21, Q22]])
                    val_rmax = l @ m @ r / (
                        (max_lon - min_lon) * (max_colat - min_colat))

                    # linear interpolation top and bottom
                    taper = (val_rmax * (min_dep - rad) + val_rmin * (
                        rad - max_dep)) / (min_dep - max_dep)
                else:
                    taper = 1.0

                for _i, component in enumerate(self.components):
                    # Bottom interpolation
                    Q11 = self.grid_data_ses3d.df[component].values[row_idx_min_min_min]
                    Q12 = self.grid_data_ses3d.df[component].values[row_idx_max_min_min]
                    Q21 = self.grid_data_ses3d.df[component].values[row_idx_min_max_min]
                    Q22 = self.grid_data_ses3d.df[component].values[row_idx_max_max_min]
                    m = np.array([[Q11, Q12], [Q21, Q22]])
                    val_rmin = l @ m @ r / ((max_lon - min_lon) * (max_colat - min_colat))

                    # Top interpolation
                    Q11 = self.grid_data_ses3d.df[component].values[row_idx_min_min_max]
                    Q12 = self.grid_data_ses3d.df[component].values[row_idx_max_min_max]
                    Q21 = self.grid_data_ses3d.df[component].values[row_idx_min_max_max]
                    Q22 = self.grid_data_ses3d.df[component].values[row_idx_max_max_max]
                    m = np.array([[Q11, Q12], [Q21, Q22]])
                    val_rmax = l @ m @ r / (
                        (max_lon - min_lon) * (max_colat - min_colat))

                    # linear interpolation between top and bottom
                    val = (val_rmax * (min_dep - rad) + val_rmin * (rad - max_dep)) / (min_dep - max_dep)

                    if self.model_info['component_type'] == 'perturbation_to_1D':
                        one_d = ses3d_dmn.df['one_d_{}'.format(component)].values[idx]
                        vals[_i]  = ((one_d + val) * taper) + (1.0 - taper) * ses3d_dmn.df[component].values[idx]
                    elif self.model_info['component_type'] == 'perturbation_to_3D':
                        vals[_i] = (ses3d_dmn.df[component].values[idx] + val) * taper + (1.0 - taper) * ses3d_dmn.df[component].values[idx]
                    elif self.model_info['component_type'] == 'absolute':
                        vals[_i] = taper * val + (1.0 - taper) * ses3d_dmn.df[component].values[idx]
                    else:
                        print(
                            'No valid component_type. Must be perturbation_to_1D, perturbation_to_3D or absolute')
                return vals

            a = np.vectorize(set_val, signature='()->(n)')
            return a(point_indices)

        import multiprocessing

        work_list = np.arange(len(ses3d_dmn.df["r"]))

        num_processes = multiprocessing.cpu_count()
        n = min(20 * num_processes, len(work_list))

        task_list = np.array_split(work_list, n)
        results = []
        with multiprocessing.Pool(num_processes) as pool:
            with tqdm(total=len(task_list)) as pbar:
                for i, r in enumerate(pool.imap(_process, task_list)):
                    results.append(r)
                    pbar.update()

            pool.close()
            pool.join()
        results = np.concatenate(results)

        for _i, component in enumerate(self.components):
            ses3d_dmn.df[component].values[:] = results[:, _i]

            # Rotate back to the actual physical domain.
        geometry = self.model_info['geometry']
        if geometry['rotation'] is True and self.interp_method == "trilinear":
            ses3d_dmn.rotate(np.radians(geometry['rot_angle']), geometry['rot_x'], geometry['rot_y'], geometry['rot_z'])

        # Finally update GridData object
        GridData.df.update(ses3d_dmn.df)

