import io
import os
import numpy as np
import scipy.interpolate as interp
import xarray

from .topography import Topography

# TODO
# Eliminate smoothing of crustal properties.
# Introduce crustal anisotropy.
# The problem is really that the different scalings from vs to vp and rho depend on topography, which has a sharp boundary.

from ..helpers import load_lib
lib = load_lib()

class Crust(object):
    """
    Class handling crustal models.
    """

    def __init__(self):

        # Setup directories, data structure and smoothing factors.
        directory = os.path.split(os.path.split(__file__)[0])[0]
        self.directory = os.path.join(directory, 'data', 'crust')
        self._data = xarray.Dataset()
        self.crust_dep_smooth_fac = 0.0
        self.crust_vs_smooth_fac = 0.0

        # Read colatitude values.
        with io.open(os.path.join(self.directory, 'crust_x'), 'rt') as fh:
            col = np.asarray(fh.readlines(), dtype=float)

        # Read longitude values.
        with io.open(os.path.join(self.directory, 'crust_y'), 'rt') as fh:
            lon = np.asarray(fh.readlines(), dtype=float)

        # Read crustal depth and vs.
        for p in ['crust_dep', 'crust_vs']:

            with io.open(os.path.join(self.directory, p), 'rt') as fh:
                val = np.asarray(fh.readlines(), dtype=float)
            val = val.reshape(len(col), len(lon))
            self._data[p] = (('col', 'lon'), val)
            if p == 'crust_dep':
                self._data[p].attrs['units'] = 'km'
            elif p == 'crust_vs':
                self._data[p].attrs['units'] = 'km/s'

        # Add coordinates, converted to radians.
        self._data.coords['col'] = np.radians(col)
        self._data.coords['lon'] = np.radians(lon)

        # Add units.
        self._data.coords['col'].attrs['units'] = 'radians'
        self._data.coords['lon'].attrs['units'] = 'radians'


    def interpolate(self, colat, lon, param=None, smooth_fac=1e5):
        """
        Evaluate crustal depth and vs given on a regular spherical grid on the points of an arbitrary grid, using
        spline interpolation.
        :param colat: colatitude array.
        :param lon: longitude array.
        :param param: either 'crust_dep' or 'crust_vs'.
        :param smooth_fac: smoothing factor.
        :return: Interpolated crustal depth or crustal vs at the input colatitudes and longitudes.
        """

        # Create smoother object.
        lut = interp.RectSphereBivariateSpline(self._data.coords['col'][::-1], self._data.coords['lon'], self._data[param], s=smooth_fac)

        # Because the colatitude array is reversed, we must also reverse the request.
        colat_reverse = np.pi - colat

        # Convert longitudes to coordinate system of the crustal model.
        lon_reverse = np.copy(lon)
        lon_reverse[lon_reverse < 0.0] = 2.0 * np.pi + lon_reverse[lon_reverse < 0.0]
        return lut.ev(colat_reverse, lon_reverse)


    def eval_point_cloud_grid_data(self, GridData):
        """
        Get crustal depth and velocities at the grid points.
        :param GridData: GridData structure
        :return: Updated GridData.
        """

        print('Evaluating Crust')

        # Split into crustal and non crustal zone.
        cst_zone = GridData.df[GridData.df['r'] >= (6371.0 - 100.5)]

        # Compute crustal depths and vs for crustal zone coordinates at the relevant grid points through interpolation.
        crust_dep = self.interpolate(cst_zone['c'], cst_zone['l'], param='crust_dep', smooth_fac=self.crust_dep_smooth_fac)
        crust_vs = self.interpolate(cst_zone['c'], cst_zone['l'], param='crust_vs', smooth_fac=self.crust_vs_smooth_fac)

        # Get Topography.
        top = Topography()
        top.read()
        topo = top.eval(cst_zone['c'], cst_zone['l'])

        # Convert crust_depth to thickness, below mountains increase with (positive) topography. In oceans add (negative) topography.
        crust_dep += topo

        # Add crust and apply a 25 percent taper.
        #cst_zone = add_crust_all_params_topo_griddata_with_taper(cst_zone, crust_dep, crust_vs, topo, taper_percentage=0.25)
        lib.add_crust(len(cst_zone), crust_dep, crust_vs, topo, cst_zone['vsv'].values, cst_zone['vsh'].values,
                      cst_zone['vpv'].values, cst_zone['vph'].values, cst_zone['rho'].values, cst_zone['r'].values)



        # Append crustal and non crustal zone back together.
        GridData.df.update(cst_zone)

        return GridData


def add_crust_all_params_topo_griddata_with_taper(cst_zone, crust_dep, crust_vs, topo, taper_percentage=0.25):
    """
    Scale crustal vs to the other parameters.
    :param cst_zone: Part of GridData that are actually within the crust.
    :param crust_dep: Array of crustal depth.
    :param crust_vs: Array of crustal velocity.
    :param topo: Array of topography, needed to discriminate oceanic and continental scaling.
    :param taper_percentage: Vertical taper percentage of the crustal thickness.
    :return: Updated GridData.
    """

    r_earth = 6371.0
    r_ani = 6191.0
    s_ani = 0.0011

    for i in range(len(cst_zone['r'])):

        taper_hwidth = crust_dep[i] * taper_percentage

        # If above taper overwrite with crust.
        if cst_zone['r'].values[i] > (r_earth - crust_dep[i] + taper_hwidth):

            # Ascribe crustal vsh and vsv based on the averaged vs by Meier et al. (2007).
            if 'vsv' in cst_zone.columns:
                cst_zone['vsv'].values[i] = crust_vs[i] - 0.5 * s_ani * (cst_zone['r'].values[i] - r_ani)
            if 'vsh' in cst_zone.columns:
                cst_zone['vsh'].values[i] = crust_vs[i] + 0.5 * s_ani * (cst_zone['r'].values[i] - r_ani)

            # Scaling to P velocities and density for continental crust.
            if topo[i] >= 0:
                if 'vpv' in cst_zone.columns:
                    cst_zone['vpv'].values[i] = 1.5399 * crust_vs[i] + 0.840
                if 'vph' in cst_zone.columns:
                    cst_zone['vph'].values[i] = 1.5399 * crust_vs[i] + 0.840
                if 'vp' in cst_zone.columns:
                    cst_zone['vp'].values[i] = 1.5399 * crust_vs[i] + 0.840
                if 'rho' in cst_zone.columns:
                    cst_zone['rho'].values[i] = 0.2277 * crust_vs[i] + 2.016

            # Scaling to P velocities and density for oceanic crust.
            if topo[i] < 0:
                if 'vpv' in cst_zone.columns:
                    cst_zone['vpv'].values[i] = 1.5865 * crust_vs[i] + 0.844
                if 'vph' in cst_zone.columns:
                    cst_zone['vph'].values[i] = 1.5865 * crust_vs[i] + 0.844
                if 'vp' in cst_zone.columns:
                    cst_zone['vp'].values[i] = 1.5865 * crust_vs[i] + 0.844
                if 'rho' in cst_zone.columns:
                    cst_zone['rho'].values[i] = 0.2547 * crust_vs[i] + 1.979

        # If below taper region in mantle, do nothing and continue.
        elif cst_zone['r'].values[i] < (r_earth - crust_dep[i] - taper_hwidth):
            continue

        # In taper region, taper linearly to mantle properties.
        else:
            dist_from_mantle = cst_zone['r'].values[i] - (r_earth - crust_dep[i] - taper_hwidth)
            taper_width = 2.0 * taper_hwidth
            frac_crust = dist_from_mantle / taper_width
            frac_mantle = 1.0 - frac_crust

            # Ascribe crustal vsh and vsv based on the averaged vs by Meier et al. (2007).
            if 'vsv' in cst_zone.columns:
                vsv_crust = crust_vs[i] - 0.5 * s_ani * (cst_zone['r'].values[i] - r_ani)
                cst_zone['vsv'].values[i] = (vsv_crust * frac_crust) + (cst_zone['vsv'].values[i] * frac_mantle)
            if 'vsh' in cst_zone.columns:
                vsh_crust = crust_vs[i] + 0.5 * s_ani * (cst_zone['r'].values[i] - r_ani)
                cst_zone['vsh'].values[i] = (vsh_crust * frac_crust) + (cst_zone['vsh'].values[i] * frac_mantle)

            # Scaling to P velocities and density for continental crust.
            if topo[i] >= 0:
                if 'vpv' in cst_zone.columns:
                    cst_zone['vpv'].values[i] = (1.5399 * crust_vs[i] + 0.840) * frac_crust + (cst_zone['vpv'].values[i] * frac_mantle)
                if 'vph' in cst_zone.columns:
                    cst_zone['vph'].values[i] = (1.5399 * crust_vs[i] + 0.840) * frac_crust + (cst_zone['vph'].values[i] * frac_mantle)
                if 'vp' in cst_zone.columns:
                    cst_zone['vp'].values[i] = (1.5399 * crust_vs[i] + 0.840) * frac_crust + (cst_zone['vp'].values[i] * frac_mantle)
                if 'rho' in cst_zone.columns:
                    cst_zone['rho'].values[i] = (0.2277 * crust_vs[i] + 2.016) * frac_crust + (cst_zone['rho'].values[i] * frac_mantle)

            # Scling to P velocity and density for oceanic crust.
            if topo[i] < 0:
                if 'vpv' in cst_zone.columns:
                    cst_zone['vpv'].values[i] = (1.5865 * crust_vs[i] + 0.844) * frac_crust + (cst_zone['vpv'].values[i] * frac_mantle)
                if 'vph' in cst_zone.columns:
                    cst_zone['vph'].values[i] = (1.5865 * crust_vs[i] + 0.844) * frac_crust + (cst_zone['vph'].values[i] * frac_mantle)
                if 'vp' in cst_zone.columns:
                    cst_zone['vp'].values[i] = (1.5865 * crust_vs[i] + 0.844) * frac_crust + (cst_zone['vp'].values[i] * frac_mantle)
                if 'rho' in cst_zone.columns:
                    cst_zone['rho'].values[i] = (0.2547 * crust_vs[i] + 1.979) * frac_crust + (cst_zone['rho'].values[i] * frac_mantle)

    return cst_zone
