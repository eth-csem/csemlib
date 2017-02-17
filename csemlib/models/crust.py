import io
import os

import numpy as np
import scipy.interpolate as interp
import xarray

from .model import Model
from .topography import Topography


class Crust(Model):
    """
    Class handling crustal models.
    """

    def __init__(self):

        super(Crust, self).__init__()
        self._data = xarray.Dataset()

        directory = os.path.split(os.path.split(__file__)[0])[0]
        self.directory = os.path.join(directory, 'data', 'crust')
        self._data = xarray.Dataset()
        self.crust_dep_smooth_fac = 1e1
        self.crust_vs_smooth_fac = 0

    def data(self):
        pass

    def read(self):

        with io.open(os.path.join(self.directory, 'crust_x'), 'rt') as fh:
            col = np.asarray(fh.readlines(), dtype=float)

        with io.open(os.path.join(self.directory, 'crust_y'), 'rt') as fh:
            lon = np.asarray(fh.readlines(), dtype=float)

        for p in ['crust_dep', 'crust_vs']:

            with io.open(os.path.join(self.directory, p), 'rt') as fh:
                val = np.asarray(fh.readlines(), dtype=float)
            val = val.reshape(len(col), len(lon))
            self._data[p] = (('col', 'lon'), val)
            if p == 'crust_dep':
                self._data[p].attrs['units'] = 'km'
            elif p == 'crust_vs':
                self._data[p].attrs['units'] = 'km/s'

        # Add coordinates.
        self._data.coords['col'] = np.radians(col)
        self._data.coords['lon'] = np.radians(lon)

        # Add units.
        self._data.coords['col'].attrs['units'] = 'radians'
        self._data.coords['lon'].attrs['units'] = 'radians'

    def write(self):
        pass

    def eval(self, x, y, z=0, param=None, smooth_fac=1e5):
        # Create smoother object.
        lut = interp.RectSphereBivariateSpline(self._data.coords['col'][::-1],
                                               self._data.coords['lon'],
                                               self._data[param],
                                               s=smooth_fac)

        # Because the colatitude array is reversed, we must also reverse the request.
        x = np.pi - x

        # Convert longitudes to coordinate system of the crustal model
        lon = np.copy(y)
        lon[lon < 0] = 2 * np.pi + lon[lon < 0]
        return lut.ev(x, lon)


    def eval_point_cloud_grid_data(self, GridData):
            print('Evaluating Crust')
            self.read()
            r_earth = 6371.0

            # Split into crustal and non crustal zone
            cst_zone = GridData.df[GridData.df['r'] >= (r_earth - 100.5)]

            # Compute crustal depths and vs for crustal zone coordinates
            crust_dep = self.eval(cst_zone['c'], cst_zone['l'], param='crust_dep', smooth_fac=self.crust_dep_smooth_fac)
            crust_vs = self.eval(cst_zone['c'], cst_zone['l'], param='crust_vs', smooth_fac=self.crust_vs_smooth_fac)

            # Get Topography
            top = Topography()
            top.read()
            topo = top.eval(cst_zone['c'], cst_zone['l'])

            # Convert crust_depth to thickness, below mountains increase with (positive) topography,
            # In oceans add (negative) topography
            crust_dep += topo

            # Add crust and apply a 25 percent taper
            cst_zone = add_crust_all_params_topo_griddata_with_taper(cst_zone, crust_dep, crust_vs, topo,
                                                                     taper_percentage=0.25)
            # Append crustal and non crustal zone back together
            GridData.df.update(cst_zone)

            return GridData

def add_crust_all_params_topo_griddata_with_taper(cst_zone, crust_dep, crust_vs, topo, taper_percentage=0.25):

    r_earth = 6371.0
    for i in range(len(cst_zone['r'])):
        taper_hwidth = crust_dep[i] * taper_percentage
        # If Above taper overwrite with crust
        if cst_zone['r'].values[i] > (r_earth - crust_dep[i] + taper_hwidth):
            # Do something with param here
            if 'vsv' in cst_zone.columns:
                cst_zone['vsv'].values[i] = crust_vs[i]
            if 'vsh' in cst_zone.columns:
                cst_zone['vsh'].values[i] = crust_vs[i]

            # Continental crust
            if topo[i] >= 0:
                if 'vpv' in cst_zone.columns:
                    cst_zone['vpv'].values[i] = 1.5399 * crust_vs[i] + 0.840
                if 'vph' in cst_zone.columns:
                    cst_zone['vph'].values[i] = 1.5399 * crust_vs[i] + 0.840
                if 'vp' in cst_zone.columns:
                    cst_zone['vp'].values[i] = 1.5399 * crust_vs[i] + 0.840
                if 'rho' in cst_zone.columns:
                    cst_zone['rho'].values[i] = 0.2277 * crust_vs[i] + 2.016

            # Oceanic crust
            if topo[i] < 0:
                if 'vpv' in cst_zone.columns:
                    cst_zone['vpv'].values[i] = 1.5865 * crust_vs[i] + 0.844
                if 'vph' in cst_zone.columns:
                    cst_zone['vph'].values[i] = 1.5865 * crust_vs[i] + 0.844
                if 'vp' in cst_zone.columns:
                    cst_zone['vph'].values[i] = 1.5865 * crust_vs[i] + 0.844
                if 'rho' in cst_zone.columns:
                    cst_zone['rho'].values[i] = 0.2547 * crust_vs[i] + 1.979

        # If below taper region in mantle, do nothing and continue
        elif cst_zone['r'].values[i] < (r_earth - crust_dep[i] - taper_hwidth):
            continue

        # In taper region
        else:
            dist_from_mantle = cst_zone['r'].values[i] - (r_earth - crust_dep[i] - taper_hwidth)
            taper_width = 2 * taper_hwidth
            frac_crust = dist_from_mantle / taper_width
            frac_mantle = 1.0 - frac_crust

            # Do something with param here
            if 'vsv' in cst_zone.columns:
                cst_zone['vsv'].values[i] = (crust_vs[i] * frac_crust) + (cst_zone['vsv'].values[i] * frac_mantle)
            if 'vsh' in cst_zone.columns:
                cst_zone['vsh'].values[i] = (crust_vs[i] * frac_crust) + (cst_zone['vsh'].values[i] * frac_mantle)

            # Continental crust
            if topo[i] >= 0:
                if 'vpv' in cst_zone.columns:
                    cst_zone['vpv'].values[i] = ((1.5399 * crust_vs[i] + 0.840) * frac_crust) +\
                                                (cst_zone['vpv'].values[i] * frac_mantle)
                if 'vph' in cst_zone.columns:
                    cst_zone['vph'].values[i] = ((1.5399 * crust_vs[i] + 0.840) * frac_crust) +\
                                                (cst_zone['vph'].values[i] * frac_mantle)
                if 'vp' in cst_zone.columns:
                    cst_zone['vp'].values[i] = ((1.5399 * crust_vs[i] + 0.840) * frac_crust) +\
                                               (cst_zone['vp'].values[i] * frac_mantle)
                if 'rho' in cst_zone.columns:
                    cst_zone['rho'].values[i] = ((0.2277 * crust_vs[i] + 2.016) * frac_crust) +\
                                                (cst_zone['rho'].values[i] * frac_mantle)

            # Oceanic crust
            if topo[i] < 0:
                if 'vpv' in cst_zone.columns:
                    cst_zone['vpv'].values[i] = ((1.5865 * crust_vs[i] + 0.844) * frac_crust) +\
                                                (cst_zone['vpv'].values[i] * frac_mantle)
                if 'vph' in cst_zone.columns:
                    cst_zone['vph'].values[i] = ((1.5865 * crust_vs[i] + 0.844) * frac_crust) +\
                                                (cst_zone['vph'].values[i] * frac_mantle)
                if 'vp' in cst_zone.columns:
                    cst_zone['vph'].values[i] = ((1.5865 * crust_vs[i] + 0.844) * frac_crust) +\
                                                (cst_zone['vp'].values[i] * frac_mantle)
                if 'rho' in cst_zone.columns:
                    cst_zone['rho'].values[i] = ((0.2547 * crust_vs[i] + 1.979) * frac_crust) +\
                                                (cst_zone['rho'].values[i] * frac_mantle)

    return cst_zone
