import os

import numpy as np
import scipy.interpolate as interp
import xarray
import h5py


class Topography(object):
    """
    Class handling topography models.
    """

    def __init__(self):

        super(Topography, self).__init__()
        self._data = xarray.Dataset()

        directory = os.path.split(os.path.split(__file__)[0])[0]
        self.directory = os.path.join(directory, 'data', 'topography')

    def data(self):
        pass

    def read(self):
        # Read txt file
        #val = np.genfromtxt(os.path.join(self.directory, 'topo_resampled_1hour.txt'))

        # write hdf5
        # filename = os.path.join(self.directory, 'topo_resampled_1hour.hdf5')
        # f = h5py.File(filename, "w")
        # f.create_dataset('topo_resampled', data=val, dtype='f')
        # f.close()

        # Read values hdf5
        filename = os.path.join(self.directory, 'topo_resampled_1hour.hdf5')
        f = h5py.File(filename, "r")
        val = f['topo_resampled'][:]

        # sampling:
        start = 1
        col = np.linspace(start, 180-start, 179)
        lon = np.linspace(start, 360-start, 359)
        # Reshape
        val = val.reshape(len(col), len(lon))

        # Convert to km
        val /= 1000.0
        self._data['topo'] = (('col', 'lon'), val)
        self._data['topo'].attrs['units'] = 'km'

        # Add coordinates.
        self._data.coords['col'] = np.radians(col)
        self._data.coords['lon'] = np.radians(lon)

        # Add units.
        self._data.coords['col'].attrs['units'] = 'radians'
        self._data.coords['lon'].attrs['units'] = 'radians'


    def write(self):
        pass

    def eval(self, c, l, topo_smooth_factor=0):
        """
        :param c: colatitude in radians
        :param l: longitude in radians
        :param topo_smooth_factor:
        :return: topography at colat, lon
        """
        # Fit a spline through the topography grid
        lut = interp.RectSphereBivariateSpline(self._data.coords['col'],
                                               self._data.coords['lon'],
                                               self._data['topo'],
                                               s=topo_smooth_factor, pole_values=[-4.228, -0.056])

        # Convert to coordinate system used for topography 0-2pi instead of -pi-pi
        l = l + np.pi
        return lut.ev(c, l)
