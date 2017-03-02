import os
import time
import numpy as np
from csemlib.utils import lagrange as L
from ..lib import s20eval

from ..lib.helpers import load_lib
lib = load_lib()

class S20rts(object):
    """
    Class handling S20rts evaluations.
    """

    def __init__(self):
        """
        Initialise S20RTS evaluations. This includes setting up info for the evaluation of spherical harmonics
        (the original S20RTS), and the gridded version of S20RTS (made for better scaling).
        """

        # Basic info for the original S20RTS (spherical harmonics).
        directory, _ = os.path.split(os.path.split(__file__)[0])
        self.directory = os.path.join(directory, 'data', 's20rts')
        self.mfl = os.path.join(self.directory, 'S20RTS.sph')
        self.layers = np.array([6346.63, 6296.63, 6241.64, 6181.14, 6114.57, 6041.34, 5960.79,
                                5872.18, 5774.69, 5667.44, 5549.46, 5419.68, 5276.89, 5119.82,
                                4947.02, 4756.93, 4547.81, 4317.74, 4064.66, 3786.25, 3479.96])
        self.r_earth = 6371.0
        self.wasread = False

        # Read gridded S20RTS file.
        #filename = os.path.join(self.directory, 's20rts_gridded.dat')
        filename = os.path.join(self.directory, 's20rts_gridded_again.dat')

        # fid = open(filename, 'r')
        # v_dummy = np.zeros(1898611)
        # i = 0
        # for f in fid:
        #     v_dummy[i] = float(f.split(' ')[3])
        #     i += 1
        # fid.close()
        #
        # self.dv=np.reshape(v_dummy,(187,71,143))
        dv = np.genfromtxt(filename)
        self.dv=np.reshape(dv,(187,73,143))
    def eval(self, colats, lons, rads):
        """
        Evaluate S20RTS using the original Fortran codes.
        :param colats: colatitude [radians]
        :param lons: longitude [radians]
        :param rads: radius [km]
        :return: Fractional S velocity perturbation.
        """

        # Compute latitude and longitude in degrees, and depth in km.
        lat = 90.0 - np.degrees(colats)
        lon = np.degrees(lons) - 360.0
        dep = self.r_earth - rads

        # If depth less than minimum depth in S20RTS (24.37 km), set to that depth.
        for i in np.where(dep<24.37)[0]:
            dep[i]=24.37

        # Get velocity perturbation
        dv = np.zeros(len(lat))
        dv = s20eval.sph2v(lat, lon, dep, dv, self.mfl, self.wasread)
        self.wasread = False
        return dv


    def eval_gridded(self, colat, lon, rad):
        """
        Evaluate S20RTS using a gridded version.
        :param colat: colatitude [radians]
        :param lon: longitude [radians]
        :param rad: radius [km]
        :return: Fractional S velocity perturbation.
        """

        # Convert to longitude range from 0 - 2*pi, copy is important, otherwise the line below will change the coordinates in grid_data
        lon_copy = lon.copy()
        lon_copy[lon_copy < 0.0] = lon_copy[lon_copy<0.0] + 2.0 * np.pi

        # Set coordinate axes.
        d_deg = 2.5/180 * np.pi
        c = np.linspace(0.0, np.pi, 73)
        #l = np.linspace(0.0, 2 * np.pi, 145)
        l = np.arange(d_deg, 2.0 * np.pi, d_deg)
        r_1 = np.arange(3480.0, 5480.0, 20.0)
        r_2 = np.arange(5480.0, 6350.0, 10.0)
        r = np.concatenate((r_1, r_2))

        # March through all input coordinates.
        n = len(colat)
        dv_out = np.zeros(n)
        lib.s20eval_grid(len(c), len(l), len(r), n, c, l, r, colat, lon_copy, rad, dv_out, self.dv)

        return dv_out


    def split_domains_griddata(self, GridData):
        """
        This returns a new GridData object which only includes the points that lie within the mantle.
        :param GridData:
        :return s20rts_dmn:
        """

        s20rts_dmn = GridData.copy()
        s20rts_dmn.df = s20rts_dmn.df[s20rts_dmn.df['r'] >= self.layers[-1]]

        return s20rts_dmn


    def eval_point_cloud_griddata(self, GridData):
        """
        This returns the linearly interpolated perturbations of s20rts. Careful only points that fall inside
        of the domain of s20rts are returned.
        :param GridData: Object of the GridData class
        :return updates GridData
        """
        print('Evaluating S20RTS')
        s20rts_dmn = self.split_domains_griddata(GridData)

        if len(s20rts_dmn) < 1:
            return GridData

        # Get velocity perturbation
        dv = self.eval_gridded(s20rts_dmn.df['c'].values, s20rts_dmn.df['l'].values, s20rts_dmn.df['r'].values)
        #dv = self.eval(s20rts_dmn.df['c'], s20rts_dmn.df['l'], s20rts_dmn.df['r'])

        # Compute vp perturbations
        R0 = 1.25
        R2891 = 3.0
        vp_slope = (R2891 - R0) / 2891.0
        rDep = vp_slope * (self.r_earth - s20rts_dmn.df['r']) + R0
        vp_val = dv / rDep

        # Add perturbations
        if 'vpv' in s20rts_dmn.components:
            s20rts_dmn.df['vpv'] *= (1 + vp_val)

        if 'vph' in s20rts_dmn.components:
            s20rts_dmn.df['vph'] *= (1 + vp_val)

        if 'vp' in s20rts_dmn.components:
            s20rts_dmn.df['vp'] *= (1 + vp_val)

        if 'vsv' in s20rts_dmn.components:
            s20rts_dmn.df['vsv'] *= (1 + dv)

        if 'vsh' in s20rts_dmn.components:
            s20rts_dmn.df['vsh'] *= (1 + dv)

        GridData.df.update(s20rts_dmn.df)

        return GridData
