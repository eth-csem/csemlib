import io
import os

import numpy as np
from scipy.special import sph_harm

from ...models.model import Model
import s20eval

mfl = os.path.join(os.path.split(__file__)[0], 'S20RTS.sph')
#mfl = '/home/sed/CSEM/csemlib/csemlib/models/s20RTS/S20RTS.sph'

class S20rts_f2py(Model):
    """
    Class handling S20rts evaluations.
    """

    def data(self):
        pass

    def __init__(self):
        super(S20rts_f2py, self).__init__()
        directory, _ = os.path.split(os.path.split(__file__)[0])
        self.directory = os.path.join(directory, 'data', 's20rts')
        self.layers = np.array([6346.63, 6296.63, 6241.64, 6181.14, 6114.57, 6041.34, 5960.79,
                                5872.18, 5774.69, 5667.44, 5549.46, 5419.68, 5276.89, 5119.82,
                                4947.02, 4756.93, 4547.81, 4317.74, 4064.66, 3786.25, 3479.96])
        self.r_earth = 6371.0

    def read(self):
        pass

    def write(self):
        pass

    def eval(self, c, l, rad):

        lat = 90.0 - np.degrees(c)
        lon = np.degrees(l) - 360.0
        dep = self.r_earth - rad
        wasread = False
        dv = np.zeros(len(lat))

        # Interpolate
        #mfl = '/home/sed/CSEM/csemlib/csemlib/models/s20RTS/S20RTS.sph'
        print(s20eval.sph2v(lat, lon, dep, dv, mfl, wasread))
        return dv


    def split_domains_griddata(self, GridData):
        """
        This splits an array of pts of all values into a
        :param pts:
        :return:
        """

        s20rts_dmn = GridData.copy()
        s20rts_dmn.df = s20rts_dmn.df[s20rts_dmn.df['r'] <= self.layers[0]]
        s20rts_dmn.df = s20rts_dmn.df[s20rts_dmn.df['r'] >= self.layers[-1]]

        return s20rts_dmn


    def eval_point_cloud_griddata(self, GridData):
        """
        This returns the linearly interpolated perturbations of s20rts. Careful only points that fall inside
        of the domain of s20rts are returned.
        :param c: colatitude
        :param l: longitude
        :param r: distance from core in km
        :param rho: param to be returned - currently not used
        :param vpv: param to be returned - currently not used
        :param vsv: param to be returned - currently not used
        :param vsh: param to be returned - currently not used
        :return c, l, r, rho, vpv, vsv, vsh
        """
        print('Evaluating S20RTS')
        self.read()
        s20rts_dmn = self.split_domains_griddata(GridData)

        # import f2py library



        # convert to lat, lon, dep format as required by f2py
        # lat format f2py: -90 to 90
        # lon format f2py -180 to 180
        # dep is distance from r_earth

        lat = 90.0 - np.degrees(s20rts_dmn.df['c'])
        lon = np.degrees(s20rts_dmn.df['l']) - 360.0
        dep = self.r_earth - s20rts_dmn.df['r']
        wasread = False
        dv = np.zeros(len(lat))

        # Interpolate
        dv = s20eval.sph2v(lat, lon, dep, dv, mfl, wasread)

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