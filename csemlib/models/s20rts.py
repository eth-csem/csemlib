import io
import os

import numpy as np
from scipy.special import sph_harm

from .model import Model
from ..lib import s20eval


class S20rts(Model):
    """
    Class handling S20rts evaluations.
    """

    def data(self):
        pass

    def __init__(self):
        super(S20rts, self).__init__()
        directory, _ = os.path.split(os.path.split(__file__)[0])
        self.directory = os.path.join(directory, 'data', 's20rts')
        self.mfl = os.path.join(self.directory, 'S20RTS.sph')
        self.layers = np.array([6346.63, 6296.63, 6241.64, 6181.14, 6114.57, 6041.34, 5960.79,
                                5872.18, 5774.69, 5667.44, 5549.46, 5419.68, 5276.89, 5119.82,
                                4947.02, 4756.93, 4547.81, 4317.74, 4064.66, 3786.25, 3479.96])
        self.r_earth = 6371.0
        self.wasread = False

    def read(self):
        pass

    def write(self):
        pass

    def eval(self, c, l, rad):

        lat = 90.0 - np.degrees(c)
        lon = np.degrees(l) - 360.0
        dep = self.r_earth - rad

        dv = np.zeros(len(lat))

        # Get velocity perturbation
        dv = s20eval.sph2v(lat, lon, dep, dv, self.mfl, self.wasread)
        self.wasread = False
        return dv

    def split_domains_griddata(self, GridData):
        """
        This returns a new GridData object which only includes the points that lie within the S20RTS domain
        :param GridData:
        :return s20rts_dmn:
        """

        s20rts_dmn = GridData.copy()
        s20rts_dmn.df = s20rts_dmn.df[s20rts_dmn.df['r'] <= self.layers[0]]
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
        self.read()
        s20rts_dmn = self.split_domains_griddata(GridData)

        if len(s20rts_dmn) < 1:
            return GridData

        # Get velocity perturbation
        dv = self.eval(s20rts_dmn.df['c'], s20rts_dmn.df['l'], s20rts_dmn.df['r'])

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
