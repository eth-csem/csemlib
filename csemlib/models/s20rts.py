import os
import numpy as np

from model import Model
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

        #- Convert to longitude range from 0 - 2*pi.
        for i in np.where(lon<0.0)[0]:
            lon[i]+=2.0*np.pi

        # Set coordinate axes.
        d_deg = 5.0 * np.pi / 180.0
        colats = np.arange(d_deg, np.pi, d_deg)
        lons = np.arange(d_deg, 2.0 * np.pi, d_deg)
        rads = np.arange(3480.0, 6371.0, 20.0)

        # Read gridded S20RTS file.
        fid=open('csemlib/data/s20rts/s20rts_gridded.dat','r')
        dv=np.zeros(360325)
        #c=np.zeros(360325)
        #l=np.zeros(360325)
        #r=np.zeros(360325)
        i=0
        for f in fid:
            #r[i]=float(f.split(' ')[0])
            #c[i]=float(f.split(' ')[1])
            #l[i]=float(f.split(' ')[2])
            dv[i]=float(f.split(' ')[3])
            i+=1
        fid.close()

        # March through all input coordinates.
        dv_out=np.zeros(len(colat))
        for i in range(len(colat)):

            # Get individual indeces of the coordinates in the grid file.
            idx_colat=min(np.where(np.min(np.abs(colat[i]-colats))==np.abs(colat[i]-colats))[0])
            idx_lon=min(np.where(np.min(np.abs(lon[i]-lons))==np.abs(lon[i]-lons))[0])
            idx_rad=min(np.where(np.min(np.abs(rad[i]-rads))==np.abs(rad[i]-rads))[0])

            # Total index.
            idx=idx_rad*(len(colats)*len(lons))+idx_colat*len(lons)+idx_lon
            dv_out[i]=dv[idx]

            # Debugging stuff.
            #dv_truth=self.eval(np.array([colat[i]]), np.array([lon[i]]), np.array([rad[i]]))
            #print r[idx], c[idx], l[idx], dv[idx]
            #print rad[i], colat[i], lon[i], dv_truth[0]
            #print '--------------------------------\n'

        return dv_out


    def split_domains_griddata(self, GridData):
        """
        This returns a new GridData object which only includes the points that lie within the S20RTS domain,
        i.e. that are above the core-mantle boundary. Points at a depth less than the minimum depth of S20RTS
        (24.37 km), are set to that minimum depth in self.eval.
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
        self.read()
        s20rts_dmn = self.split_domains_griddata(GridData)

        if len(s20rts_dmn) < 1:
            return GridData

        # Get velocity perturbation
        dv = self.eval_gridded(s20rts_dmn.df['c'], s20rts_dmn.df['l'], s20rts_dmn.df['r'])

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
