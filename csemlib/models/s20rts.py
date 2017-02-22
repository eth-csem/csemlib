import os
import numpy as np
from csemlib.utils import lagrange as L
from ..lib import s20eval


class S20rts(object):
    """
    Class handling S20rts evaluations.
    """

    def __init__(self):
        """
        Initialise S20RTS evaluations. This includes setting up info for the evaluation of spherical harmonics
        (the original S20RTS), and the gridded version of S20RTS (made for better scaling).
        """
        super(S20rts, self).__init__()

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
        fid = open('csemlib/data/s20rts/s20rts_gridded.dat', 'r')
        v_dummy = np.zeros(1898611)
        i = 0
        for f in fid:
            v_dummy[i] = float(f.split(' ')[3])
            i += 1
        fid.close()

        self.dv=np.reshape(v_dummy,(187,71,143))


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

        start=time.time()

        #- Convert to longitude range from 0 - 2*pi.
        for i in np.where(lon<0.0)[0]:
            lon[i]+=2.0*np.pi

        # Set coordinate axes.
        d_deg = 2.5 * np.pi / 180.0
        c = np.arange(d_deg, np.pi, d_deg)
        l = np.arange(d_deg, 2.0 * np.pi, d_deg)
        r_1 = np.arange(3480.0, 5480.0, 20.0)
        r_2 = np.arange(5480.0, 6350.0, 10.0)
        r = np.concatenate((r_1, r_2))

        # March through all input coordinates.
        dv_out = np.zeros(len(colat))

        for i in range(len(colat)):

                colat_i=colat[i]
                lon_i=lon[i]
                rad_i=rad[i]

                # Get individual indeces of the coordinates in the grid file.
                ic=min(np.where(np.min(np.abs(colat_i-c))==np.abs(colat_i-c))[0])
                il=min(np.where(np.min(np.abs(lon_i-l))==np.abs(lon_i-l))[0])
                ir=min(np.where(np.min(np.abs(rad_i-r))==np.abs(rad_i-r))[0])

                # Indeces of depth layers above and below.
                if (r[ir]>rad_i):
                    irm=ir
                    irp=ir+1
                else:
                    irm=ir-1
                    irp=ir

                # Weights for linear depth interpolation.
                m=(rad_i-r[irp])/(r[irm]-r[irp])
                p=(rad_i-r[irm])/(r[irp]-r[irm])

                # Quadratic interpolation in latitude and longitude, except at the poles.
                if (ic>0 and ic<70):
                    # Handle the crossing of the longitude dateline.
                    if (il==0):
                        ilm=142
                    else:
                        ilm=il-1
                    if (il==142):
                        ilp=0
                    else:
                        ilp=il+1

                    # Precompute terms for Lagrange interpolation.
                    if ((i==0) or (lon_i!=lon[i-1])):
                        Ll0mp=L(lon_i,l[il],l[ilm],l[ilp])
                        Llm0p=L(lon_i,l[ilm],l[il],l[ilp])
                        Llpm0=L(lon_i,l[ilp],l[ilm],l[il])
                    if ((i==0) or (colat_i!=colat[i-1])):
                        Lc0mp=L(colat_i,c[ic],c[ic-1],c[ic+1])
                        Lcm0p=L(colat_i,c[ic-1],c[ic],c[ic+1])
                        Lcpm0=L(colat_i,c[ic+1],c[ic-1],c[ic])

                    # Lagrange interpolation.
                    dv_out[i]=(p*self.dv[irp,ic,il]+m*self.dv[irm,ic,il])*Lc0mp*Ll0mp \
                    +(p*self.dv[irp,ic-1,il]+m*self.dv[irm,ic-1,il])*Lcm0p*Ll0mp \
                    +(p*self.dv[irp,ic+1,il]+m*self.dv[irm,ic+1,il])*Lcpm0*Ll0mp \
                    +(p*self.dv[irp,ic,ilm]+m*self.dv[irm,ic,ilm])*Lc0mp*Llm0p \
                    +(p*self.dv[irp,ic-1,ilm]+m*self.dv[irm,ic-1,ilm])*Lcm0p*Llm0p \
                    +(p*self.dv[irp,ic+1,ilm]+m*self.dv[irp,ic+1,ilm])*Lcpm0*Llm0p \
                    +(p*self.dv[irp,ic,ilp]+m*self.dv[irm,ic,ilp])*Lc0mp*Llpm0 \
                    +(p*self.dv[irp,ic-1,ilp]+m*self.dv[irm,ic-1,ilp])*Lcm0p*Llpm0 \
                    +(p*self.dv[irp,ic+1,ilp]+m*self.dv[irm,ic+1,ilp])*Lcpm0*Llpm0
                else:
                    dv_out[i] = (p*self.dv[irp, ic, il]+m*self.dv[irm, ic, il])

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
        #dv = self.eval_gridded(s20rts_dmn.df['c'], s20rts_dmn.df['l'], s20rts_dmn.df['r'])
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
