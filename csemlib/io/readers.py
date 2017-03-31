import os
import io
import yaml
import numpy as np
from ..utils import rotate, get_rot_matrix

# Read cartesian grid points. ==========================================================================================

def read_from_grid(filename):
    """
    Read cartesian grid points from a file named "filename", organised as
    x1, y1, z1
    x2, y2, z2
    ... .
    """

    x=[]
    y=[]
    z=[]

    fid=open(filename,'r')

    for point in fid:
        x.append(float(point.split()[0]))
        y.append(float(point.split()[1]))
        z.append(float(point.split()[2]))

    fid.close()

    return x, y, z


# Read from ses3d block files. =========================================================================================

def read_from_ses3d_block(directory):
    """
    Compute Cartesian x, y, z coordinates based on SES3D block files and a modelinfo.yml file that contains the
    relevant rotation parameters.
    :param directory: Directory where block_* and modelinfo.yml are located.
    :return: Cartesian x, y, z coordinates.
    """

    # Initialise arrays of Cartesian coordinates.

    x=[]
    y=[]
    z=[]

    # Read yaml file containing information on the ses3d submodel.
    with io.open(os.path.join(directory,'modelinfo.yml'), 'rt') as fh:
        model_info = yaml.load(fh)

    rot_vec = np.array([model_info['geometry']['rot_x'], model_info['geometry']['rot_y'], model_info['geometry']['rot_z']])
    rot_angle = model_info['geometry']['rot_angle']

    # Read block files.

    fid_x = open(os.path.join(directory,'block_x'), 'r')
    fid_y = open(os.path.join(directory,'block_y'), 'r')
    fid_z = open(os.path.join(directory,'block_z'), 'r')

    dx = np.array(fid_x.read().strip().split('\n'), dtype=float)
    dy = np.array(fid_y.read().strip().split('\n'), dtype=float)
    dz = np.array(fid_z.read().strip().split('\n'), dtype=float)

    fid_x.close()
    fid_y.close()
    fid_z.close()

    # Setup of coordinate lines.

    nsubvol = int(dx[0])

    idx = np.ones(nsubvol, dtype=int)
    idy = np.ones(nsubvol, dtype=int)
    idz = np.ones(nsubvol, dtype=int)

    for k in np.arange(1, nsubvol, dtype=int):
        idx[k] = int(dx[idx[k - 1]]) + idx[k - 1] + 1
        idy[k] = int(dy[idy[k - 1]]) + idy[k - 1] + 1
        idz[k] = int(dz[idz[k - 1]]) + idz[k - 1] + 1

    for k in np.arange(nsubvol, dtype=int):

        # Coordinates of the box corners.
        colat = dx[(idx[k] + 1):(idx[k] + 1 + int(dx[idx[k]]))]
        lon = dy[(idy[k] + 1):(idy[k] + 1 + int(dy[idy[k]]))]
        rad = dz[(idz[k] + 1):(idz[k] + 1 + int(dz[idz[k]]))] 
        
        # Coordinates of the box centroids.
        colat_c = (np.array(colat[0:-1])+np.array(colat[1:]))/2.0
        lon_c = (np.array(lon[0:-1]) + np.array(lon[1:]))/2.0
        rad_c = (np.array(rad[0:-1]) + np.array(rad[1:]))/2.0
        
        # Compute Cartesian coordinates for all grid points.
        for c in colat_c:
            for l in lon_c:
                xx=np.cos(l*np.pi/180.0)*np.sin(c*np.pi/180.0)
                yy=np.sin(l*np.pi/180.0)*np.sin(c*np.pi/180.0)
                zz=np.cos(c*np.pi/180.0)
                for r in rad_c:
                    x.append(r*xx)
                    y.append(r*yy)
                    z.append(r*zz)
                        #if (k==1):
                        #print r*np.sqrt(xx*xx+yy*yy+zz*zz)

    # Rotate, if needed.

    if (rot_angle!=0.0):
        rot_mat = get_rot_matrix(rot_angle*np.pi/180.0, *rot_vec)
        x, y, z = rotate(x, y, z, rot_mat)

    # Return.

    return x, y, z