import os
import numpy as np

def write_to_ses3d(directory, component, filename, GridData):
    """
    Write ses3d-style model files. This assumes that the coordinates in GridData are properly sorted so that they
    actually make a meaningful regular ses3d grid. Also, it is assumed that the block_* files and the modelinfo.yml
    file are located in the target directory.
    :param directory: Target directory where the model file will be written.
    :param component: Component, e.g. 'vsv', 'rho', etc.
    :param filename: Filename of the model file.
    :param GridData: Valid GridData structure.
    :return: No return value.
    """

    if (component in GridData.components):

        print('Write ses3d file.\n')

        fid_m = open(os.path.join(directory,filename), 'w')

        # Read block files.

        fid_x = open(os.path.join(directory, 'block_x'), 'r')
        fid_y = open(os.path.join(directory, 'block_y'), 'r')
        fid_z = open(os.path.join(directory, 'block_z'), 'r')

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

        # March through the subvolumes and write file.

        fid_m.write(str(nsubvol) + '\n')

        for n in np.arange(nsubvol, dtype=int):

            nx = int(dx[idx[n]]) - 1
            ny = int(dy[idy[n]]) - 1
            nz = int(dz[idz[n]]) - 1

            fid_m.write(str(nx * ny * nz) + '\n')

            for i in np.arange(nx*ny*nz):
                fid_m.write(str(GridData.df[component][i]) + '\n')

        # Clean up.

        fid_m.close()

    else:

        print(component+' is not a valid component.\n')