
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
