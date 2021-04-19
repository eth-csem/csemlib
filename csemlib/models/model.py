import numpy as np

from meshpy.tet import build, Options, MeshInfo


def triangulate(x, y, z, true_x=None, true_y=None, true_z=None):
    """Triangulate a point cloud.

    Given a 3-D point cloud, defined by x, y, z, perform a Delauny triangulation
    and return a list of elements.
    :param x: (Cartesian) x value.
    :param y: (Cartesian) y value.
    :param z: (Cartesian) z value.
    :returns: A list-of-lists containing connectivity of the resulting mesh.
    """

    print("COMPUTING DELAUNEY TRIANGULATION")
    # Set up the simplex vertices.
    pts = np.array((x, y, z)).T

    # Do the triangulation with MeshPy.
    # Currently, this seems like the fastest way.
    mesh_info = MeshInfo()
    mesh_info.set_points(pts)
    opts = Options("Q")
    mesh = build(mesh_info, options=opts)

    return mesh.elements


def write_vtk(filename, points, tetra, vals, name=None):
    """Writes a vtk from the given set of grid locations, values and connectivity

    :param points: An ndarray containing all the grid locations in cartesion coordinates
           in the form of:  | x0 y0 z0 |
                            | :  :  :  |
                            | xn yn zn |

    :param vals: an array containing the values to be specified on the mesh
    :param tetra: connectivity of the mesh

    """
    import pyvtk
    from pyvtk import PointData, Scalars
    vtkElements = pyvtk.VtkData(
                    pyvtk.UnstructuredGrid(
                    points,
                    tetra=tetra),
                    PointData(Scalars(vals, name)),
                    "Mesh")

    vtkElements.tofile(filename)
