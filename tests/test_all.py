import os

import numpy as np
from meshpy.tet import MeshInfo, build, Options

import csemlib
import csemlib.background.skeleton as skl
import csemlib.models.crust as crust
import csemlib.models.s20rts as s20
from csemlib.models.model import triangulate, write_vtk
from csemlib.models.topography import Topography
from csemlib.utils import cart2sph
import boltons.fileutils

TEST_DATA_DIR = os.path.join(os.path.split(__file__)[0], 'test_data')
VTK_DIR = os.path.join(os.path.split(__file__)[0], 'vtk')
boltons.fileutils.mkdir_p(VTK_DIR)
DECIMAL_CLOSE = 3


def test_fibonacci_sphere():
    true_y = np.array([-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9])
    true_x = np.array([0.43588989, -0.52658671, 0.0757129, 0.58041368, -0.97977755,
                       0.83952592, -0.24764672, -0.39915719, 0.67080958, -0.40291289])
    true_z = np.array([0., 0.48239656, -0.86270943, 0.75704687, -0.17330885,
                       -0.53403767, 0.92123347, -0.76855288, 0.24497858, 0.16631658])

    points = skl.fibonacci_sphere(10)
    np.testing.assert_almost_equal(points[0], true_x, decimal=DECIMAL_CLOSE)
    np.testing.assert_almost_equal(points[1], true_y, decimal=DECIMAL_CLOSE)
    np.testing.assert_almost_equal(points[2], true_z, decimal=DECIMAL_CLOSE)

def test_fibonacci_plane():
    true_x = np.array([0., 0.21360879, -0.44550123, 0.43467303, -0.11016234, -0.37952405, 0.74803968,
                       -0.74249263, 0.30682308, 0.36197619])
    true_y = np.array([0., -0.23317651, 0.03909797, 0.33325569, -0.62278749, 0.59662509, -0.20108863,
                       -0.38562248, 0.84015451, -0.87691119])

    points = skl.fib_plane(10)
    np.testing.assert_almost_equal(points[0], true_x, decimal=DECIMAL_CLOSE)
    np.testing.assert_almost_equal(points[1], true_y, decimal=DECIMAL_CLOSE)


def test_crust():
    """
    Test to ensure that the crust returns correct values.
    """

    proper_dep = np.array([[38.69471863, 17.96798953],
                           [38.69471863, 17.96798953]])
    proper_vs = np.array([[3.64649739, 3.1255109],
                          [3.64649739, 3.1255109]])

    cst = crust.Crust()
    cst.read()

    x = np.radians([179, 1])
    y = np.radians([1, 1])
    lats, lons = np.meshgrid(x, y)
    vals_dep = cst.eval(lats, lons, param='crust_dep')
    vals_vs = cst.eval(lats, lons, param='crust_vs')

    np.testing.assert_almost_equal(vals_dep, proper_dep, decimal=DECIMAL_CLOSE)
    np.testing.assert_almost_equal(vals_vs, proper_vs, decimal=DECIMAL_CLOSE)


def test_gourard_shading():
    """
    Test to see if the interpolation function works over a tetrahedra.
    """

    true_val = 4
    data = np.array([2, 2, 2, 4]).T
    bry = np.array([[0.5, 0.5, 0.5, 0.25]]).T
    idx = np.array([[0, 1, 2, 3]]).T

    np.testing.assert_almost_equal(
        csemlib.models.model.interpolate(idx, bry, data), true_val, decimal=DECIMAL_CLOSE)


def test_barycenter_detection():
    """
    Simple test to ensure that the interpolation routine works.
    """

    true_ind = np.array([[0, 0], [3, 3], [7, 7], [2, 2]], dtype=np.int64)
    true_bar = np.array([[1.00000000e+00, 0.00000000e+00],
                         [0.00000000e+00, 0.00000000e+00],
                         [0.00000000e+00, 1.00000000e+00],
                         [0.00000000e+00, 0.00000000e+00]])

    vertices = [
        (0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0),
        (0, 0, 12), (2, 0, 12), (2, 2, 12), (0, 2, 12),
    ]
    x_mesh, y_mesh, z_mesh = np.array(vertices)[:, 0], np.array(vertices)[:, 1], np.array(vertices)[:, 2]
    x_target, y_target, z_target = np.array([0, 0]), np.array([0, 2]), np.array([0, 12])
    mesh_info = MeshInfo()
    mesh_info.set_points(list(vertices))
    mesh_info.set_facets([
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 4, 5, 1],
        [1, 5, 6, 2],
        [2, 6, 7, 3],
        [3, 7, 4, 0],
    ])
    opts = Options("Q")
    mesh = build(mesh_info, options=opts)
    elements = np.array(mesh.elements)
    ind, bary = csemlib.models.model.shade(x_target, y_target, z_target,
                                           x_mesh, y_mesh, z_mesh,
                                           elements)
    np.testing.assert_almost_equal(ind, true_ind, decimal=DECIMAL_CLOSE)
    np.testing.assert_almost_equal(bary, true_bar, decimal=DECIMAL_CLOSE)

def test_s20rts():
    """
    Test to ensure that s20 rts calls returns some proper values.
    :return:
    """

    true = np.array([-0.01322954, -0.00347698,  0.00816347,  0.02046614,  0.02683848,
                     -0.00343986, -0.01152393, -0.01071749,  0.01334541,  0.01193581,
                     -0.01322954,  0.04389081,  0.04001087,  0.00036642, -0.02461444,
                     0.00270064,  0.00245818, -0.011406,  0.00755754,  0.01193708,
                     -0.01322954,  0.02528974,  0.02795101,  0.03040496, -0.00620491,
                     -0.01366555, -0.0165236, -0.0185142 ,  0.00993133,  0.011935,
                     -0.01322954,  0.0314661,  0.00938958,  0.00566042,  0.00808682,
                     0.01278651,  0.02467417, -0.00284243,  0.01496288,  0.01193301,
                     -0.01322954,  0.00321608,  0.0011454 , -0.00482245, -0.00309979,
                     -0.00798616, -0.02092096, -0.02821881, -0.0042038 ,  0.01193439,
                     -0.01322954,  0.00637217, -0.00427041, -0.0196469 , -0.02052032,
                     -0.02033845, -0.00154149, -0.01177908, -0.02167864,  0.01193683,
                     -0.01322954,  0.027213,  0.00939572, -0.0225926 , -0.01075957,
                     -0.01903457, -0.01392874, -0.01559766, -0.0103415 ,  0.01193628,
                     -0.01322954,  0.02329051,  0.03291302,  0.01023119,  0.00480987,
                     0.00873816, -0.00828948, -0.0032955 , -0.00431307,  0.01193364,
                     -0.01322954,  0.01203387, -0.00319457, -0.01521767, -0.00078595,
                     0.02169883,  0.00201059, -0.00751255,  0.00561583,  0.01193328,
                     -0.01322954, -0.00347699,  0.00816347,  0.02046615,  0.02683855,
                     -0.00343987, -0.01152392, -0.01071749,  0.01334541,  0.01193581])

    mod = s20.S20rts()

    size = 10
    col = np.linspace(0, np.pi, size)
    lon = np.linspace(0, 2 * np.pi, size)
    cols, lons = np.meshgrid(col, lon)
    cols = cols.flatten()
    lons = lons.flatten()
    rad = np.ones_like(cols) * mod.layers[3]

    vals = mod.eval(cols, lons, rad)

    np.testing.assert_almost_equal(vals, true, decimal=DECIMAL_CLOSE)

def test_s20rts_vtk_single_sphere():
    """
    Test to ensure that a vtk of a single sphere of s20rts is written succesfully.
    :return:

    """
    s20mod = s20.S20rts()
    s20mod.read()

    rad = s20mod.layers[0]
    rel_rad = rad/ s20mod.r_earth
    x, y, z = skl.fibonacci_sphere(500)
    c, l, r = cart2sph(x, y, z)
    vals = s20mod.eval(c, l, r)

    elements = triangulate(x,y,z)

    pts = np.array((x, y, z)).T * rel_rad
    write_vtk(os.path.join(VTK_DIR, 'test_s20rts.vtk'), pts, elements, vals, 'vs')


def test_topo():
    """
    Test to ensure that a vtk of the topography is written succesfully.
    :return:

    """
    topo = Topography()
    topo.read()

    x, y, z = skl.fibonacci_sphere(300)
    c, l, _ = cart2sph(x, y, z)

    vals = topo.eval(c, l)
    elements = triangulate(x, y, z)

    pts = np.array((x, y, z)).T
    write_vtk(os.path.join(VTK_DIR, 'topo.vtk'), pts, elements, vals, 'topo')

    north_pole = np.array([-4.228])
    south_pole = np.array([-0.056])
    random_point = np.array([0.103])

    np.testing.assert_almost_equal(topo.eval(0, 0), north_pole, decimal=DECIMAL_CLOSE)
    np.testing.assert_almost_equal(topo.eval(np.pi, 0), south_pole, decimal=DECIMAL_CLOSE)
    np.testing.assert_almost_equal(topo.eval(np.radians(90 - 53.833333), np.radians(76.500000)), random_point, decimal=DECIMAL_CLOSE)
