import os

import numpy as np
import pytest
import xarray
from meshpy.tet import MeshInfo, build, Options

import csemlib
import csemlib.background.skeleton as skl
import csemlib.models.crust as crust
import csemlib.models.one_dimensional as m1d
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

    true = np.array([
        [-1.01877232e-02, -1.01877232e-02, -1.01877232e-02, -1.01877232e-02, -1.01877232e-02,
         -1.01877232e-02, -1.01877232e-02, -1.01877232e-02, -1.01877232e-02, -1.01877232e-02],
        [-1.44056273e-02, 2.96664697e-02, 2.92642415e-02, 1.61460041e-02, -1.57275509e-02,
         7.13132098e-03, 2.90914878e-02, 2.99952254e-02, 5.79711610e-03, -1.44056273e-02],
        [3.37437506e-02, 2.04684792e-02, -1.24400607e-02, 2.72054351e-03, 8.33735766e-03,
         1.18519683e-02, 5.28123669e-03, 2.79334604e-02, 1.14312565e-02, 3.37437506e-02],
        [1.88464920e-02, -2.84314823e-03, 9.61633282e-03, 3.41507489e-02, 2.75727421e-02,
         1.68134034e-02, -1.60801974e-02, 3.64814426e-02, -4.61877955e-03, 1.88464920e-02],
        [3.49627974e-02, -2.25299414e-02, 5.38457115e-03, -1.28656351e-02, 2.23747856e-02,
         1.37116499e-02, -1.02294856e-02, -9.12301242e-03, 4.90924855e-03, 3.49627974e-02],
        [1.52122538e-02, 1.95654151e-02, -1.82730716e-03, 1.83242680e-03, -3.33209386e-02,
         2.42266632e-02, -2.14003047e-02, 4.65260346e-03, 3.98520761e-02, 1.52122538e-02],
        [1.77482107e-03, 1.45018273e-02, -2.46039369e-02, 3.74249736e-02, -6.59335407e-03,
         1.66440321e-02, -2.50129693e-02, -1.12087136e-02, 2.13203960e-02, 1.77482107e-03],
        [-1.76785861e-02, -4.01646331e-04, -2.15678403e-02, -2.20824982e-02, -1.08647419e-02,
         -2.65258612e-03, -3.65854079e-02, -1.95070464e-03, 1.47419745e-02, -1.76785861e-02],
        [1.03761537e-02, 1.48621690e-02, 1.61364041e-02, 2.67424633e-02, -1.33043420e-02,
         -2.34031725e-02, 5.95206701e-04, -4.95703024e-03, -9.53130089e-05, 1.03761537e-02],
        [5.83822775e-03, 5.83822775e-03, 5.83822775e-03, 5.83822775e-03, 5.83822775e-03,
         5.83822775e-03, 5.83822775e-03, 5.83822775e-03, 5.83822775e-03, 5.83822775e-03]])

    mod = s20.S20rts()
    mod.read()

    size = 10
    col = np.linspace(0, np.pi, size)
    lon = np.linspace(0, 2 * np.pi, size)
    cols, lons = np.meshgrid(col, lon)
    rad = mod.layers[0]

    vals = mod.eval(cols, lons, rad).reshape(size, size).T
    dat = xarray.DataArray(vals, dims=['lat', 'lon'], coords=[90 - np.degrees(col), np.degrees(lon)])
    np.testing.assert_almost_equal(dat.values, true, decimal=DECIMAL_CLOSE)


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
    c, l, _ = cart2sph(x, y, z)
    vals = s20mod.eval(c, l, rad)

    elements = triangulate(x,y,z)

    pts = np.array((x, y, z)).T * rel_rad
    write_vtk(os.path.join(VTK_DIR, 'test_s20rts.vtk'), pts, elements, vals, 'vs')


def test_s20rts_out_of_bounds():
    mod = s20.S20rts()

    with pytest.raises(ValueError):
        mod.find_layer_idx(3200)

    with pytest.raises(ValueError):
        mod.find_layer_idx(6204)

    with pytest.raises(ValueError):
        mod.eval(0, 0, 7000)


def test_topo():
    """
    Test to ensure that a vtk of the topography is written succesfully.
    :return:

    """
    topo = Topography()
    topo.read()

    x, y, z = skl.fibonacci_sphere(10000)
    c, l, _ = cart2sph(x, y, z)

    vals = topo.eval(c, l, param='topo')
    elements = triangulate(x, y, z)

    pts = np.array((x, y, z)).T
    write_vtk(os.path.join(VTK_DIR, 'topo.vtk'), pts, elements, vals, 'topo')

    north_pole = np.array([-4.228])
    south_pole = np.array([-0.056])
    random_point = np.array([0.103])

    np.testing.assert_almost_equal(topo.eval(0, 0, param='topo'), north_pole, decimal=DECIMAL_CLOSE)
    np.testing.assert_almost_equal(topo.eval(np.pi, 0, param='topo'), south_pole, decimal=DECIMAL_CLOSE)
    np.testing.assert_almost_equal(topo.eval(np.radians(90 - 53.833333), np.radians(76.500000), param='topo'), random_point, decimal=DECIMAL_CLOSE)
