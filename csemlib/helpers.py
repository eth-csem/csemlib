#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Library loading helper.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2014
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lgpl.html)
"""
from __future__ import absolute_import, division, print_function

import ctypes as C
import glob
import inspect
import os

import numpy as np


LIB_DIR = os.path.join(os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe()))), "lib")
print(LIB_DIR)
cache = []


def load_lib():
    if cache:  # pragma: no cover
        return cache[0]
    else:
        # Enable a couple of different library naming schemes.
        possible_files = glob.glob(os.path.join(LIB_DIR, "csemlib*.so"))
        if not possible_files:  # pragma: no cover
            raise ValueError("Could not find suitable csemlib shared "
                             "library.")
        filename = possible_files[0]
        lib = C.CDLL(filename)

        # A couple of definitions.
        lib.centroid.restype = C.c_void_p
        lib.centroid.argtypes = [
            C.c_int,
            C.c_int,
            C.c_int,
            np.ctypeslib.ndpointer(dtype=np.int64, ndim=2,
                                   flags=['C_CONTIGUOUS']),
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=2,
                                   flags=['C_CONTIGUOUS']),
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=2,
                                   flags=['C_CONTIGUOUS'])]

        lib.s20eval_grid.restype = C.c_void_p
        lib.s20eval_grid.argtypes = [
            C.c_int,
            C.c_int,
            C.c_int,
            C.c_int,
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1,
                                   flags=['C_CONTIGUOUS']),
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1,
                                   flags=['C_CONTIGUOUS']),
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1,
                                   flags=['C_CONTIGUOUS']),
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1,
                                   flags=['C_CONTIGUOUS']),
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1,
                                   flags=['C_CONTIGUOUS']),
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1,
                                   flags=['C_CONTIGUOUS']),
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1,
                                   flags=['C_CONTIGUOUS']),
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=3,
                                   flags=['C_CONTIGUOUS'])]



        cache.append(lib)
        return lib
