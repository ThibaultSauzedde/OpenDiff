import pytest
import numpy as np
import numpy.testing as npt
import scipy

import opendiff.operators as op


def allclose_sparse(A, B, atol=1e-8):
    # If you want to check matrix shapes as well
    if np.array_equal(A.shape, B.shape) == 0:
        return False

    r1, c1, v1 = scipy.sparse.find(A)
    r2, c2, v2 = scipy.sparse.find(B)
    index_match = np.array_equal(r1, r2) & np.array_equal(c1, c2)

    if index_match == 0:
        return False
    else:
        return np.allclose(v1, v2, atol=atol)


def test_diff_removal_1d(macrolib_1d, datadir):
    macrolib, x_mesh = macrolib_1d
    dx = x_mesh[1:]-x_mesh[:-1]
    A = op.diff_removal_op(dx, macrolib)
    A_ref = scipy.sparse.load_npz(datadir / "r.npz")
    assert allclose_sparse(A, A_ref)


def test_diff_fission_1d(macrolib_1d, datadir):
    macrolib, x_mesh = macrolib_1d
    dx = x_mesh[1:]-x_mesh[:-1]
    A = op.diff_fission_op(dx, macrolib)
    A_ref = scipy.sparse.load_npz(datadir / "f.npz")
    assert allclose_sparse(A, A_ref)


def test_diff_scatering_1d(macrolib_1d, datadir):
    macrolib, x_mesh = macrolib_1d
    dx = x_mesh[1:]-x_mesh[:-1]
    A = op.diff_scatering_op(dx, macrolib)
    A_ref = scipy.sparse.load_npz(datadir / "s.npz")
    assert allclose_sparse(A, A_ref)


def test_diff_diffusion_1d(macrolib_1d, datadir):
    macrolib, x_mesh = macrolib_1d
    dx = x_mesh[1:]-x_mesh[:-1]
    A = op.diff_diffusion_op_1d(dx, macrolib, -1., -1.)
    A_ref = scipy.sparse.load_npz(datadir / "d.npz")
    # import sys
    # np.set_printoptions(threshold=sys.maxsize)
    # large_width = 400
    # np.set_printoptions(linewidth=large_width)
    # print("\n", A.toarray(), "\n")
    # print(A_ref.toarray())
    assert allclose_sparse(A, A_ref)
