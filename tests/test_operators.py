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
    A_ref = scipy.sparse.load_npz(datadir / "r_1d.npz")
    assert allclose_sparse(A, A_ref)


def test_diff_fission_1d(macrolib_1d, datadir):
    macrolib, x_mesh = macrolib_1d
    dx = x_mesh[1:]-x_mesh[:-1]
    A = op.diff_fission_op(dx, macrolib)
    A_ref = scipy.sparse.load_npz(datadir / "f_1d.npz")
    assert allclose_sparse(A, A_ref)


def test_diff_scatering_1d(macrolib_1d, datadir):
    macrolib, x_mesh = macrolib_1d
    dx = x_mesh[1:]-x_mesh[:-1]
    A = op.diff_scatering_op(dx, macrolib)
    A_ref = scipy.sparse.load_npz(datadir / "s_1d.npz")
    assert allclose_sparse(A, A_ref)


def test_diff_diffusion_1d(macrolib_1d, datadir):
    macrolib, x_mesh = macrolib_1d
    dx = x_mesh[1:]-x_mesh[:-1]
    A = op.diff_diffusion_op_1d(dx, macrolib, -1., -1.)
    A_ref = scipy.sparse.load_npz(datadir / "d_1d.npz")
    # import sys
    # np.set_printoptions(threshold=sys.maxsize)
    # large_width = 400
    # np.set_printoptions(linewidth=large_width)
    # print("\n", A.toarray(), "\n")
    # print(A_ref.toarray())
    assert allclose_sparse(A, A_ref)

def test_diff_removal_2d(macrolib_2d, datadir):
    macrolib, x_mesh, y_mesh = macrolib_2d
    dx = x_mesh[1:]-x_mesh[:-1]
    dy = y_mesh[1:]-y_mesh[:-1]
    surf = np.multiply.outer(dx, dy)
    surf_1d = surf.reshape(-1)
    A = op.diff_removal_op(surf_1d, macrolib)
    A_ref = scipy.sparse.load_npz(datadir / "r_2d.npz")
    assert allclose_sparse(A, A_ref)


def test_diff_fission_2d(macrolib_2d, datadir):
    macrolib, x_mesh, y_mesh = macrolib_2d
    dx = x_mesh[1:]-x_mesh[:-1]
    dy = y_mesh[1:]-y_mesh[:-1]
    surf = np.multiply.outer(dx, dy)
    surf_1d = surf.reshape(-1)
    A = op.diff_fission_op(surf_1d, macrolib)
    A_ref = scipy.sparse.load_npz(datadir / "f_2d.npz")
    assert allclose_sparse(A, A_ref)


def test_diff_scatering_2d(macrolib_2d, datadir):
    macrolib, x_mesh, y_mesh = macrolib_2d
    dx = x_mesh[1:]-x_mesh[:-1]
    dy = y_mesh[1:]-y_mesh[:-1]
    surf = np.multiply.outer(dx, dy)
    surf_1d = surf.reshape(-1)
    A = op.diff_scatering_op(surf_1d, macrolib)
    A_ref = scipy.sparse.load_npz(datadir / "s_2d.npz")
    assert allclose_sparse(A, A_ref)


def test_diff_diffusion_2d(macrolib_2d, datadir):
    macrolib, x_mesh, y_mesh = macrolib_2d
    dx = x_mesh[1:]-x_mesh[:-1]
    dy = y_mesh[1:]-y_mesh[:-1]
    A = op.diff_diffusion_op_2d(dx, dy, macrolib, 1., -1., 1., -1.)
    A_ref = scipy.sparse.load_npz(datadir / "d_2d.npz")
    # import sys
    # # np.set_printoptions(threshold=sys.maxsize)
    # np.set_printoptions(threshold=20, edgeitems=10, linewidth=140)
    # print("\n", A.toarray(), "\n")
    # print(A_ref.toarray())
    assert allclose_sparse(A, A_ref)

def test_diff_removal_3d(macrolib_3d, datadir):
    macrolib, x_mesh, y_mesh, z_mesh = macrolib_3d
    dx = x_mesh[1:]-x_mesh[:-1]
    dy = y_mesh[1:]-y_mesh[:-1]
    dz = z_mesh[1:]-z_mesh[:-1]
    surf = np.multiply.outer(dx, dy)
    vol = np.multiply.outer(surf, dz)
    vol_1d = vol.reshape(-1, order='F')
    A = op.diff_removal_op(vol_1d, macrolib)
    A_ref = scipy.sparse.load_npz(datadir / "r_3d.npz")
    assert allclose_sparse(A, A_ref)


def test_diff_fission_3d(macrolib_3d, datadir):
    macrolib, x_mesh, y_mesh, z_mesh = macrolib_3d
    dx = x_mesh[1:]-x_mesh[:-1]
    dy = y_mesh[1:]-y_mesh[:-1]
    dz = z_mesh[1:]-z_mesh[:-1]
    surf = np.multiply.outer(dx, dy)
    vol = np.multiply.outer(surf, dz)
    vol_1d = vol.reshape(-1, order='F')
    A = op.diff_fission_op(vol_1d, macrolib)
    A_ref = scipy.sparse.load_npz(datadir / "f_3d.npz")
    assert allclose_sparse(A, A_ref)


def test_diff_scatering_3d(macrolib_3d, datadir):
    macrolib, x_mesh, y_mesh, z_mesh = macrolib_3d
    dx = x_mesh[1:]-x_mesh[:-1]
    dy = y_mesh[1:]-y_mesh[:-1]
    dz = z_mesh[1:]-z_mesh[:-1]
    surf = np.multiply.outer(dx, dy)
    vol = np.multiply.outer(surf, dz)
    vol_1d = vol.reshape(-1, order='F')

    A = op.diff_scatering_op(vol_1d, macrolib)
    A_ref = scipy.sparse.load_npz(datadir / "s_3d.npz")
    assert allclose_sparse(A, A_ref)


def test_diff_diffusion_3d(macrolib_3d, datadir):
    macrolib, x_mesh, y_mesh, z_mesh = macrolib_3d
    dx = x_mesh[1:]-x_mesh[:-1]
    dy = y_mesh[1:]-y_mesh[:-1]
    dz = z_mesh[1:]-z_mesh[:-1]
    A = op.diff_diffusion_op_3d(dx, dy, dz, macrolib, 1., 0., 1., 0., 0., 0.)
    A_ref = scipy.sparse.load_npz(datadir / "d_3d.npz")
    # np.set_printoptions(threshold=20, edgeitems=10, linewidth=140,
    # formatter = dict( float = lambda x: "%.3g" % x ))
    # print("\n", A.toarray(), "\n")
    # print(A_ref.toarray())
    assert allclose_sparse(A, A_ref)