from __future__ import division
import random


import numpy as np
cimport numpy as np

cimport cython
from libc.stdint cimport int32_t, int64_t

from cpython.array cimport array
from libc.math cimport sqrt, log, exp
from cython cimport integral

from cython.parallel cimport prange

def test(x):
    return np.empty(x)

@cython.boundscheck(False)
@cython.wraparound(False)
def sample_states(double[:,:] z_mat):

    cdef:
        unsigned int l,n,i
        np.ndarray[np.int_t, ndim=1] out_mat

    l = z_mat.shape[0]
    n = z_mat.shape[1]

    out_mat = np.empty(l, dtype=np.int)

    for i in range(l):

        out_mat[i] = switch_process_c(z_mat[i])

    return out_mat


# Todo: make this include the output too
@cython.boundscheck(False)
@cython.wraparound(False)
def markov_sample(double[:,:] m_mat, int s0, int N):

    cdef:
        int l, i, s
        int32_t[::1] out_mat = np.empty(N,dtype=np.int32)

    s = s0
    for i in range(N):
        s = switch_process_c(m_mat[s])
        out_mat[i] = s

    return np.asarray(out_mat)

@cython.boundscheck(False)
@cython.wraparound(False)
def switch_count(integral[::1] states, integral[:,::1] c_mat):

    cdef:
        unsigned int i, l
        integral z0, z1

    l = states.shape[0]
    z0 = states[0]
    for i in range(1, l):
        z1 = states[i]
        c_mat[z0, z1] += 1
        z0 = z1

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int switch_process_c(double[:] prob_mass):

    cdef:
        double n = random.random()
        double p_thresh = 0.0

        unsigned int i, l = prob_mass.shape[0]

    for i in range(l):
        p_thresh += prob_mass[i]
        if n < p_thresh:
            break

    return i




@cython.boundscheck(False)
@cython.wraparound(False)
def sample_states_nogil(double[:,:] z_mat):

    cdef:
        int l, n, i
        np.ndarray[np.int_t, ndim=1] out_mat
        #long[:] out_mat
        double[:] randoms


    l = z_mat.shape[0]
    n = z_mat.shape[1]

    out_mat = np.empty(l, dtype=np.int)
    randoms = np.random.rand(l)

    for i in prange(l, nogil=True):

        out_mat[i] = switch_process_nogil(z_mat[i], randoms[i])

    return out_mat

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int switch_process_nogil(double[:] prob_mass, double n) nogil:

    cdef:
        double p_thresh = 0.0
        unsigned int i, l = prob_mass.shape[0]

    for i in range(l):
        p_thresh += prob_mass[i]
        if n < p_thresh:
            break

    return i

@cython.boundscheck(False)
@cython.wraparound(False)
def matrix_normalize_nogil(double[:,:] mat):
    """
    normalizes rows of matrices
    :param mat:
    :return:
    """
    cdef:
        int i, j, k
        int l = mat.shape[0], n = mat.shape[1]
        double s = 0.0

    for i in prange(l, nogil=True):
        s = 0.0

        for k in range(n):

            s += mat[i, k]

        for j in range(n):

            mat[i, j] /= s

    return mat

@cython.boundscheck(False)
@cython.wraparound(False)
def matrix_normalize_np(np.ndarray[dtype=np.float64_t, ndim = 2] mat):
    """
    normalizes rows of matrices
    :param mat:
    :return:
    """
    return mat / mat.sum(axis=1)[:, None]


@cython.boundscheck(False)
@cython.wraparound(False)
def matrix_normalize(double[:,:] mat):
    """
    normalizes rows of matrices
    :param mat:
    :return:
    """
    cdef:
        int i, j, k
        int l = mat.shape[0], n = mat.shape[1]
        double s = 0.0

    for i in range(l):
        s = 0.0

        for k in range(n):

            s += mat[i, k]

        for j in range(n):

            mat[i, j] /= s

    return mat

def gauss_dist_2d(double[:] point, double[:] mean, double[:,:] cov):

    cdef:
        double md, det, out
        double pi
        double pp[2]
        double mm[2]
        double inv[2][2]


    det = cov[0,0] * cov[1,1] - cov[0,1] * cov[1,0]
    pi = 3.141592653589793

    inv[0][0] = cov[1,1] / det
    inv[1][1] = cov[0,0] / det
    inv[1][0] = -cov[1,0] / det
    inv[0][1] = -cov[0,1] / det

    pp[0] = point[0] - mean[0]
    pp[1] = point[1] - mean[1]

    mm[0] = pp[0]*inv[0][0] + pp[1] * inv[1][0]
    mm[1] = pp[1]*inv[1][1] + pp[0] * inv[0][1]

    md = mm[0] * pp[0] + mm[1] * pp[1]
    md *= -.5

    out = exp(md) / (2*pi * det**.5)

    return out