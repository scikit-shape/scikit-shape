import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from numba import jit


def _flattened_grid_indices(resolution):
    m,n = resolution
    j = np.arange(n-1)
    nodes = np.empty( (4, (m-1)*(n-1)), dtype=int )
    for i in range(m-1):
        nodes[0,i*(n-1)+j] = i*n + j
    nodes[1,:] = nodes[0,:] +  n
    nodes[2,:] = nodes[0,:] + (n+1)
    nodes[3,:] = nodes[0,:] +  1
    return nodes


def load_vector(grid, f):
    m,n = grid.resolution
    hx,hy = grid.increment

    f_at_mid_pts = f[0:m-1,0:n-1] + f[0:m-1,1:n] + f[1:m,0:n-1] + f[1:m,1:n]
    f_at_mid_pts *= 0.25*0.25*hx*hy

    F = np.zeros_like(f)
    F[0:m-1,0:n-1]  = f_at_mid_pts
    F[0:m-1, 1:n ] += f_at_mid_pts
    F[ 1:m, 0:n-1] += f_at_mid_pts
    F[ 1:m,  1:n ] += f_at_mid_pts

    return F.flatten()


_local_stiff0 = np.array([[ 2.0,-2.0,-1.0, 1.0 ],
                          [-2.0, 2.0, 1.0,-1.0 ],
                          [-1.0, 1.0, 2.0,-2.0 ],
                          [ 1.0,-1.0,-2.0, 2.0 ]]) / 6.0

_local_stiff1 = np.array([[ 2.0, 1.0,-1.0,-2.0 ],
                          [ 1.0, 2.0,-2.0,-1.0 ],
                          [-1.0,-2.0, 2.0, 1.0 ],
                          [-2.0,-1.0, 1.0, 2.0 ]]) / 6.0

_local_mass = np.array([[ 4.0, 2.0, 1.0, 2.0 ],
                        [ 2.0, 4.0, 2.0, 1.0 ],
                        [ 1.0, 2.0, 4.0, 2.0 ],
                        [ 2.0, 1.0, 2.0, 4.0 ]]) / 36.0


def assemble_system_matrix(grid, alpha, beta):
    m,n = grid.resolution
    hx,hy = grid.increment
    N = m*n

    stiff = (hy/hx) * _local_stiff0 + (hx/hy) * _local_stiff1

    mass = hx*hy * _local_mass

    if np.isscalar(alpha):
        a = np.empty((m-1)*(n-1))
        a[:] = alpha
    else:
        a = 0.25 * (alpha[0:m-1,0:n-1] + alpha[0:m-1,1:n] + \
                    alpha[ 1:m, 0:n-1] + alpha[ 1:m, 1:n])
        a = a.flatten()

    if np.isscalar(beta):
        b = np.empty((m-1)*(n-1))
        b[:] = beta
    else:
        b = 0.25 * (beta[0:m-1,0:n-1] + beta[0:m-1,1:n] + \
                    beta[ 1:m, 0:n-1] + beta[ 1:m, 1:n])
        b = b.flatten()

    nodes = _flattened_grid_indices( (m,n) )

    A = csr_matrix((N,N))
    for i,I in enumerate(nodes):
        for j,J in enumerate(nodes):
            A = A + coo_matrix( (a*mass[i,j] + b*stiff[i,j], (I,J)),
                                shape=(N,N) ).tocsr()
    return A


@jit( nopython = True )
def _ellipt_matvec_loops(a,M,b,A,u,m,n):
    v = np.zeros((m,n))
    for i in range(m-1):
        for j in range(n-1):
            v1 = M[0,0] * u[ i, j] + M[0,3] * u[ i, j+1] + \
                 M[0,1] * u[i+1,j] + M[0,2] * u[i+1,j+1]
            v2 = A[0,0] * u[ i, j] + A[0,3] * u[ i, j+1] + \
                 A[0,1] * u[i+1,j] + A[0,2] * u[i+1,j+1]
            v[i,j] += a[i,j] * v1 + b[i,j] * v2

            v1 = M[3,0] * u[ i, j] + M[3,3] * u[ i, j+1] + \
                 M[3,1] * u[i+1,j] + M[3,2] * u[i+1,j+1]
            v2 = A[3,0] * u[ i, j] + A[3,3] * u[ i, j+1] + \
                 A[3,1] * u[i+1,j] + A[3,2] * u[i+1,j+1]
            v[i,j+1] += a[i,j] * v1 + b[i,j] * v2

            v1 = M[1,0] * u[ i, j] + M[1,3] * u[ i, j+1] + \
                 M[1,1] * u[i+1,j] + M[1,2] * u[i+1,j+1]
            v2 = A[1,0] * u[ i, j] + A[1,3] * u[ i, j+1] + \
                 A[1,1] * u[i+1,j] + A[1,2] * u[i+1,j+1]
            v[i+1,j] += a[i,j] * v1 + b[i,j] * v2

            v1 = M[2,0] * u[ i, j] + M[2,3] * u[ i, j+1] + \
                 M[2,1] * u[i+1,j] + M[2,2] * u[i+1,j+1]
            v2 = A[2,0] * u[ i, j] + A[2,3] * u[ i, j+1] + \
                 A[2,1] * u[i+1,j] + A[2,2] * u[i+1,j+1]
            v[i+1,j+1] += a[i,j] * v1 + b[i,j] * v2

    return v.reshape(m*n)


def ellipt_matvec(grid, alpha, beta, u):
    m,n = grid.resolution
    hx,hy = grid.increment
    u = u.reshape((m,n))

    if np.isscalar(alpha):
        a = np.empty((m-1,n-1))
        a[:] = alpha
    else:
        a = 0.25 * (alpha[0:m-1,0:n-1] + alpha[0:m-1,1:n] + \
                    alpha[ 1:m, 0:n-1] + alpha[ 1:m, 1:n])

    if np.isscalar(beta):
        b = np.empty((m-1,n-1))
        b[:] = beta
    else:
        b = 0.25 * (beta[0:m-1,0:n-1] + beta[0:m-1,1:n] + \
                    beta[ 1:m, 0:n-1] + beta[ 1:m, 1:n])

    A = (hy/hx) * _local_stiff0 + (hx/hy) * _local_stiff1
    M = (hx*hy) * _local_mass

    v = _ellipt_matvec_loops(a,M,b,A,u,m,n)

    return v


@jit( nopython = True )
def _inv_diag_ellipt_matvec_loops(a,mass,b,stiff,u,m,n):
    d = np.zeros((m,n))
    v = np.empty((m,n))

    M0 = mass[0,0];  M1 = mass[1,1];  M2 = mass[2,2];  M3 = mass[3,3]
    A0 = stiff[0,0]; A1 = stiff[1,1]; A2 = stiff[2,2]; A3 = stiff[3,3]

    for i in range(m-1):
        for j in range(n-1):
            d[ i,  j ] += a[i,j] * M0 + b[i,j] * A0
            d[ i, j+1] += a[i,j] * M3 + b[i,j] * A3
            d[i+1, j ] += a[i,j] * M1 + b[i,j] * A1
            d[i+1,j+1] += a[i,j] * M2 + b[i,j] * A2

    for i in range(0,m):
        for j in range(0,n):
            v[i,j] = u[i,j] / d[i,j]

    return v.reshape(m*n)


def inv_diag_ellipt_matvec(grid, alpha, beta, u):
    m,n = grid.resolution
    hx,hy = grid.increment
    u = u.reshape((m,n))

    if np.isscalar(alpha):
        a = np.empty((m-1,n-1))
        a[:] = alpha
    else:
        a = 0.25 * (alpha[0:m-1,0:n-1] + alpha[0:m-1,1:n] + \
                    alpha[ 1:m, 0:n-1] + alpha[ 1:m, 1:n])

    if np.isscalar(beta):
        b = np.empty((m-1,n-1))
        b[:] = beta
    else:
        b = 0.25 * (beta[0:m-1,0:n-1] + beta[0:m-1,1:n] + \
                    beta[ 1:m, 0:n-1] + beta[ 1:m, 1:n])

    stiff = (hy/hx) * _local_stiff0 + (hx/hy) * _local_stiff1
    mass  = (hx*hy) * _local_mass

    v = _inv_diag_ellipt_matvec_loops(a,mass,b,stiff,u,m,n)
    return v


def solve_elliptic_pde(grid, alpha, beta, f, g=0.0):
    if grid.dim() != 2:
        raise ValueError("Grid should be two-dimensional!")
    if g != 0.0:
        raise ValueError("Nonhomogeneous Neumann boundary condition with g != 0.0 has not been implemented yet!")
        # rhs[:] += assemble_neumann_bc( grid, g )

    rhs = load_vector( grid, f )

    A = assemble_system_matrix( grid, alpha, beta )

    u = spsolve( A, rhs )

    u = u.reshape( grid.resolution )

    return u


###########################################################################

class ReferenceElement:

    def __init__(self):
        self._quadrature = None

    def quadrature(self, degree=None, order=None):
        raise NotImplementedError("This function has not been implemented.")

    def local_to_global(self, s, mesh, mask=None, coarsened=False):
        raise NotImplementedError("This function has not been implemented.")

    def interpolate(self,u,s):
        raise NotImplementedError("This function has not been implemented.")
