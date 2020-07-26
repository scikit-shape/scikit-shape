import numpy as np
from numba import jit
from numpy.random import rand, lognormal

###########################################################################

@jit(nopython = True)
def _local_to_global_loop(x0,x1,x2,s0,s1):
    n = x0.shape[1]
    result = np.empty((2,n))
    c0 = 1.0 - s0 - s1;   c1 = s0;   c2 = s1
    for i in range(0,n):
        result[0,i] = c0*x0[0,i] + c1*x1[0,i] + c2*x2[0,i]
        result[1,i] = c0*x0[1,i] + c1*x1[1,i] + c2*x2[1,i]
    return result


class ReferenceElement:

    def __init__(self):
        self._quadrature = None
        self._mesh_type = None

    def quadrature(self, degree=None, order=None):
        if (degree is None) and (order is None):
            if self._quadrature is not None:
                return self._quadrature
            else:
                order = 0

        if order is not None:
            if (self._quadrature is None) or (self._quadrature.order() != order):
                from ...numerics.integration import Quadrature2d
                self._quadrature = Quadrature2d( None, order )
        else:
            if (self._quadrature is None) or (self._quadrature.degree() != degree):
                from ...numerics.integration import Quadrature2d
                self._quadrature = Quadrature2d( degree, None )

        return self._quadrature

    def local_to_global(self, s, mesh, mask=None, coarsened=False):
        x0, x1, x2 = mesh.coords()
        if mask is None:
            # result = (1.0-s[0]-s[1]) * x0 + s[0] * x1 + s[1] * x2
            s0, s1 = s
            result = _local_to_global_loop( x0, x1, x2, s0, s1 )
        else:
            result = (1.0-s[0]-s[1])*x0[:,mask] + s[0]*x1[:,mask] + s[1]*x2[:,mask]
        return result

    def _generate_random_triangles(self, n_grid, n_tri, diameter=1.0):
        min_side_factor = 0.5
        max_side_factor = 1.5

        n = min(n_grid) - 1.0
        boundary_size = (diameter * n_grid[0]/n, diameter * n_grid[1]/n)

        side_length = diameter / n

        x0 = side_length * max_side_factor + \
             (boundary_size[0] - 2*side_length*max_side_factor) * rand(n_tri)
        y0 = side_length * max_side_factor + \
             (boundary_size[1] - 2*side_length*max_side_factor) * rand(n_tri)

        min_side = min_side_factor * side_length
        max_side = max_side_factor * side_length
        side1 = min_side + (max_side - min_side) * rand(n_tri)
        side2 = min_side + (max_side - min_side) * rand(n_tri)

        theta0 = 2.0*np.pi * rand(n_tri)

        min_theta = np.pi / 6.0
        max_theta = 2.0 * np.pi / 3.0
        theta = (np.pi/3) * lognormal(0., 0.25, n_tri)

        small_idx = theta < min_theta
        small_theta = theta[ small_idx ]
        large_idx = theta > max_theta
        large_theta = theta[ large_idx ]
        theta[ small_idx ] = 2*min_theta - small_theta
        theta[ large_idx ] = 2*max_theta - large_theta

        x1,y1 = x0 + side1 * np.cos(theta0), y0 + side1 * np.sin(theta0)

        theta += theta0
        x2,y2 = x0 + side2 * np.cos(theta), y0 + side2 * np.sin(theta)

        vertices = np.vstack(( np.hstack((x0,x1,x2)),
                               np.hstack((y0,y1,y2)) ))
        triangles = np.vstack(( np.arange(n_tri),
                                np.arange(n_tri,2*n_tri),
                                np.arange(2*n_tri,3*n_tri) ))

        return (triangles, vertices)

    def random_elements(self, grid_size, diameter):
        """Returns a Domain2d mesh made up of random elements about pixel size.

        Given a grid specified by grid_size and diameter, this function
        creates and returns a Domain2d mesh composed of random elements
        scattered around the physical domain defined by the grid.
        All the random elements are roughly half the size of a single pixel
        of the grid, about 0.5 * (diameter / min(grid_size))**2.

        Parameters
        ----------
        grid_size : tuple of two integers
            A pair of two integers that gives the size or resolution
            of the grid. By grid, we mean a set of evenly-spaced points,
            such as the pixels in an image grid.
        diameter : float
            The length of the 'shortest' side of the grid, the shortest
            edge of the rectangle

        Returns
        -------
        mesh: Domain2d object
            An instance of a Domain2d class made up of min(grid_size)
            random elements, each of which is about the size of a pixel
            of the grid specified by grid_size or diameter.
        """
        if self._mesh_type is None:
            from ...geometry.domain import Domain2d
            self._mesh_type = Domain2d

        n_tri = min( grid_size )

        tri, vtx = self._generate_random_triangles( grid_size, n_tri, diameter )
        mesh = self._mesh_type( triangles=tri, vertices=vtx )

        return mesh

    ## def interpolate(self,u,s):

    ##     if u.ndim == 1: # interpolation of scalar functions
    ##         n = len(u)
    ##         result = (1-s) * u
    ##         result[0:n-1] = result[0:n-1] + s * u[1:n]
    ##         result[n-1] = result[n-1] + s * u[0]

    ##     elif u.ndim == 2: # interpolation of vector functions
    ##         n = u.shape[1]
    ##         result = (1-s) * u
    ##         result[:,0:n-1] = result[:,0:n-1] + s * u[:,1:n]
    ##         result[:,n-1] = result[:,n-1] + s * u[:,0]

    ##     elif u.ndim == 3: # interpolation of matrix-valued functions
    ##         n = u.shape[2]
    ##         result = (1-s) * u
    ##         result[:,:,0:n-1] = result[:,:,0:n-1] + s * u[:,:,1:n]
    ##         result[:,:,n-1] = result[:,:,n-1] + s * u[:,:,0]

    ##     else:
    ##         raise ValueError, \
    ##               'Cannot interpolate arrays with dimensions higher than three!'

    ##     return result

    ## def eval_basis_fct(self,quad):
    ##     values = np.vstack( (quad.points(), 1.0-quad.points()) )
    ##     return values

    ## def eval_basis_fct_products(self,quad):

    ##     phi0 = quad.points()
    ##     phi1 = 1.0 - phi0
    ##     phi0_phi1 = phi0*phi1

    ##     values = np.zeros((2,2,len(phi0)))
    ##     values[0,0,:] = phi0**2
    ##     values[0,1,:] = phi0_phi1
    ##     values[1,0,:] = phi0_phi1
    ##     values[1,1,:] = phi1**2
    ##     return values
