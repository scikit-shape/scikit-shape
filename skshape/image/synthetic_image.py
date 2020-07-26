"""Smooth functions used to define continuous synthetic images in 2d, 3d.

This module contains smooth functions with well-defined continuous derivatives,
which can be used to define synthetic continuous images.

Examples
--------
For example, we can create a continuous 3d image defined on the unit cube [0,1]^3.
This example image includes a sphere/ball and a cube with smooth edges that are
not jump discontinuities.

    >>> import numpy as np
    >>> from skshape.image.synthetic_image import BallFunction, BoxFunction, SyntheticImage
    >>> sphere = BallFunction( (0.25,0.25,0.25), 0.2, edge_width=0.03 )
    >>> cube = BoxFunction( (0.6,0.6,0.6), (0.85,0.85,0.85), edge_width=0.05 )
    >>> img = SyntheticImage( [sphere, cube] )
    >>> x = y = z = t = np.linspace(0.0,1.0,1000)
    >>> line_coords = np.array((x,y,z))
    >>> img_values = img( line_coords )
    >>> img_grad_values = img.gradient( line_coords )

"""

import numpy as np
from numpy import exp, pi
from itertools import product
from scipy.special import erf
from numba import vectorize, float64

@vectorize([float64(float64)])
def _polynomial_func(x):
    y = x**2
    return (abs(x)<1.) * (0.5 + x * (1.09375 + y * (-1.09375 + y * (0.65625 - 0.15625*y)))) + (x>=1.)

@vectorize([float64(float64)])
def _polynomial_deriv1(x):
    y = x**2
    return (abs(x)<1.) * (1.09375 + y * (- 3.28125 + y * (3.28125 - 1.09375*y)))

@vectorize([float64(float64)])
def _polynomial_deriv2(x):
    y = x**2
    return (abs(x)<1.) * (x * (-6.5625 + y * (13.125 - 6.5625*y)))

@vectorize([float64(float64)])
def _polynomial_deriv3(x):
    y = x**2
    return (abs(x)<1.) * (-6.5625 + y * (39.375 - 32.8125*y))


edge_functions = {

    "gaussian":  {"function": lambda x: 0.5 + 0.5*erf(2.*x),
                  "deriv1": lambda x: 2./(pi**.5*exp(4.*x**2)),
                  "deriv2": lambda x: -(16.*x)/(pi**.5*exp(4.*x**2)),
                  "deriv3": lambda x: (128.*x**2)/(pi**.5*exp(4.*x**2)) - 16./(pi**.5*exp(4.*x**2))
                  },

    "polynomial":{"function": _polynomial_func,
                  "deriv1": _polynomial_deriv1,
                  "deriv2": _polynomial_deriv2,
                  "deriv3": _polynomial_deriv3
                  },
    } # end of edge functions


class BallFunction(object):

    def __init__(self, center, radius, edge_width=0.1, edge_func_type="polynomial"):
        self.center = center
        self.radius = radius
        self.eps = edge_width
        self._edge_func   = edge_functions[ edge_func_type ]["function"]
        self._edge_deriv1 = edge_functions[ edge_func_type ]["deriv1"]
        self._edge_deriv2 = edge_functions[ edge_func_type ]["deriv2"]
        self._edge_deriv3 = edge_functions[ edge_func_type ]["deriv3"]

    def transition_width(self):
        return self.eps

    def _radial_coordinate(self,x,deriv=0):
        dim,n = x.shape

        # Compute the distance of all the given points to the center
        dist_vec = x.copy()
        for i in range(dim):
            dist_vec[i,:] -= self.center[i]  # dist_vec[i] = x[i] - center[i]
        dist = np.sum( dist_vec**2, 0 )**0.5 # dist = sqrt( sum_i( dist_vec[i]^2 ))

        if deriv == 0:  return (dist, dist_vec)

        deriv1 = dist_vec.copy()
        for i in range(dim):
            deriv1[i,:] /= dist

        if deriv == 1:  return (dist, dist_vec, deriv1)

        deriv2 = np.zeros((dim,dim,n))
        for i in range(dim):
            deriv2[i,i,:] = 1.0
            for j in range(dim):
                deriv2[i,j,:] -= deriv1[i,:] * deriv1[j,:]
                deriv2[i,j,:] /= dist

        if deriv == 2:  return (dist, dist_vec, deriv1, deriv2)

        deriv3 = np.zeros((dim,dim,dim,n))
        for i,j,k in product(range(dim),repeat=3):
            deriv3[i,j,k] = -( deriv1[i] * deriv2[j,k] +
                               deriv1[j] * deriv2[i,k] +
                               deriv1[k] * deriv2[i,j] ) / dist

        if deriv == 3:  return (dist, dist_vec, deriv1, deriv2, deriv3)

    def __call__(self,x):
        dist, dist_vec = self._radial_coordinate( x )
        normalized_coord = (self.radius - dist) / self.eps
        func_value = self._edge_func( normalized_coord )
        return func_value

    def gradient(self,x):
        dim = len( self.center )
        dist, dist_vec, dist_grad = self._radial_coordinate( x, deriv=1 )
        normalized_coord = (self.radius - dist) / self.eps
        edge_deriv = (-1.0/self.eps) * self._edge_deriv1( normalized_coord )

        grad = np.empty( x.shape )
        for i in range(dim):
            grad[i,:] = edge_deriv * dist_grad[i,:]
        return grad

    def hessian(self,x):
        dim,n = x.shape
        dist, dist_vec, dist_grad, dist_hess = self._radial_coordinate( x, deriv=2 )
        normalized_coord = (self.radius - dist) / self.eps

        edge_deriv1 = (-1.0/self.eps)    * self._edge_deriv1( normalized_coord )
        edge_deriv2 = ( 1.0/self.eps**2) * self._edge_deriv2( normalized_coord )

        hess = np.empty(( dim, dim, n ))
        for i,j in product(range(dim),repeat=2):
            hess[i,j,:] = edge_deriv1 * dist_hess[i,j,:] \
                          + edge_deriv2 * dist_grad[i,:] * dist_grad[j,:]
        return hess

    def third_deriv(self,x):
        dim,n = x.shape
        dist, dist_vec, dx, d2x, d3x = self._radial_coordinate( x, deriv=3 )

        normalized_coord = (self.radius - dist) / self.eps
        edge_deriv1 = (-1.0/self.eps)    * self._edge_deriv1( normalized_coord )
        edge_deriv2 = ( 1.0/self.eps**2) * self._edge_deriv2( normalized_coord )
        edge_deriv3 = (-1.0/self.eps**3) * self._edge_deriv3( normalized_coord )

        deriv3 = np.empty(( dim, dim, dim, n ))
        for i,j,k in product(range(dim),repeat=3):
            deriv3[i,j,k,:] = edge_deriv3 * dx[i,:] * dx[j,:] * dx[k,:] \
                              + edge_deriv1 * d3x[i,j,k,:] \
                              + edge_deriv2 * (d2x[i,k,:] * dx[j,:] + \
                                               d2x[j,k,:] * dx[i,:] + \
                                               d2x[i,j,:] * dx[k,:])
        return deriv3


class BoxFunction(object):

    def __init__(self, min_bound, max_bound, edge_width=0.1, edge_func_type="polynomial"):
        dim = len(min_bound)
        for i in range(0,dim):
            if min_bound[i] >= max_bound[i]:
                raise ValueError("min_bound should be smaller than max_bound!")
        self.min_bound = min_bound
        self.max_bound = max_bound
        self.eps = edge_width
        self._edge_func   = edge_functions[ edge_func_type ]["function"]
        self._edge_deriv1 = edge_functions[ edge_func_type ]["deriv1"]
        self._edge_deriv2 = edge_functions[ edge_func_type ]["deriv2"]
        self._edge_deriv3 = edge_functions[ edge_func_type ]["deriv3"]

    def transition_width(self):
        return self.eps

    def __call__(self,x):
        dim, n_pts = x.shape
        func_value = np.ones( n_pts )
        for i in range(0,dim):
            s = (1.0/self.eps) * x[i,:]
            minB = self.min_bound[i] / self.eps
            maxB = self.max_bound[i] / self.eps
            func_value *= self._edge_func( s - minB )
            func_value *= self._edge_func( maxB - s )

        return func_value

    def _indices_other_than(self,i,dim):
        l = list( range(0,dim) )
        l.remove(i)
        return l

    def gradient(self,x):
        dim, n_pts = x.shape
        x1 = np.zeros(( dim, n_pts ))
        x2 = np.zeros(( dim, n_pts ))
        for i in range(0,dim):
            x1[i,:] = (x[i,:] - self.min_bound[i]) / self.eps
            x2[i,:] = (self.max_bound[i] - x[i,:]) / self.eps

        func_val  = np.zeros(( dim, n_pts ))
        deriv_val = np.zeros(( dim, n_pts ))
        for i in range(0,dim):
            left_val  = self._edge_func( x1[i,:] )
            right_val = self._edge_func( x2[i,:] )
            left_deriv  =  (1./self.eps) * self._edge_deriv1( x1[i,:] )
            right_deriv = -(1./self.eps) * self._edge_deriv1( x2[i,:] )
            func_val[i,:]  = left_val * right_val
            deriv_val[i,:] = left_deriv * right_val + left_val * right_deriv

        # grad[i] = deriv[i] * Product_{j!=i} func_val[j]
        grad = np.zeros(( dim, n_pts ))
        for i in range(0,dim):
            grad[i,:] = deriv_val[i,:]
            for non_i in self._indices_other_than(i,dim): # loop indices other than i
                grad[i,:] *= func_val[non_i,:]

        return grad

    def hessian(self,x):
        dim, n_pts = x.shape
        x1 = np.zeros(( dim, n_pts ))
        x2 = np.zeros(( dim, n_pts ))
        for i in range(0,dim):
            x1[i,:] = (x[i,:] - self.min_bound[i]) / self.eps
            x2[i,:] = (self.max_bound[i] - x[i,:]) / self.eps

        func_val = np.zeros(( dim, n_pts ))
        deriv1 = np.zeros(( dim, n_pts ))
        deriv2 = np.zeros(( dim, n_pts ))
        for i in range(0,dim):
            left_val  = self._edge_func( x1[i,:] )
            right_val = self._edge_func( x2[i,:] )
            left_deriv1  =  (1./self.eps) * self._edge_deriv1( x1[i,:] )
            right_deriv1 = -(1./self.eps) * self._edge_deriv1( x2[i,:] )
            left_deriv2  =  (1./self.eps**2) * self._edge_deriv2( x1[i,:] )
            right_deriv2 =  (1./self.eps**2) * self._edge_deriv2( x2[i,:] )
            func_val[i,:] = left_val * right_val
            deriv1[i,:] = left_deriv1 * right_val + left_val * right_deriv1
            deriv2[i,:] = left_deriv2 * right_val + left_val * right_deriv2
            deriv2[i,:] += 2.0 * left_deriv1 * right_deriv1

        hess = np.zeros(( dim, dim, n_pts ))
        for i in range(0,dim):
            hess[i,i,:] = deriv2[i,:]
            for non_i in self._indices_other_than(i,dim):
                hess[i,i,:] *= func_val[non_i,:]
        for i in range(0,dim):
            for j in self._indices_other_than(i,dim):
                hess[i,j,:] = deriv1[i,:] * deriv1[j,:]
                if dim == 3:
                    k = dim - i - j
                    hess[i,j,:] *= func_val[k,:]

        return hess

    def third_deriv(self,x):
        dim, n_pts = x.shape
        x1 = np.zeros(( dim, n_pts ))
        x2 = np.zeros(( dim, n_pts ))
        for i in range(0,dim):
            x1[i,:] = (x[i,:] - self.min_bound[i]) / self.eps
            x2[i,:] = (self.max_bound[i] - x[i,:]) / self.eps

        func_val = np.zeros(( dim, n_pts ))
        deriv1 = np.zeros(( dim, n_pts ))
        deriv2 = np.zeros(( dim, n_pts ))
        deriv3 = np.zeros(( dim, n_pts ))
        for i in range(0,dim):
            left_val  = self._edge_func( x1[i,:] )
            right_val = self._edge_func( x2[i,:] )
            left_deriv1  =  (1./self.eps) * self._edge_deriv1( x1[i,:] )
            right_deriv1 = -(1./self.eps) * self._edge_deriv1( x2[i,:] )
            left_deriv2  =  (1./self.eps**2) * self._edge_deriv2( x1[i,:] )
            right_deriv2 =  (1./self.eps**2) * self._edge_deriv2( x2[i,:] )
            left_deriv3  =  (1./self.eps**3) * self._edge_deriv3( x1[i,:] )
            right_deriv3 = -(1./self.eps**3) * self._edge_deriv3( x2[i,:] )
            func_val[i,:] = left_val * right_val
            deriv1[i,:] = left_deriv1 * right_val + left_val * right_deriv1
            deriv2[i,:] = left_deriv2 * right_val + left_val * right_deriv2
            deriv2[i,:] += 2.0 * left_deriv1 * right_deriv1
            deriv3[i,:] = left_deriv3 * right_val + left_val * right_deriv3
            deriv3[i,:] += 3.0 * left_deriv2 * right_deriv1
            deriv3[i,:] += 3.0 * left_deriv1 * right_deriv2

        result = np.zeros(( dim, dim, dim, n_pts ))
        # First the derivs: I_xxx, I_yyy, I_zzz
        for i in range(0,dim):
            result[i,i,i,:] = deriv3[i,:]
            for non_i in self._indices_other_than(i,dim):
                result[i,i,i,:] *= func_val[non_i,:]
        # Twice-repeated derivs: I_xxy, I_xxz, I_xyx, ..., I_yzz
        for i in range(0,dim):
            for j in self._indices_other_than(i,dim):
                result[i,i,j,:] = deriv2[i,:] * deriv1[j,:]
                if dim == 3:
                    k = dim - i - j
                    result[i,i,j,:] *= func_val[k,:]
                result[i,j,i,:] = result[j,i,i,:] = result[i,i,j,:]
        # All derivs distinct: I_xyz, I_xzy, ..., I_zyx
        if dim == 3:
            for i in range(0,dim):
                for j in self._indices_other_than(i,dim):
                    k = dim - i - j
                    result[i,j,k,:] = deriv1[i,:] * deriv1[j,:] * deriv1[k,:]

        return result


class SyntheticImage(object):

    def __init__(self,objects):
        self._objects = objects

    def transition_width(self):
        min_eps = min((obj.eps for obj in self._objects))
        return min_eps

    def __call__(self,x):
        f = np.zeros( x.shape[1] )
        for obj_func in self._objects:
            f += obj_func(x)
        return f

    def gradient(self,x):
        grad = np.zeros( x.shape )
        for obj_func in self._objects:
            grad += obj_func.gradient(x)
        return grad

    def hessian(self,x):
        dim = x.shape[0]
        hess = np.zeros(( dim, dim, x.shape[1] ))
        for obj_func in self._objects:
            hess += obj_func.hessian(x)
        return hess

    def third_deriv(self,x):
        dim = x.shape[0]
        deriv3 = np.zeros(( dim, dim, dim, x.shape[1] ))
        for obj_func in self._objects:
            deriv3 += obj_func.third_deriv(x)
        return deriv3


images = [

    SyntheticImage( [ BallFunction( (0.5,0.5), 0.25, edge_width=0.2 ) ]
                    ),

    SyntheticImage( [ BallFunction( (0.30,0.30), 0.30 ),
                      BallFunction( (0.70,0.70), 0.30 ) ]
                    )
    ] # end of image list
