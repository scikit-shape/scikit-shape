import numpy as np
from numba import jit
from scipy.ndimage import map_coordinates, spline_filter, convolve
from scipy.ndimage.filters import sobel, gaussian_gradient_magnitude
from itertools import product
from ..image.synthetic_image import BoxFunction


_FUNC_ID_COUNTER = 0

class GeneralizedFunction(object):

    def __init__(self):
        global _FUNC_ID_COUNTER
        _FUNC_ID_COUNTER += 1
        self.id = _FUNC_ID_COUNTER

    def transition_width(self):
        return None

    def __call__(self):
        raise NotImplementedError("This function has not been implemented.")

    def gradient(self):
        raise NotImplementedError("This function has not been implemented.")

    def hessian(self):
        raise NotImplementedError("This function has not been implemented.")

    def third_deriv(self):
        raise NotImplementedError("This function has not been implemented.")


class BoxIndicatorFunction(GeneralizedFunction):

    def __init__(self, min_x, max_x, transition_width):
        super( BoxIndicatorFunction, self ).__init__()
        self._min_x = Xm = np.array( min_x )
        self._max_x = XM = np.array( max_x )
        self._eps = eps = transition_width
        self._f = BoxFunction( Xm - 0.5*eps, XM + 0.5*eps, eps )

    def transition_width():
        return self.eps

    def _outside_pts(self, x):
        outside = np.zeros( x.shape[1], dtype=bool ) # No pt is outside yet
        for xi, min_xi, max_xi in zip( x, self._min_x, self._max_x):
            outside |= (xi < min_xi) | (max_xi < xi)
        outside = np.nonzero( outside )[0]
        return outside

    def __call__(self, x):
        y = np.ones( x.shape[1] )
        k = self._outside_pts(x)  # indices of outside points
        y[k] = self._f( x[:,k] )
        return y

    def gradient(self, x):
        grad = np.zeros_like(x)
        k = self._outside_pts(x)  # indices of outside points
        grad[:,k] = self._f.gradient( x[:,k] )
        return grad

    def hessian(self, x):
        dim = x.shape[1]
        hess = np.zeros((dim, dim, n_pts))
        k = self._outside_pts(x)  # indices of outside points
        hess[:,:,k] = self._f.hessian( x[:,k] )
        return hess


class RadialFunction(GeneralizedFunction):
    def __init__(self, core_fct):
        super( RadialFunction, self ).__init__()
        self.core_fct = core_fct

    def core(self,r):
        return self.core_fct( r )

    def core_deriv1(self,r):
        return self.core_fct.deriv1( r )

    def core_deriv2(self,r):
        return self.core_fct.deriv2( r )

    def transition_width(self):
        try:
            return self.core_fct.transition_width()
        except Exception:
            return None

    def __call__(self,x):
        r = np.sqrt( np.sum( x**2, 0 ) )
        return self.core_fct( r )

    def gradient(self,x):
        r = np.sqrt( np.sum( x**2, 0 ) )
        dg = self.core_fct.deriv1(r) / r
        grad = np.vstack([ dg*xi for xi in x ])
        return grad

    def hessian(self,x):
        dim,n = x.shape
        r_sqr = np.sum( x**2, 0 )
        r = np.sqrt( r_sqr )
        dg = self.core_fct.deriv1(r) / r
        d2g = self.core_fct.deriv2(r) / r_sqr
        hess = np.empty((dim,dim,n))
        for i,j in product(range(dim),repeat=2):
            hess[i,j] = (d2g - dg/r_sqr) * x[i]*x[j]
        for i in range(dim):
            hess[i,i] += dg
        return hess


## Mesh function cache is used to cache function (or derivative)
## values at quadrature points. It consists of a dictionary
## for each (id(mesh),id(function)) key pair, and the dictionary
## or cache for this key pair maps the order of function/derivative
## to another dictionary which contains NumPy arrays of values
## corresponding to a given quadrature value. The mappings are
## as follows:
##
##     _mesh_function_cache: (id(mesh),id(function)) => dict1
##     dict1: order => dict2
##     dict2: quad point s => the values array
##
## Order would be
##     0 for the function,
##     1 for the first derivative,
##     2 for the second derivative.
## For a function of two variables f = f(x,y), order could be
## the derivative and the code number for the variable w.r.t.
## which the derivative is taken. For example, order = (2,(1,0))
## would denote the second derivative w.r.t. y first, then w.r.t. x.
## The quadrature point s could be a float, for a 1d element, or
## a pair of floats for a 2d element.



# TODO: We need to make sure that mesh function caches don't keep multiple
# copies of images when mesh functions are based on ImageFunctions.

_mesh_function_cache = {}
_mesh_function_cache_counter = {}

def _get_mesh_function_key(mesh, func):
    try:
        func_id = func.id
    except:
        func_id = id(func)
    key = ( mesh.id, func_id )
    return key

def _init_mesh_function_cache(mesh, func):
    global _mesh_function_cache
    key = _get_mesh_function_key( mesh, func )
    count = _mesh_function_cache_counter.get(key,0)
    _mesh_function_cache_counter[key] = count + 1

def _update_mesh_function_cache(mesh, func, s, order, new_value):
    global _mesh_function_cache
    key = _get_mesh_function_key( mesh, func )

    if not _mesh_function_cache.has_key( key ):
        timestamp, cache = mesh.timestamp, {order:{}}
        _mesh_function_cache[ key ] = ( timestamp, cache )
    else: # _mesh_function_cache has the key
        timestamp, cache = _mesh_function_cache[ key ]
        if timestamp != mesh.timestamp:
            timestamp, cache = mesh.timestamp, {order:{}}
            _mesh_function_cache[ key ] = ( timestamp, cache )
        elif not cache.has_key( order ):
            cache[order] = {}

    if not np.isscalar(s):  s = tuple(s)

    cache[order][s] = new_value.copy()

def _check_mesh_function_cache(mesh, func, s, order, mask=None):
    global _mesh_function_cache
    key = _get_mesh_function_key( mesh, func )
    if not _mesh_function_cache.has_key( key ):  return None

    timestamp, cache = _mesh_function_cache[ key ]
    if timestamp != mesh.timestamp:  return None

    order_cache = cache.get( order, None )
    if order_cache is None:  return None

    if not np.isscalar(s):  s = tuple(s)

    cached_value = order_cache.get( s, None )
    if cached_value is None:
        return None
    elif mask is None:
        return cached_value.copy()
    else:
        return cached_value[...,mask]

def _remove_from_mesh_function_cache(mesh, func):
    global _mesh_function_cache
    key = _get_mesh_function_key( mesh, func )
    _mesh_function_cache_counter[key] -= 1
    if _mesh_function_cache_counter[key] < 1:
        _mesh_function_cache_counter.pop( key )
        _mesh_function_cache.pop( key, None )


class MeshFunction(GeneralizedFunction):

    def __init__(self, func, caching=True, map_coords=True):
        #>>>>>>>>>>>>> TODO: CHECK
        caching = False
        #>>>>>>>>>>>>>
        super( MeshFunction, self ).__init__()
        self._func = func
        # self.mesh = mesh
        self._map_coords = map_coords
        if map_coords:
            # self._map_to_x = mesh.local_to_global
            self._caching_results = caching
        else: # not mapping coords
            # self._map_to_x = lambda s,p1,p2: s # mapping fct is identity
            self._caching_results = False
        if self._caching_results:
            pass
            # _init_mesh_function_cache( mesh, func )

    ## def __del__(self):
    ##     if self._caching_results:
    ##         _remove_from_mesh_function_cache( self.mesh, self._func )

    ## def clone(self, mesh=None):
    ##     if mesh is None:  mesh = self.mesh
    ##     return MeshFunction( self._func, mesh, caching=self._caching_results,
    ##                          map_coords=self._map_coords )

    def resolution(self):
        return self._func.resolution()

    def diameter(self):
        return self._func.diameter()

    def transition_width(self):
        try:
            return self._func.transition_width()
        except Exception:
            return None

    def __call__(self, mesh, s, mask=None, coarsened=False):
        f = self._func
        # mesh = self.mesh
        order = 0 # no derivatives, just the function value
        # Check if f is already computed and is available in cache.
        if self._caching_results and not coarsened:
            f_val = _check_mesh_function_cache( mesh, f, s, order, mask )
            if f_val is not None:  return f_val

        # f not available in cache, so compute from scratch & store in cache.
        # x = self._map_to_x( s, mask, coarsened )
        x = mesh.local_to_global( s, mask, coarsened )
        f_val = f(x)
        if self._caching_results and not coarsened and mask is None:
            _update_mesh_function_cache( mesh, f, s, order, f_val )
        return f_val

    def gradient(self, mesh, s, mask=None, coarsened=False):
        f = self._func
        # mesh = self.mesh
        order = 1 # the gradient values
        # Check if df is already computed and is available in cache.
        if self._caching_results and not coarsened:
            df = _check_mesh_function_cache( mesh, f, s, order, mask )
            if df is not None:  return df

        # df not available in cache, so compute from scratch & store in cache.
        # x = self._map_to_x( s, mask, coarsened )
        x = mesh.local_to_global( s, mask, coarsened )
        df = f.gradient(x)
        if self._caching_results and not coarsened and mask is None:
            _update_mesh_function_cache( mesh, f, s, order, df )
        return df

    def hessian(self, mesh, s, mask=None, coarsened=False):
        f = self._func
        # mesh = self.mesh
        order = 2 # the hessian values
        # Check if d2f is already computed and is available in cache.
        if self._caching_results and not coarsened:
            d2f = _check_mesh_function_cache( mesh, f, s, order, mask )
            if d2f is not None:  return d2f

        # d2f not available in cache, so compute from scratch & store in cache.
        # x = self._map_to_x( s, mask, coarsened )
        x = mesh.local_to_global( s, mask, coarsened )
        d2f = f.hessian(x)
        if self._caching_results and not coarsened and mask is None:
            _update_mesh_function_cache( mesh, f, s, order, d2f )
        return d2f

    def third_deriv(self, mesh, s, mask=None, coarsened=False):
        f = self._func
        # mesh = self.mesh
        order = 3 # the hessian values
        # Check if d3f is already computed and is available in cache.
        if self._caching_results and not coarsened:
            d3f = _check_mesh_function_cache( mesh, f, s, order, mask )
            if d3f is not None:  return d3f

        # d3f not available in cache, so compute from scratch & store in cache.
        # x = self._map_to_x( s, mask, coarsened )
        x = mesh.local_to_global( s, mask, coarsened )
        d3f = f.third_deriv(x)
        if self._caching_results and not coarsened and mask is None:
            _update_mesh_function_cache( mesh, f, s, order, d3f )
        return d3f


class AnisotropicMeshFunction(GeneralizedFunction):

    def __init__(self, func, caching=True, map_coords=True,
                 normal_type='pwlinear'):
        #>>>>>>>>>>>>> TODO: CHECK
        caching = False
        #>>>>>>>>>>>>>
        super( AnisotropicMeshFunction, self ).__init__()
        self._func = func
        # self.mesh = mesh
        self._normal_type = normal_type
        self._map_coords = map_coords
        if map_coords:
            # self._map_to_x = mesh.local_to_global
            self._caching_results = caching
        else: # not mapping coords
            # self._map_to_x = lambda s,p1,p2: s # mapping fct is identity
            self._caching_results = False
        if self._caching_results:
            pass
            # _init_mesh_function_cache( mesh, func )

    ## def clone(self, mesh=None):
    ##     if mesh is None:  mesh = self.mesh
    ##     return AnisotropicMeshFunction( self._func, mesh,
    ##                                     caching=self._caching_results,
    ##                                     map_coords=self._map_coords,
    ##                                     normal_type=self._normal_type )

    def resolution(self):
        return self._func.resolution()

    def diameter(self):
        return self._func.diameter()

    def transition_width(self):
        try:
            return self._func.transition_width()
        except Exception:
            return None

    def __call__(self, mesh, s, mask=None, coarsened=False):
        f = self._func
        # mesh = self.mesh
        order = 0 # no derivatives, just the function value
        # Check if f is already computed and is available in cache.
        if self._caching_results and not coarsened:
            f_val = _check_mesh_function_cache( mesh, f, s, order, mask )
            if f_val is not None:  return f_val

        # f not available in cache, so compute from scratch & store in cache.
        # x = self._map_to_x( s, mask, coarsened )
        x = mesh.local_to_global( s, mask, coarsened )
        normal = mesh.normals( s, mask, coarsened, smoothness=self._normal_type )
        f_val = f( x, normal )
        if self._caching_results and not coarsened and mask is None:
            _update_mesh_function_cache( mesh, f, s, order, f_val )
        return f_val

    def gradient(self, mesh, s, mask=None, coarsened=False, variable=0 ):
        f = self._func
        # mesh = self.mesh
        order = (1,variable) # the gradient values w.r.t. variable
        # Check if df is already computed and is available in cache.
        if self._caching_results and not coarsened:
            df = _check_mesh_function_cache( mesh, f, s, order, mask )
            if df is not None:  return df

        # df not available in cache, so compute from scratch & store in cache.
        # x = self._map_to_x( s, mask, coarsened )
        x = mesh.local_to_global( s, mask, coarsened )
        normal = mesh.normals( s, mask, coarsened, smoothness=self._normal_type )
        df = f.gradient( x, normal, variable )
        if self._caching_results and not coarsened and mask is None:
            _update_mesh_function_cache( mesh, f, s, order, df )
        return df

    def hessian(self, mesh, s, mask=None, coarsened=False, variable=(0,0) ):
        f = self._func
        # mesh = self.mesh
        order = (2,variable) # the hessian values w.r.t. variable
        # Check if d2f is already computed and is available in cache.
        if self._caching_results and not coarsened:
            d2f = _check_mesh_function_cache( mesh, f, s, order, mask )
            if d2f is not None:  return d2f

        # d2f not available in cache, so compute from scratch & store in cache.
        # x = self._map_to_x( s, mask, coarsened )
        x = mesh.local_to_global( s, mask, coarsened )
        normal = mesh.normals( s, mask, coarsened, smoothness=self._normal_type )
        d2f = f.hessian( x, normal, variable )
        if self._caching_results and not coarsened and mask is None:
            _update_mesh_function_cache( mesh, f, s, order, d2f )
        return d2f

    def third_deriv(self, mesh, s, mask=None, coarsened=False, variable=(0,0,0) ):
        f = self._func
        # mesh = self.mesh
        order = (3,variable) # the 3rd deriv values w.r.t. variable
        # Check if d3f is already computed and is available in cache.
        if self._caching_results and not coarsened:
            d3f = _check_mesh_function_cache( mesh, f, s, order, mask )
            if d3f is not None:  return d3f

        # d3f not available in cache, so compute from scratch & store in cache.
        # x = self._map_to_x( s, mask, coarsened )
        x = mesh.local_to_global( s, mask, coarsened )
        normal = mesh.normals( s, mask, coarsened, smoothness=self._normal_type )
        d3f = f.third_deriv( x, normal, variable )
        if self._caching_results and not coarsened and mask is None:
            _update_mesh_function_cache( mesh, f, s, order, d3f )
        return d3f


class CurvatureDependentMeshFunction(GeneralizedFunction):

    def __init__(self, func, caching=True, map_coords=True, normal_type='pwlinear'):
        #>>>>>>>>>>>>> TODO: CHECK
        caching = False
        #>>>>>>>>>>>>>
        super( CurvatureDependentMeshFunction, self ).__init__()
        self._func = func
        # self.mesh = mesh
        self._normal_type = normal_type
        self._map_coords = map_coords
        if map_coords:
            # self._map_to_x = mesh.local_to_global
            self._caching_results = caching
        else: # not mapping coords
            # self._map_to_x = lambda s,p1,p2: s # mapping fct is identity
            self._caching_results = False
        if self._caching_results:
            pass
            # _init_mesh_function_cache( mesh, func )

    ## def clone(self, mesh=None):
    ##     if mesh is None:  mesh = self.mesh
    ##     return CurvatureDependentMeshFunction( self._func, mesh,
    ##                                            caching=self._caching_results,
    ##                                            map_coords=self._map_coords )

    def resolution(self):
        return self._func.resolution()

    def diameter(self):
        return self._func.diameter()

    def transition_width(self):
        try:
            return self._func.transition_width()
        except Exception:
            return None

    def __call__(self, mesh, s, mask=None, coarsened=False):
        f = self._func
        # mesh = self.mesh
        order = 0 # no derivatives, just the function value
        # Check if f is already computed and is available in cache.
        if self._caching_results and not coarsened:
            f_val = _check_mesh_function_cache( mesh, f, s, order, mask )
            if f_val is not None:  return f_val

        # f not available in cache, so compute from scratch & store in cache.
        # x = self._map_to_x( s, mask, coarsened )
        x = mesh.local_to_global( s, mask, coarsened )
        normal = mesh.normals( s, mask, coarsened,
                               smoothness=self._normal_type )
        curvature = mesh.curvature( s, mask, coarsened )
        f_val = f( x, normal, curvature )
        if self._caching_results and not coarsened and mask is None:
            _update_mesh_function_cache( mesh, f, s, order, f_val )
        return f_val

# TODO: CACHING SHOULD BE TRUE BY DEFAULT

class MeshFunction2(GeneralizedFunction):

    def __init__(self, func, caching=True, map_coords=True):
        #>>>>>>>>>>>>> CHECK
        caching = False
        #>>>>>>>>>>>>>
        super( MeshFunction2, self ).__init__()
        self._func = func
        self._map_coords = map_coords
        if map_coords:
            ## self._map_to_x1 = mesh1.local_to_global
            ## self._map_to_x2 = mesh2.local_to_global
            self._caching_results = caching
        else: # not mapping coords
            ## self._map_to_x1 = lambda s,p1,p2: s # mapping fct is identity
            ## self._map_to_x2 = lambda s,p1,p2: s # mapping fct is identity
            self._caching_results = False
        if self._caching_results:
            _init_mesh_function_cache( mesh, func )

    ## def clone(self, mesh1=None, mesh2=None):
    ##     if mesh1 is None:  mesh1 = self.mesh1
    ##     if mesh2 is None:  mesh2 = self.mesh2
    ##     return AnisotropicMeshFunction2( self._func, mesh1, mesh2,
    ##                                      caching=self._caching_results,
    ##                                      map_coords=self._map_coords,
    ##                                      normal_type=self._normal_type )

    def resolution(self):
        return self._func.resolution()

    def diameter(self):
        return self._func.diameter()

    def transition_width(self):
        try:
            return self._func.transition_width()
        except Exception:
            return None

    def __call__(self, mesh1, mesh2, s1, s2, mask1=None, mask2=None,
                 coarsened1=False, coarsened2=False):
        f = self._func
        # mesh1,mesh2 = self.mesh1,self.mesh2
        order = 0 # no derivatives, just the function value
        # Check if f is already computed and is available in cache.
        if self._caching_results and not (coarsened1 or coarsened2):
            f_val = _check_mesh_function_cache( (mesh1,mesh2), f, (s1,s2),
                                                order, mask )
            if f_val is not None:  return f_val

        # f not available in cache, so compute from scratch & store in cache.
        # x1 = self._map_to_x1( s1, mask1, coarsened1 )
        # x2 = self._map_to_x2( s2, mask2, coarsened2 )
        x1 = mesh1.local_to_global( s1, mask1, coarsened1 )
        x2 = mesh2.local_to_global( s2, mask2, coarsened2 )
        f_val = f( x1, x2 )
        if self._caching_results and not coarsened1 and not coarsened2 \
               and mask1 is None and mask2 is None:
            _update_mesh_function_cache( (mesh1,mesh2), f, (s1,s2), order, f_val )
        return f_val

# TODO: CACHING SHOULD BE TRUE BY DEFAULT

class AnisotropicMeshFunction2(GeneralizedFunction):

    def __init__(self, func, caching=True, map_coords=True,
                 normal_type='pwlinear'):
        #>>>>>>>>>>>>> CHECK
        caching = False
        #>>>>>>>>>>>>>
        super( AnisotropicMeshFunction2, self ).__init__()
        self._func = func
        self._normal_type = normal_type
        self._map_coords = map_coords
        if map_coords:
            ## self._map_to_x1 = mesh1.local_to_global
            ## self._map_to_x2 = mesh2.local_to_global
            self._caching_results = caching
        else: # not mapping coords
            ## self._map_to_x1 = lambda s,p1,p2: s # mapping fct is identity
            ## self._map_to_x2 = lambda s,p1,p2: s # mapping fct is identity
            self._caching_results = False
        if self._caching_results:
            _init_mesh_function_cache( mesh, func )

    ## def clone(self, mesh1=None, mesh2=None):
    ##     if mesh1 is None:  mesh1 = self.mesh1
    ##     if mesh2 is None:  mesh2 = self.mesh2
    ##     return AnisotropicMeshFunction2( self._func, mesh1, mesh2,
    ##                                      caching=self._caching_results,
    ##                                      map_coords=self._map_coords,
    ##                                      normal_type=self._normal_type )

    def resolution(self):
        return self._func.resolution()

    def diameter(self):
        return self._func.diameter()

    def transition_width(self):
        try:
            return self._func.transition_width()
        except Exception:
            return None

    def __call__(self, mesh1, mesh2, s1, s2, mask1=None, mask2=None,
                 coarsened1=False, coarsened2=False):
        f = self._func
        # mesh1,mesh2 = self.mesh1,self.mesh2
        order = 0 # no derivatives, just the function value
        # Check if f is already computed and is available in cache.
        if self._caching_results and not (coarsened1 or coarsened2):
            f_val = _check_mesh_function_cache( (mesh1,mesh2), f, (s1,s2),
                                                order, mask )
            if f_val is not None:  return f_val

        # f not available in cache, so compute from scratch & store in cache.
        # x1 = self._map_to_x1( s1, mask1, coarsened1 )
        # x2 = self._map_to_x2( s2, mask2, coarsened2 )
        x1 = mesh1.local_to_global( s1, mask1, coarsened1 )
        x2 = mesh2.local_to_global( s2, mask2, coarsened2 )
        normal1 = mesh1.normals( s1, mask1, coarsened1, smoothness='pwlinear' )
        normal2 = mesh2.normals( s2, mask2, coarsened2, smoothness='pwlinear' )
        f_val = f( x1, x2, normal1, normal2 )
        if self._caching_results and not coarsened1 and not coarsened2 \
               and mask1 is None and mask2 is None:
            _update_mesh_function_cache( (mesh1,mesh2), f, (s1,s2), order, f_val )
        return f_val


class InnerProductFunction(GeneralizedFunction):
    def __init__(self, f=None, g=None):
        if (f is None) or (g is not None):
            raise NotImplementedError('f = None or g != None not supported yet')
        super( InnerProductFunction, self ).__init__()
        self._f = f

    def __call__(self, x,y):
        return np.sum( y * self._f(x), 0 )

    def gradient(self, x,y, variable=0):
        dim = x.shape[0]
        if variable == 0:
            G = self._f.gradient(x)
            return np.vstack(( np.sum(y*G[:,i], 0) for i in range(dim) ))
        elif variable == 1:
            return self._f(x)
        else:
            raise ValueError("Need variable = 0 or 1!")

    def hessian(self, x,y, variable=(0,0)):
        dim = x.shape[0]
        if variable == (0,0):
            D2f = self._f.hessian(x)
            H = np.empty((dim, dim, x.shape[1]))
            for i,j in product(range(dim),repeat=2):
                H[i,j] = np.sum(y * D2f[:,i,j], 0 )
        elif variable == (1,1):
            H = np.zeros((dim, dim, x.shape[1]))
        elif variable == (1,0):
            H = self._f.gradient(x)
        elif variable == (0,1):
            Df = self._f.gradient(x)
            H = np.empty((dim, dim, x.shape[1]))
            for i,j in product(range(dim),repeat=2):
                H[i,j] = Df[j,i]
        else:
            raise ValueError("Need variable = one of (0,0),(0,1),(1,0),(1,1)!")
        return H

    def third_deriv(self, x,y, variable=(0,0,0)):
        dim = x.shape[0]
        if variable == (0,0,0):
            D3f = self._f.third_deriv(x)
            D3 = np.empty((dim, dim, dim, x.shape[1]))
            for i,j,k in product(range(dim),repeat=3):
                D3[i,j,k] = np.sum(y * D3f[:,i,j,k], 0 )
        elif variable in [ (1,1,1), (1,1,0), (1,0,1), (0,1,1) ]:
            D3 = np.zeros((dim, dim, dim, x.shape[1]))
        elif variable == (1,0,0):
            D3 = self._f.hessian(x)
        else: # variable in [ (0,0,0), (0,0,1), (0,1,0) ]
            D2f = self._f.hessian(x)
            D3 = np.empty((dim, dim, dim, x.shape[1]))
            if variable == (0,0,1):
                for i,j,k in product(range(dim),repeat=3):
                    D3[i,j,k] = D2f[k,i,j]
            elif variable == (0,1,0):
                for i,j,k in product(range(dim),repeat=3):
                    D3[i,j,k] = D2f[j,i,k]
            else:
                raise ValueError("Need variable in (0,0,0),(0,0,1),...,(1,1,1)!")
        return D3


@jit(nopython = True)
def _linear_interpolate(f, x, outside_value=-1.0):
    """Fast linear interpolation of image f at given points x.

    This function computes the linearly interpolated image values for
    the given points x from the given image array f. For points outside
    the image domain, either outside_value or the nearest image value
    are assigned.

    Parameters
    ----------
    f : NumPy array
        A 2d array storing the pixel values of a grayscale image.
    x : NumPy array
        An array of shape (2,n) storing the (x,y) coordinated of n points.
    outside_value : float, optional
        If x contains points outside the image domain, their image values
        are not interpolated, they are assigned the outside_value.
        If outside_value is -1.0 (default value), then the points are
        assigned the nearest image values.

    Returns
    -------
    y : NumPy array
        Values of the interpolated image. If x has n points, then y is
        an array of length n.

    """

    if x.shape[0] != 2:
        raise ValueError("_linear_interpolate currently works only for x with size = (2,N)!")
    if f.ndim != 2:
        raise ValueError("Data f for interpolation should be a 2d array!")

    if outside_value == -1.0:
        assign_const_outside = False
    else:
        assign_const_outside = True

    max_i = f.shape[0] - 1
    max_j = f.shape[1] - 1
    factor = float( min( max_i, max_j ) )
    n = x.shape[1]
    y = np.empty( n )

    if assign_const_outside: # for points outside image domain
        value_00 = value_0n = value_n0 = value_nn = outside_value

    else: # if assigning the nearest pixel to outside regions at corners
        value_00 = f[ 0, 0 ]
        value_0n = f[ 0, max_j ]
        value_n0 = f[ max_i, 0 ]
        value_nn = f[ max_i, max_j ]

    for k in range(n):
        continuous_i = factor * x[0,k]
        continuous_j = factor * x[1,k]
        i0 = int( np.floor( continuous_i ) )
        j0 = int( np.floor( continuous_j ) )
        i1 = i0 + 1
        j1 = j0 + 1

        # if ((i0 < 0) || (j0 < 0) || (i1 > max_i) || (j1 > max_j)):
        #      y[k] = outside_value

        if i0 < 0:

            if j0 < 0:
                y[k] = value_00
            elif j1 > max_j:
                y[k] = value_0n
            else: # i0 < 0, 0 <= j <= max_j
                if assign_const_outside:
                    y[k] = outside_value
                else: # assign the nearest value
                    weight_j0 = j1 - continuous_j
                    weight_j1 = continuous_j - j0
                    y[k] = weight_j0 * f[ 0, j0 ] + weight_j1 * f[ 0, j1 ]

        elif i1 > max_i:

            if j0 < 0:
                y[k] = value_n0
            elif j1 > max_j:
                y[k] = value_nn
            else: # i1 > max_i, 0 <= j <= max_j
                if assign_const_outside:
                    y[k] = outside_value
                else: # assign the nearest value
                    weight_j0 = j1 - continuous_j
                    weight_j1 = continuous_j - j0
                    y[k] = weight_j0 * f[ max_i,j0 ] + weight_j1 * f[ max_i,j1 ]

        elif j0 < 0: # and  0 <= i <= max_i
            if assign_const_outside:
                y[k] = outside_value
            else: # assign the nearest value
                weight_i0 = i1 - continuous_i
                weight_i1 = continuous_i - i0
                y[k] = weight_i0 * f[ i0, 0 ] + weight_i1 * f[ i1, 0 ]

        elif j1 > max_j: # and  0 <= i <= max_i
            if assign_const_outside:
                y[k] = outside_value
            else: # assign the nearest value
                weight_i0 = i1 - continuous_i
                weight_i1 = continuous_i - i0
                y[k] = weight_i0 * f[ i0, max_j] + weight_i1 * f[ i1, max_j]

        else: # x point is inside the image domain

            weight_i0 = i1 - continuous_i
            weight_i1 = continuous_i - i0
            weight_j0 = j1 - continuous_j
            weight_j1 = continuous_j - j0
            y[k] = weight_i0 * weight_j0 * f[ i0, j0 ] + \
                   weight_i0 * weight_j1 * f[ i0, j1 ] + \
                   weight_i1 * weight_j0 * f[ i1, j0 ] + \
                   weight_i1 * weight_j1 * f[ i1, j1 ]

    return y


class ImageFunction(GeneralizedFunction):

    def __init__(self, image, order=1, derivatives=0, edge_width_in_pixels=None):
        super( ImageFunction, self ).__init__()
        if image.ndim > 2:
            raise ValueError("ImageFunction currently works for 2d images!")
        self._factor = min(image.shape) - 1.0
        self._order = order
        ndim = image.ndim
        self._I = self.pixels = image
        self._origin = np.zeros(ndim)
        self._domain = np.array([ (n-1.0)/self._factor for n in image.shape ])

        self._edge_width = 1.0 / (max(image.shape) - 1)
        if edge_width_in_pixels is not None:
            self._edge_width *= edge_width_in_pixels

        # Assign the image pixels or the corresponding interp. coefficients.
        if order < 2 or order > 5: # order of interpolation
            self._prefilter = True  # Data should be prefiltered for spline!
            self._I = image
        else: # 2 <= order <= 5
            self._prefilter = False  # Data should not be prefiltered for spline!
            self._I = spline_filter( image, order )

        # Compute the image gradient if 1 or more derivatives are requested.
        if derivatives < 1:
            self._dI = None
        else: # derivatives >= 1, so compute the image gradient dI.
            dI_shape = [ ndim ] + list( image.shape ) # = (d,nx,ny) in 2d
            self._dI = np.empty( dI_shape )
            if order < 2 or order > 5:
                for k in range(ndim):
                    sobel( image, k, self._dI[k], mode='nearest' )
            else:
                deriv_mask = (self._factor / 12.0) * \
                             np.array([[1.,4.,1.], [0.,0.,0.], [-1.,-4.,-1.]])
                deriv_mask = [ deriv_mask, deriv_mask.T ]
                for k in range(ndim):
                    convolve( self._I, deriv_mask[k], self._dI[k], mode='mirror' )

        # Compute the image hessian if 2 or more derivatives are requested.
        if derivatives < 2:
            self._d2I = None
        else: # derivatives >= 2, so compute the image hessian d2I
            d2I_shape = [ndim, ndim] + list(image.shape) # (d,d,nx,ny) in 2d
            self._d2I = np.empty( d2I_shape )
            if order < 2 or order > 5:
                for k,l in product(range(ndim),repeat=2):
                    sobel( image, k, self._dI[k], mode='nearest' )
            else:
                mask_xx = (self._factor**2 / 6.0) * \
                          np.array([[1.,4.,1.], [-2.,-8.,-2.], [1.,4.,1.]])
                mask_yy = mask_xx.T
                mask_xy = mask_yx = (self._factor**2 / 4.0) * \
                          np.array([[1.,0.,-1.],[0.,0.,0.],[-1.,0.,1.]])
                hess_mask = [ [ mask_xx, mask_xy ], [ mask_yx, mask_yy ] ]
                for k,l in product(range(ndim),repeat=2):
                    convolve( self._I, hess_mask[k][l], self._d2I[k,l], mode='mirror')

    def resolution(self):
        return self._I.shape

    def origin(self):
        return self._origin

    def domain(self):
        return self._domain

    def diameter(self):
        return np.max( self._domain - self._origin )

    def transition_width(self):
        return self._edge_width

    def __call__(self,x):
        if x.shape[1] == 0:
            y = np.empty(0)
        elif self._order == 1:
            y = _linear_interpolate( self._I, x )
        else:
            y = map_coordinates( self._I, self._factor*x, order=self._order,
                                 mode='nearest', prefilter=self._prefilter )
        return y

    def gradient(self,x):
        if self._order < 2:
            raise NotImplementedError("Need interpolation order > 1 for gradient!")
        if x.shape[1] == 0:
            return np.empty_like(x)
        dim, n = x.shape
        grad = np.empty((dim,n))
        scaled_x = self._factor * x
        for k in range(dim):
            map_coordinates( self._dI[k], scaled_x, grad[k],
                             order=self._order-1, mode='nearest',
                             prefilter=self._prefilter )
        return grad

    def hessian(self,x):
        if self._order < 3:
            raise NotImplementedError("Need interpolation order > 2 for hessian!")
        if x.shape[1] == 0:
            return np.empty_like(x)
        dim, n = x.shape
        hess = np.empty((dim,dim,n))
        scaled_x = self._factor * x
        for k,l in product(range(dim),repeat=2):
            map_coordinates( self._d2I[k,l], scaled_x, hess[k,l],
                             order=self._order-2, mode='nearest',
                             prefilter=self._prefilter )
        return hess

    def boundary(self):
        if self._I.ndim != 2:
            raise NotImplementedError("Boundary is defined only for 2d images!")
        from ..geometry.curve import Curve

        xmin,ymin = self._origin
        xmax = xmin + self._domain[0]
        ymax = ymin + self._domain[1]

        return Curve( np.array([[ xmin, xmin, xmax, xmax ],
                                [ ymin, ymax, ymax, ymin ]]),
                      adaptivity_parameters = None )


class MultiImageFunction(GeneralizedFunction):

    def __init__(self, images, order=1, derivatives=0):
        super( MultiImageFunction, self ).__init__()
        self._order = order
        self._I = self.pixels = images
        self._I_fct = [ ImageFunction(I, order, derivatives) for I in images ]
        self._origin = self._I_fct[0].origin().copy()
        self._domain = self._I_fct[0].domain().copy()

    def resolution(self):
        return [ image.shape for image in self._I ]

    def origin(self):
        return self._origin

    def domain(self):
        return self._domain

    def diameter(self):
        return np.max( self._domain - self._origin )

    def __call__(self,x):
        y = np.array([ I(x) for I in self._I_fct ])
        return y

    def gradient(self,x):
        if self._order < 2:
            raise NotImplementedError("Need interpolation order > 1 for gradient!")
        grad = np.array([ I.gradient(x) for I in self._I_fct ])
        return grad

    def hessian(self,x):
        if self._order < 3:
            raise NotImplementedError("Need interpolation order > 2 for hessian!")
        hess = np.array([ I.hessian(x) for I in self._I_fct ])
        return hess

    def boundary(self):
        if self._I[0].ndim != 2:
            raise NotImplementedError("Boundary is defined only for 2d images!")
        from geometry.curve import Curve

        xmin,ymin = self._origin
        xmax = xmin + self._domain[0]
        ymax = ymin + self._domain[1]

        return Curve( np.array([[ xmin, xmin, xmax, xmax ],
                                [ ymin, ymax, ymax, ymin ]]),
                      adaptivity_parameters = None )


class EdgeIndicatorFunction(GeneralizedFunction):

    def __init__(self, image, rho, sigma=1.0, derivatives=2 ):
        super( EdgeIndicatorFunction, self ).__init__()
        self.I = image
        if rho <= 0.0:  raise ValueError('Parameter rho has to be a positive scalar!')
        self.rho = rho
        self.sigma = sigma
        self._discrete_image = isinstance( image, np.ndarray )
        if self._discrete_image:
            self._create_interpolant( image, rho, sigma, derivatives )
            self._transition_width = 2.0 * sigma / (min(image.shape) - 1.0)
        else:
            try:
                self._transition_width = image.transition_width()
            except Exception:
                self._transition_width = 0.05

    def _create_interpolant(self, image, rho, sigma, derivatives):
        if image.ndim > 2:
            raise ValueError("Images of dim > 2 are not supported!")
        mag = min(image.shape) - 1.0 # Scaling for magnitude of the derivative
        dI = mag * gaussian_gradient_magnitude( image, sigma, mode='nearest' )
        interp_order = 3
        self._g = 1.0 / (1.0 + dI**2/rho**2)
        self._g_fct = ImageFunction( self._g, interp_order, derivatives )

    def transition_width(self):
        return self._transition_width

    def __call__(self, x):
        if x.shape[1] == 0:
            return np.empty(0)
        if self._discrete_image: # Then we use the interpolant to evaluate g.
            return self._g_fct(x)
        else: # The underlying image is a continuous function.
            DI = self.I.gradient(x)
            g = 1.0 / (1.0 + np.sum(DI**2,0)/self.rho**2)
            return g

    def gradient(self, x):
        if x.shape[1] == 0:  return np.empty_like(x)

        if self._discrete_image: # Then we use the interpolant to evaluate Dg.
            return self._g_fct.gradient(x)

        else: # The underlying image is a continuous function.
            dim = x.shape[0]
            rho = self.rho
            DI  = self.I.gradient(x)
            D2I = self.I.hessian(x)

            # Need g^2 = 1 / (1 + |DI|^2/rho^2)^2  (scaled with coefficient)
            sqr_coef = (-2./rho**2) / (1. + np.sum(DI**2,0)/rho**2)**2

            # Compute  Dg[k] = -2/rho^2 * g^2 * D2I[l,k] * DI[l]
            Dg = np.zeros( x.shape )
            for k,l in product(range(dim),repeat=2):
                Dg[k] += D2I[l,k] * DI[l]  # over all x points
            for k in range(dim):
                Dg[k] *= sqr_coef

            return Dg

    def hessian(self, x):
        if x.shape[1] == 0:
            return np.empty((x.shape[0], x.shape[0], x.shape[1]))

        if self._discrete_image: # Then we use the interpolant to evaluate D2g.
            return self._g_fct.hessian(x)

        else: # The underlying image is a continuous function.
            dim  = x.shape[0]
            size = x.shape[1]
            rho  = self.rho
            DI  = self.I.gradient(x)
            D2I = self.I.hessian(x)
            D3I = self.I.third_deriv(x)

            # Compute and store D2I[k,l]*DI[k]
            D2I_x_DI = np.zeros( x.shape )
            for k,l in product(range(dim),repeat=2):
                D2I_x_DI[l] += D2I[k,l] * DI[k]

            g = 1.0 / (1.0 + np.sum(DI**2,0)/rho**2)

            # Need g^2 = 1 / (1 + |DI|^2/rho^2)^2  (scaled with coefficient)
            sqr_coef = (2.0/rho**2) * g**2
            D2g = np.zeros( (dim,dim,size) )
            for m,k,l in product(range(dim),repeat=3):
                D2g[k,l] -= D3I[m,k,l] * DI[m] + D2I[m,k] * D2I[m,l]
            for k,l in product(range(dim),repeat=2):
                D2g[k,l] += (4*g/rho**2) * D2I_x_DI[k] * D2I_x_DI[l]
                D2g[k,l] *= sqr_coef

            return D2g


class UnitVectorFunction(GeneralizedFunction):

    def __init__(self, vector_field, order=1, derivatives=0 ):
        super( UnitVectorFunction, self ).__init__()
        if isinstance( vector_field, np.ndarray ) \
            or (isinstance( vector_field, list ) and
                isinstance( vector_field[0], np.ndarray )) \
            or (isinstance( vector_field, tuple ) and
                isinstance( vector_field[0], np.ndarray )):
            self.vec = vector_field
            vec_norm = np.sum(( vec_comp**2 for vec_comp in vector_field ))
            vec_norm = np.sqrt( vec_norm )
            unit_vec_field = [ vec_comp/vec_norm for vec_comp in vector_field ]
            zero_indx = np.nonzero( vec_norm == 0.0 )
            for k in range(len(unit_vec_field)): # Remove inf,nan b/c div by 0.
                unit_vec_field[k][ zero_indx ] = 0.0
            self._vec_fct = MultiImageFunction( unit_vec_field,
                                                order, derivatives )
            self._normalized = True
        else:
            self.vec = None
            self._vec_fct = vector_field
            self._normalized = False

    def __call__(self, x):
        if x.shape[1] == 0:
            return np.empty(0)
        if self._normalized:
            return self._vec_fct(x)
        else: # not normalized
            v = self._vec_fct(x)
            norm_v = np.sqrt( np.sum(( v_comp**2 for v_comp in v )) )
            for k in range(len(v)):
                v[k] /= norm_v
            return v

    def gradient(self, x):
        if x.shape[1] == 0:  return np.empty_like(x)

        if self._normalized:
            return self._vec_fct.gradient(x)

        else: # not normalized
            v = self._vec_fct(x)
            grad = self._vec_fct.gradient(x)
            norm_v_sqr = np.sum(( v_comp**2 for v_comp in v ))
            vdim,xdim = grad.shape[0:2]

            f = norm_v_sqr**(-0.5)
            df = [ -np.sum(v*grad[:,i], 0) * f**3 for i in range(xdim) ]

            for m in range(vdim):
                for i in range(xdim):
                    grad[m,i] *= f
                    grad[m,i] += v[m] * df[i]
            return grad

    def hessian(self, x):
        if x.shape[1] == 0:
            return np.empty((x.shape[0], x.shape[0], x.shape[1]))

        if self._normalized:
            return self._vec_fct.hessian(x)

        else: # not normalized
            v = self._vec_fct(x)
            grad = self._vec_fct.gradient(x)
            hess = self._vec_fct.hessian(x)
            norm_v_sqr = np.sum(( v_comp**2 for v_comp in v ))
            vdim,xdim = grad.shape[0:2]

            f = norm_v_sqr**(-0.5)
            f_cube = f**3
            df = [ -np.sum(v*grad[:,i], 0) * f_cube for i in range(xdim) ]
            d2f = np.empty((xdim, xdim, x.shape[1]))
            for i,j in product(range(xdim),repeat=2):
                d2f[i,j] = sum((grad[n,i] * grad[n,j] + v[n] * hess[n,i,j]
                                for n in range(vdim)))
                d2f[i,j] *= -f_cube
                d2f[i,j] += (3/f) * df[i]*df[j]

            for m in range(vdim):
                for i,j in product(range(xdim),repeat=2):
                    hess[m,i,j] = f * hess[m,i,j] + v[m] * d2f[i,j] + \
                                  df[i] * grad[m,j] + df[j] * grad[m,i]
            return hess
