"""Definitions of shape energies.

This module contains several shape energies that can be used to guide
an iterative shape optimization algorithm for segmentation. In such an
approach, initial curves (or surfaces in 3d) are placed in the image,
and these are deformed iteratively to converge on the actual region
boundaries in the image. The shape energy is used to guide the curves/surfaces
through the optimization. When the shape energy of a curve is minimized,
the optimal location in the image has been achieved.

"""

import numpy as np
from numba import jit
from itertools import product
from ..statistics import estimate_integration_errors as stat_estimate_integration_errors
from ...numerics.integration import integrate, adapt_integrate
from ...numerics.matrix import MatrixWithRankOneUpdates
from ...numerics.function import ImageFunction, MeshFunction, AnisotropicMeshFunction, CurvatureDependentMeshFunction, BoxIndicatorFunction
from ...numerics.marking import equidistribution_marking, fixed_ratio_marking
from ...geometry.curve_adaptivity import compute_data_error as curve_compute_data_error, compute_data_coarsening_error as curve_compute_data_coarsening_error
from ...geometry.curve_examples import curve_example
from ...geometry.domain import Domain2d


default_world_boundary = curve_example( 'square', 128 )


class ShapeEnergy(object):
    """
    Parent class inherited by all the ShapeEnergy child classes.
    It includes default implementations of the some of the shared methods,
    such as
    :meth:'skshape.image.segmentation.shape_energy.ShapeEnergy.shape_gradient_norm',
    :meth:'skshape.image.segmentation.shape_energy.ShapeEnergy.shape_derivative',
    :meth:'skshape.image.segmentation.shape_energy.ShapeEnergy.shape_is_minimized'.
    """

    def __init__(self):
        self._reset_cache() # This initializes the cache.

    def basin_width(self):
        return None

    def __call__(self, surface):
        raise NotImplementedError("This function has not been implemented.")

    def shape_gradient(self, surface):
        raise NotImplementedError("This function has not been implemented.")

    def shape_gradient_norm(self, surface, function_space='L2'):
        if surface.size() == 0:  return 0.0
        G = self.shape_gradient( surface )

        if function_space == 'L2':
            M = surface.FEM.mass_matrix( surface )
            sh_grad_norm = np.dot( G, M.solve(G) )**0.5

        elif function_space == 'H1':
            A = surface.FEM.stiffness_matrix( surface )
            sh_grad_norm = np.dot( G, A.solve(G) )**0.5

        elif function_space in ['H-1','H^-1']:
            A = surface.FEM.stiffness_matrix( surface )
            sh_grad_norm = np.dot( G, A*G )**0.5
        else:
            raise ValueError("function_space should be one of 'L2','H1','H^-1'.")

        return sh_grad_norm

    def shape_derivative(self, surface, velocity):
        if surface.size() == 0:  return 0.0
        # dJ(surf;V) = \int G V = \sum_i V_i \int G \phi_i = \sum_i V_i G_i
        return np.dot( velocity, self.shape_gradient(surface) )

    def shape_hessian(self, surface):
        raise NotImplementedError("This function has not been implemented.")

    def is_minimized(self, surface, parameters={}, history=None):
        abs_tol = parameters.get( 'absolute tolerance', 0.01 )
        norm_G = self.shape_gradient_norm( surface )
        return ( norm_G < abs_tol * np.sqrt( surface.surface_area() ) )

    def _get_from_cache(self, surface, key):
        cache = self._cache

        if cache['id'] != surface.id:
            # Maybe surface is a submesh of "mesh family" with the cache['id']
            if surface.id not in cache['submeshes'].keys():
                return None
            else: # surface.id is a submesh id, retrieve its data.
                submesh_cache = cache['submeshes'][ surface.id ]
                if submesh_cache['timestamp'] != surface.timestamp:
                    return None
                else: # submesh timestamp matches.
                    return submesh_cache.get( key )

        elif cache['timestamp'] != surface.timestamp: # but cache['id'] matched
            self._reset_cache( surface )
            return None

        else: # surface id and timestamp match those of the cache.
            return cache.get( key )

    def _put_in_cache(self, surface, key, value):
        if (self._cache['id'] != surface.id) or \
           (self._cache['timestamp'] != surface.timestamp):
            self._reset_cache( surface )
        self._cache[ key ] = value

    def _put_submesh_data_in_cache(self, surface, key, values):
        if (self._cache['id'] != surface.id) or \
           (self._cache['timestamp'] != surface.timestamp):
            self._reset_cache( surface )

        cache = self._cache['submeshes']
        for submesh,value in zip(surface.submeshes(), values):
            submesh_cache = cache[ submesh.id ]
            # If timestamps don't match, reset submesh_cache, then continue.
            if submesh_cache['timestamp'] != submesh.timestamp:
                submesh_cache.clear()
                submesh_cache['timestamp'] = submesh.timestamp
            submesh_cache[ key ] = value

    def _reset_cache(self, surface=None):
        if surface is None:
            self._cache = { 'id':None, 'timestamp':None, 'submeshes':{} }
        else:
            # TODO: if some of the submeshes haven't changed,
            # don't reset their cache data.
            submesh_cache = dict(( ( submesh.id, {'timestamp':submesh.timestamp} )
                                   for submesh in surface.submeshes() ))
            self._cache = { 'id': surface.id,
                            'timestamp': surface.timestamp,
                            'submeshes': submesh_cache }


#########################################################################
#####         The Piecewise Constant Region Energy                  #####
#########################################################################


class PwConstRegionEnergy(ShapeEnergy):
    """
    The piecewise constant energy approximates the multiphase segmentation
    energy for a curve family :math:`\Gamma = \\bigcup_k \Gamma_k`. The curves
    partition the image domain into regions :math:`\{ \Omega_k \}_k`.
    :math:`\Omega_k` is the region enclosed by the curve :math:`\Gamma_k`.
    :math:`\Omega_0` is the region left outside of all curves.
    For a given curve family :math:`\Gamma`, the energy is

    .. math::  J(\Gamma) = \mu \int_\Gamma d\Gamma + \sum_k \int_{\Omega_k} (I(x) - c_k)^2 dx,

    where :math:`I(x)` is the image intensity function, and
    :math:`c_k=\\frac{1}{|\Omega_k|} \int_{\Omega_k} I(x) dx` are the region
    averages of the image intensity function.
    
    The default version is the energy is multiphase, i.e. each curve
    :math:`\Gamma_k` encloses a separate region as in [Dogan2015A]_, but it can
    also be set to two phases (see [Chan2001]_, [Dogan2015B]_), then the union 
    of the regions bounded by :math:`\Gamma_k` with even indices represent the 
    background region, and those with odd indices represent the foreground region.

    Parameters
    ----------
    parameters : dict
        A dictionary of parameters used to initialize and setup the energy
        object. It has the following keys:
        
            'image', a NumPy array of doubles, storing the image pixels, needs
            to be specified if an image function is not given.
        
            'image function', an optional :class:`ImageFunction`, which can be
            internally defined using the image array if not provided.
        
            'domain averages', (optional) an array of region averages of image
            intensity, they are computed automatically if not provided.
        
            'number of phases', (optional) 2 or 'many', the default value is 'many'.
        
            'world boundary', (optional) a rectangle :class:Curve indicating
            the boundary of the image/world. The nodes should be in normalized
            coordinates, e.g. for a square-shaped image, the computational
            domain is a unit square, and the boundary of the unit square is
            world boundary. If not provided, it is inherited from the image.
        
            'domain integration method', (optional) the integration method for the
            domain integrals, one of the three options 'adaptive quadrature',
            'pixel summation', 'trapezoidal rule'. The default value is 'pixel
            summation'.
        
            'total integration method', (optional) the integration method for the
            of the image intensity function over all the image domain.,
            one of the three options 'adaptive quadrature', 'pixel summation',
            'trapezoidal rule'. The default value is 'trapezoidal rule'.
        
            'domain integral tol', (optional) integration error tolerance value,
            used by adaptive quadrature in the domains
        
            'surface integral tol' (optional) integration error tolerance value,
            used  by adaptive quadrature on the surfaces or curves.
        
            'use tight initial tol' (optional) boolean flag indicating whether
            to use specified integral tolerances, even at the beginning of
            shape optimization when accuracy of adaptive quadrature is not
            very important. The default value is false, initial tol is not
            tight, it is relaxed a higher integral tol.

    Notes
    -----
    .. [Chan2001] : Chan, T.F.; Vese, L.A. "Active contours without edges."
       *IEEE Transactions on Image Processing* 10(2), 266-277 (2001).
    .. [Dogan2015A] : Dogan, G. "An efficient curve evolution algorithm for multiphase
       image segmentation."
       In *International Workshop on Energy Minimization Methods in Computer Vision
       and Pattern Recognition*, 292-306, Springer, Cham (2015).
    .. [Dogan2015B] : Dogan, G. "Fast minimization of region-based active contours
       using the shape hessian of the energy."
       In *International Conference on Scale Space and Variational Methods in
       Computer Vision*, 307-319, Springer, Cham (2015).
    """

    def __init__(self, parameters):
        super( ChanVeseEnergy, self ).__init__()

        self._params = parameters

        self._pixels = pixels = parameters.get('image')
        self._image_func = image  = parameters.get('image function')

        if (pixels is None) and (image is None):
            raise ValueError('Either the image array or the image function ' +
                             'should be provided with the parameters!')
        elif image is None:
            self._image_func = image = ImageFunction(pixels,3,2)

        elif pixels is None:
            try:
                self._pixels = image.pixels
            except AttributeError:
                pass

        self._I = MeshFunction( image )

        self._c = parameters.get('domain averages')
        if self._c is not None:  self._c = np.array(self._c) # if tuple or list

        self._mu = parameters.get('mu',0.0)

        self._n_phases = parameters.get('number of phases', 2 )
        if self._n_phases == 'two':  self._n_phases = 2

        # Set the world boundary.

        if 'world boundary' in parameters.keys():
            self._world_boundary = parameters['world boundary']
        else:
            try:
                self._world_boundary = image.boundary()
            except AttributeError:
                self._world_boundary = default_world_boundary.copy()

        bdary_coords = self._world_boundary.coords()
        min_coords = bdary_coords.min(1)
        max_coords = bdary_coords.max(1)
        self._basin_width = min(max_coords - min_coords)

        # Set domain indicator function based on world boundary.

        if pixels is None:
            transition = 0.01
        else: # Use # of pixels to determine transition length.
            n = min( pixels.shape )
            transition = 1.0 / (n-1)

        min_x = min_coords - 0.5*transition
        max_x = max_coords + 0.5*transition

        D = BoxIndicatorFunction( min_x, max_x, transition )
        self._D = MeshFunction( D )
        self._IxD = MeshFunction( lambda x: image(x) * D(x) )
        not_D = lambda x: 1.0 - D(x)
        not_D_grad = lambda x: -D.gradient(x)
        self._not_D = MeshFunction( not_D )
        self._not_D_grad = MeshFunction( not_D_grad )

        # Set domain integration method and integrate all image.

        if pixels is None:
            self._domain_integration_method = 'adaptive quadrature'
            self._total_integration_method  = 'adaptive quadrature'
        else: # Image has pixels, any integration method works.
            self._domain_integration_method = \
                parameters.get('domain integration method','pixel summation')
            self._total_integration_method = \
                parameters.get('total integration method','trapezoidal rule')

        # Initialize tolerances for adaptive integration.

        self._init_tols( parameters )
        self._total_integral = None # !!! Need here, otherwise we get an error !!!
        self._total_integral = self._integrate_all_image()


    def _init_tols(self, parameters):
        d_tol = parameters.get('domain integral tol')
        s_tol = parameters.get('surface integral tol')

        if (s_tol is None) or \
               ((d_tol is None) and
                (self._domain_integration_method == 'adaptive quadrature')):
            est_d_tol, est_s_tol, I = stat_estimate_integration_errors(self._image_func)

        if d_tol is None:
            if self._domain_integration_method == 'adaptive quadrature':
                d_tol = max( 1e-4, 50.0*est_d_tol )
            else: # domain_integration_method == 'pixel summation' or 'trap..'
                d_tol = 0.0
        elif d_tol <= 0.0:
            raise ValueError("'domain integral tol' in parameters must be a positive real number!")

        if s_tol is None:
            s_tol = max( 1e-4, 50.0*est_s_tol )
        elif s_tol <= 0.0:
            raise ValueError("'surface integral tol' in parameters must be a positive real number!")

        self._domain_integral_tol  = d_tol
        self._surface_integral_tol = s_tol

        use_tight_initial_tol = parameters.get('use tight initial tol', False )

        if use_tight_initial_tol or (self._domain_integral_tol == 0.0):
            self._current_tols = {'domain':  self._domain_integral_tol,
                                  'surface': self._surface_integral_tol }
        else:
            # The error tolerances used for integration: current_tols are
            # large initially (= factor x given_tol) for cheaper integration.
            # They can be reduced to given tolerances gradually if more
            # accuracy is needed.
            max_error = abs( self._world_boundary.area() )
            factor = np.floor( np.log2(max_error / self._domain_integral_tol) )
            factor = 2.0**factor
            self._current_tols = {'domain': factor * self._domain_integral_tol,
                                  'surface': factor * self._surface_integral_tol}

    def basin_width(self):
        return self._basin_width

#    def _compute_surface_integrals(self, surface):
#        int_IxD = self._get_from_cache( surface, 'surface integral' )
#        if int_IxD is not None:  return int_IxD
#
#        tol = self._current_tols['surface']  # tolerance for integration error
#        tol *= surface.surface_area() # Scale tol by surface area.
#
#        int_IxD, error = adapt_integrate( self._IxD, surface, tol=tol )
#
#        self._put_in_cache( surface, 'surface integral', int_IxD )
#        self._put_in_cache( surface, 'surface integral error', error )
#        return int_IxD

    @jit( nopython = True )
    def _sum_pixels_loop(pixels, mark, totals, counts):
        n0,n1 = pixels.shape
        for i in range(0,n0):
            for j in range(0,n1):
                totals[ mark[i,j] ] += pixels[i,j]
                counts[ mark[i,j] ] += 1

    def _sum_pixels(self, pixels, mark, n_regions=None):
        if n_regions is None:
            n_regions = mark.max() + 1

        totals = np.zeros( n_regions, dtype=pixels.dtype )
        counts = np.zeros( n_regions, dtype=int )

        PwConstRegionEnergy._sum_pixels_loop( pixels, mark, totals, counts )

        return totals, counts


    def _compute_domain_integrals(self, surface):
        integrals = self._get_from_cache( surface, 'domain integrals' )
        if integrals is not None:  return integrals

        # Get the image pixels, pixel size and check integration method.
        if self._pixels is None:
            self._domain_integration_method = 'adaptive quadrature'
        else: # determine pixel size from number of pixels.
            n = np.min( self._pixels.shape )
            pixel_size = 1.0 / (n-1)**2

        # Set the regions defined by the surface, two-phase or multi-phase.
        if self._n_phases == 2:
            regions = [ surface ]
        else:
            regions = surface.regions()

        n_regions = len(regions) + 1 # include also the exterior

        # Perform integration on all regions by pixel summation or quadrature.

        if self._domain_integration_method == 'pixel summation':
            origin, img_domain = self._image_func.origin(), self._image_func.domain()
            mark = np.empty( self._pixels.shape, dtype=int )
            mark[:] = n_regions - 1  # initialize pixel marks as exterior domain

            for k, region in enumerate(regions):
                interior = region.interior_domain( self._world_boundary )
                interior.mark_grid( mark, origin, img_domain, value=k )

            integrals, counts = self._sum_pixels( self._pixels, mark, n_regions )
            self._region_vols = pixel_size * counts
            integrals *= pixel_size
            errors = np.zeros_like( integrals )

        else: # domain_integration_method = 'adaptive quadrature'
            tol = self._current_tols['domain']
            total_integral = self._integrate_all_image()

            if self._n_phases == 2:
                vol0 = abs( surface.interior_volume( self._world_boundary ) )
                vol = [ vol0, (-vol0 + abs( self._world_boundary.volume() )) ]
            else:
                vol = surface.region_volumes( self._world_boundary )

            integrals, errors = np.empty(n_regions), np.empty(n_regions)
            for k, region in enumerate(regions):
                domain = region.interior_domain( self._world_boundary )
                integrals[k], errors[k] = \
                              adapt_integrate( self._I, domain, tol=vol[k]*tol )

            integrals[-1] = total_integral - np.sum( integrals[0:-1] )
            errors[-1] = self._total_integral_error + np.sum( errors[0:-1] )

        self._put_in_cache( surface, 'domain integrals', integrals )
        self._put_in_cache( surface, 'domain integral errors', errors )

        return integrals


    def _compute_domain_averages(self, surface):
        # Check if domain averages are already known and return if they are.
        if self._c is not None:  return self._c

        # Check if they have already been computed and they are in the cache.
        averages = self._get_from_cache( surface, 'domain averages' )
        if averages is not None:  return averages  # otherwise, compute below

        # Need the volumes of the regions to compute domain averages.
        if self._n_phases == 2:
            volume0 = abs( surface.interior_volume( self._world_boundary ) )
            volume1 = abs( self._world_boundary.volume() ) - volume0
            volumes = np.array([ volume0, volume1 ])
        else:
            volumes = surface.region_volumes( self._world_boundary )

        # TODO: This is a temporary solution when volume is zero.
        for k,volume in enumerate(volumes):
            if volume == 0.0:  volumes[k] = np.inf

        # Domain averages are not available; we need to compute them.
        integrals = self._compute_domain_integrals( surface )
        averages = integrals / volumes

        self._put_averages_in_cache( surface, averages )

        return averages

    def _put_averages_in_cache(self, surface, averages):
        self._put_in_cache( surface, 'domain averages', averages )
        if not surface.has_submeshes():  return
        # If the surface consists of multiple submeshes, then put their
        # (c_in,c_out) inner,outer average pairs in the cache too.

        average_pairs = []
        if len(averages) == 2: # We only have two phases.
            c_in,c_out = averages
            for mesh in surface.submeshes():
                c_pair = (c_in,c_out) if mesh.orientation() > 0 else (c_out,c_in)
                average_pairs.append( c_pair )

        else: # We have multiple phases.
            top_meshes, parents, children = surface.curve_hierarchy()
            for parent, c_in in zip(parents, averages):
                c_out = averages[parent] if parent is not None else averages[-1]
                average_pairs.append( (c_in,c_out) )

        self._put_submesh_data_in_cache(surface,'domain averages',average_pairs)


    def _integrate_all_image(self):
        if self._total_integral is not None:  return self._total_integral

        # Access the pixels of the image if necessary.
        if self._pixels is None:
            self._total_integration_method = 'adaptive quadrature'

        if self._total_integration_method in ['pixel summation','trapezoidal rule']:
            n = np.min( self._pixels.shape )
            pixel_size = 1.0 / (n-1)**2

        # Perform summation or quadrature depending on integration method.
        if self._total_integration_method == 'pixel summation':
            int_I = self._pixels.sum() * pixel_size
            self._total_integral_error = 0.0

        elif self._total_integration_method == 'trapezoidal rule':
            I = self._pixels
            int_I = I[1:-1,1:-1].sum() + \
                    ( I[0,1:-1].sum() + I[-1,1:-1].sum() + \
                      I[1:-1,0].sum() + I[1:-1,-1].sum() ) / 2.0 + \
                    ( I[0,0] + I[0,-1] + I[-1,0] + I[-1,-1] ) / 4.0
            int_I = pixel_size * int_I
            self._total_integral_error = 0.0

        elif self._total_integration_method == 'adaptive quadrature':
            vol = abs( self._world_boundary.volume() )
            tol = vol * self._current_tols['domain'] # integration tolerance
            domain = Domain2d( self._world_boundary )
            int_I, error = adapt_integrate( self._I, domain, tol=tol )
            self._total_integral_error = error

        else:
            raise ValueError("'Total integration method' should be one of 'pixel summation', 'trapezoidal rule', 'adaptive quadrature'!")

        self._total_integral = int_I

        return int_I


    def __call__(self, surface):
        # If surface is empty set, then return integration over image domain.
        if surface.size() == 0:
            vol = abs( self._world_boundary.volume() )
            int_I = self._integrate_all_image()
            if self._c is None:
                return (-0.5 * int_I**2 / vol)
            else:
                return (-int_I * self._c[-1] + 0.5*vol * self._c[-1]**2)

        # Compute the term for curve length / surface area.
        if self._mu == 0.0:  # weight for the surface area term
            energy = 0.0
        else:
            energy = self._mu * surface.surface_area()

        # Compute domain integrals and averages, then the energy.
        integrals = self._compute_domain_integrals( surface )
        averages = self._c  # known already, otherwise None

        if averages is None: # then we need to compute the averages.
            averages = self._compute_domain_averages( surface )
            energy = -0.5 * np.dot( averages, integrals )

        else: # averages known, specified at initialization of energy.
            if self._n_phases == 2:
                volume0 = abs( surface.interior_volume( self._world_boundary ) )
                volume1 = abs( self._world_boundary.volume() ) - volume0
                volumes = np.array([ volume0, volume1 ])
            else:
                volumes = surface.region_volumes( self._world_boundary )
            energy = -np.dot( averages, integrals ) \
                     + 0.5 * np.dot( averages**2, volumes )

        tol = self._params['surface integral tol'] * surface.area()
        outside_surface_area, error = adapt_integrate( self._not_D, surface, tol=tol )
        energy = energy + outside_surface_area

        return energy


    def _c_in_vector(self, surface, c_vec):
        indices = [0] + [ crv.size() for crv in surface.curve_list() ]
        indices = np.cumsum( indices )
        c_in = np.empty( indices[-1] )
        for k, c_val in enumerate(c_vec[:-1]):
            i1,i2 = indices[k:k+2]
            c_in[i1:i2] = c_val
        return c_in

    def _c_out_vector(self, surface, c_vec):
        top_curves, parents, children = surface.curve_hierarchy()
        parent = np.array( parents )
        parent[ top_curves ] = -1  # use last entry of c_vec for top_curves

        indices = [0] + [ crv.size() for crv in surface.curve_list() ]
        indices = np.cumsum( indices )
        c_out = np.empty( indices[-1] )
        for k, c_val in enumerate(c_vec[:-1]):
            i1,i2 = indices[k:k+2]
            c_out[i1:i2] = c_vec[ parent[k] ]
        return c_out

    def shape_gradient(self, surface, s=None, mask=None, coarsened=None, split=False, new_mu=None, FEM='scalar'):
        if surface.size() == 0:  return np.empty(0)

        if (mask is not None) or (coarsened is not None):
            raise NotImplementedError("This function does not work with given mask or coarsened parameters.")

        mu = self._mu if new_mu is None else new_mu

        c = self._compute_domain_averages( surface )

        if s is not None: # then we want shape gradient value at quad pt s

            if len(c) == 2: # just two phases/averages: in & out
                c_in, c_out = c
            else: # n_phases > 2
                c_in  = self._c_in_vector( surface, c )
                c_out = self._c_out_vector( surface, c )

            I = self._I( surface, s )
            inside_image = self._D( surface, s )

            # TODO: Go over orientations
            ## orientations = np.ones(surface.size())
            ## offset = 0
            ## for curve in surface.curve_list():
            ##     n = curve_size()
            ##     orientations[offset:offset+n] = curve.orientation()
            ##     offset += n

            grad = inside_image * (c_out - c_in) * (I - 0.5*(c_in + c_out))
            grad *= surface.orientation()

            K = surface.curvature(s)
            N = surface.normals(s)
            outside_image = self._not_D(surface,s)
            outside_grad  = self._not_D_grad(surface,s)
            grad += outside_image * K + np.sum( outside_grad * N, 0 )

            if mu != 0.0:   grad += mu * K

            return grad

        else: # we want the discretized vector:  G_i = dJ(surf;\phi_i)
            if not split:
                grad = self._get_from_cache( surface, 'shape gradient' )
                if grad is not None:  return grad

            M = surface.FEM.mass_matrix( surface )
            if FEM == 'scalar':
                sh_grad_fct = lambda mesh,s: self.shape_gradient(mesh,s,new_mu=0.0)
                f = surface.FEM.load_vector( surface, f=sh_grad_fct )
            elif FEM == 'vector':
                f = []
                for k in range(surface.dim_of_world()):
                    sh_grad_fct = lambda mesh,s: \
                                  self.shape_gradient(mesh,s,new_mu=0.0) \
                                  * mesh.normals(s,smoothness='pwlinear')[k]
                    f0 = surface.FEM.load_vector( surface, f=sh_grad_fct )
                    f.append( f0 )
            else:
                raise ValueError("Choice of FEM discretization is either 'scalar' or 'vector'.")

            if split:
                return (mu*M, f)
            else: # shape grad = mu*M*K + f
                if FEM != 'scalar':
                    raise ValueError("split == false option for shape gradient only allowed for FEM = 'scalar'.")
                grad = f
                if mu != 0.0:
                    grad += mu * (M * surface.curvature())
                if (new_mu is None) or (new_mu == self._mu):
                    self._put_in_cache( surface, 'shape gradient', grad )
                return grad

    def _mass_coef_func(self,x,N,K):
        threshold = self._hessian_threshold
        cache = self._cache

        c = cache['domain averages']
        coef1 = c[1] - c[0]
        coef0 = -0.5 * coef1 * (c[0] + c[1])

        I  = self._image_func(x)
        dI = self._image_func.gradient( x )

        # Result: coef0*K + coef1*I*K + coef1*<dI,N>
        result = coef1 * I  # result = coef1*I
        result += coef0     # result = coef0 + coef1*I
        result *= K         # result = coef0*K + coef1*I*K
        result += coef1 * np.sum( dI * N, 0 )
        if threshold is not None:
            threshold *= self._pixels.max() * abs(coef1)
            result[ result < threshold ] = threshold
        return result

    def shape_hessian(self, surface, full_hessian=None, threshold=None):
        raise NotImplementedError('Need to add the shape hessian for "not_D" part!!!')

        params = self._params

        if surface.size() == 0:  return np.empty((0,0))

        if full_hessian is not None:
            use_full_hessian = full_hessian
        else:
            use_full_hessian = params.get('use full Hessian',
                               params.get('use full hessian',False))
        if threshold is not None:
            self._hessian_threshold = threshold
        else:
            self._hessian_threshold = params.get('Hessian threshold',
                                      params.get('hessian threshold',1.0))

        domain_averages_given = ( self._c is not None )
        ## if domain_averages_given:
        ##     c = cache['domain averages'] = self._c
        ## else:
        ##     c = self._compute_domain_averages( surface )

        ## if not cache.has_key('I'):
        ##     cache['I'] = MeshFunction( self._I )

        c = self._compute_domain_averages( surface )

        mass_coef = CurvatureDependentMeshFunction( self._mass_coef_func )

        A = surface.FEM.stiffness_matrix( surface )
        A *= self._mu
        M = surface.FEM.weighted_mass_matrix( surface, mass_coef )

        if domain_averages_given or not use_full_hessian:
            return (A + M)

        else: # using the full hessian, including the negative def. part
            vol0 = surface.interior_volume( self._world_boundary )
            vol1 = surface.exterior_volume( self._world_boundary )

            ## surface_area = surface.surface_area()
            ## int_I, int_I_sqr = self._compute_surface_integrals( surface )
            ## coef = (int_I_sqr - 2.0*c0*int_I + surface_area*c0**2) / vol0 \
            ##        + (int_I_sqr - 2.0*c1*int_I + surface_area*c1**2) / vol1
            ## print "Negative definite part: ", coef,
            ## coef = min(1.0, 0.5*self._min_beta / coef)
            ## print "   coef: ", coef
            coef = 1.0

            ## if not cache.has_key('f0'):
            ##     cache['f0'] = surface.FEM.load_vector( surface, f=None )
            ## if not cache.has_key('f1'):
            ##     cache['f1'] = surface.FEM.load_vector( surface, f=self._I )
            ## f0 = cache['f0']
            ## f1 = cache['f1']
            f0 = surface.FEM.load_vector( surface, f=None )
            f1 = surface.FEM.load_vector( surface, f=self._I )
            v0 = f1 - c[0]*f0
            v1 = f1 - c[1]*f0
            update_vecs = [ ((-coef/vol0)*v0, v0), ((-coef/vol1)*v1, v1) ]
            shape_hess = MatrixWithRankOneUpdates( A + M, update_vecs )

            return shape_hess


    def is_minimized(self, surface, parameters={}, history=None):
        abs_tol = parameters.get( 'absolute tolerance', 0.01 )

        sqrt_sf_area = surface.surface_area()**0.5

        G = self.shape_gradient( surface )
        M = surface.FEM.mass_matrix( surface )
        norm_G = np.dot( G, M.solve(G) )**0.5

        c = self._compute_domain_averages( surface )
        if len(c) == 2:
            diff_c = abs( c[1] - c[0] )
        else:
            c_in  = self._c_in_vector( surface, c )
            c_out = self._c_out_vector( surface, c )
            diff_c = np.min( np.abs( c_in - c_out ) )

        return ( norm_G < 0.5*abs_tol * diff_c**2 * sqrt_sf_area )


    def estimate_errors_from_tol(self, surface):
        """Estimates errors in energy and shape gradient based on given tol.

        This functions estimates the minimum expected error in the values
        of the energy and the L^2 norm of the shape gradient, based on
        the current surface and prescribed tolerances for domain and surface
        integration. Currently it only accounts for the domain integrals in
        the energy, but not the the surface area term.

        Parameters
        ----------
        surface : Surface object
            The current surface, e.g. an instance of a CurveHierarchy etc.

        Returns
        -------
        energy_error : float
            An estimate of the minimum expected error in the data-dependent
            parts of the energy, namely the domain integrals.
        gradient_error : float
            An estimate of the minimum expected error in the
            squared L^2 norm of the shape gradient over the surface.
        """

        if (self._domain_integral_tol is None) or \
           (self._surface_integral_tol is None):
            raise Exception("Domain and/or surface integral tolerances are not defined!")
        return self.estimate_errors( surface, self._domain_integral_tol,
                                     self._surface_integral_tol )


    def estimate_errors(self, surface, domain_error=None, surface_error=None):
        """Estimates the error in energy and shape gradient calculations.

        This functions estimates the error in the values of the energy and
        the L^2 norm of the shape gradient, based on the current surface
        and given domain and surface errors. Currently it only accounts
        for the domain integrals in the energy, but not the the surface
        area term.

        Parameters
        ----------
        surface : Surface object
            The current surface, e.g. an instance of a CurveHierarchy etc.
        domain_error: float, optional
            An average estimate of the integration error when the image is
            integrated over a unit volume.
            Its default value is None. In this case, the current internal
            error estimates for the domain integrals are used. If internal
            error estimates are not available (because domains have not
            been integrated yet), then the current tolerance for domain
            integration is used (current tolerance is not necessarily the
            domain tolerance specified at initialization, it may be larger).
        surface_error: float, optional
            An average estimate of the integration error when the image is
            integrated over a unit surface area.
            Its default value is None. In this case, the current internal
            error estimate for integral of the image over the surface is
            used. If the internal error estimate is not available, then
            the current tolerance for surface integration is used (current
            tolerance is not necessarily the domain tolerance specified at
            initialization, it may be larger).

        Returns
        -------
        energy_error : float
            An estimate of the error in the data-dependent parts of the energy,
            namely the domain integrals.
        gradient_error: float
            An estimate of the error in the squared L^2 norm of the shape gradient
            over the surface: \int_\surface G^2.
        """
        raise NotImplementedError

        cache = self._check_cache( surface )

        # Estimate the error in energy computation.

        domain_averages_given = (self._c is not None)
        if domain_averages_given:
            c = cache['domain averages'] = self._c
        else:
            c = self._compute_domain_averages( surface )

        vol0 = surface.interior_volume( self._world_boundary )
        vol1 = surface.exterior_volume( self._world_boundary )

        if domain_error is not None:
            domain0_error = vol0 * domain_error
            domain1_error = vol1 * domain_error
        elif 'domain integral errors' in cache.keys():
            domain0_error, domain1_error = cache['domain integral errors']
        else:
            domain0_error = vol0 * self._current_tols['domain']
            domain1_error = vol1 * self._current_tols['domain']

        energy_error = c[0] * domain0_error + c[1] * domain1_error

        # Estimate the error in the squared L2 norm of the shape gradient.

        surface_area = surface.surface_area()

        int_I, int_I_sqr = self._compute_surface_integrals( surface )

        if surface_error is not None:
            surface_error = surface_area * surface_error
        elif 'surface integral error' in cache.keys():
            surface_error = cache['surface integral error']
        else:
            surface_error = surface_area * self._current_tols['surface']

        b = abs(c[0] - c[1])
        a = abs(c[0] + c[1]) / 2.0

        if domain_averages_given:
            gradient_error = b**2 * (2.0*a + 1.0) * surface_error
        else:
            surface_coef = b**2 * (1.0 + a)
            domain_coef  = b * ( a*(b + 2.0*a) * surface_area +
                                 (0.5*b + 4.0*a) * abs(int_I) +
                                 2.0 * abs(int_I_sqr) )
            gradient_error = surface_coef * surface_error + \
                             domain_coef * (domain0_error / vol0 +
                                            domain1_error / vol1)

        return (energy_error, gradient_error)


    def has_maximum_accuracy(self):
        return ( (self._current_tols['domain'] <= self._domain_integral_tol) and
                 (self._current_tols['surface'] <= self._surface_integral_tol) )


    def set_accuracy(self, surface, factor=None):
        if factor is not None:
            # Reduce or increase adaptivity tolerances by the given factor.
            # For example, factor=0.5 reduces all tolerances by half.

            # Also if factor < 1.0, then tolerances for domain and surface
            # computations have been reduced, thus accuracy requirements
            # have been tightened; therefore, the old values in the cache
            # are not accurate enough any more and they should be removed.

            if self._current_tols['domain'] > self._domain_integral_tol:
                new_tol = factor * self._current_tols['domain']
                self._current_tols['domain'] = max( new_tol,
                                                    self._domain_integral_tol )

            if self._current_tols['surface'] > self._surface_integral_tol:
                new_tol = factor * self._current_tols['surface']
                self._current_tols['surface'] = max( new_tol,
                                                     self._surface_integral_tol)

                params = surface._adaptivity_parameters

                # Item 0 in params['errors and marking'] is geometric error.
                # We don't change it. Item 1 is data/image error.
                criteria = params['errors and marking'][1]
                marking_params = criteria[1][1]
                marking_params['tolerance'] = self._current_tols['surface']

                # Item 1 in params['coarsening errors and marking'] is data error.
                criteria = params['coarsening errors and marking'][0]
                marking_params = criteria[1][1]
                marking_params['tolerance'] = self._current_tols['surface']

                if factor < 1.0:  self._reset_cache()


        else: # No factor specified, so we initialize the data-related
            # parts of surface adaptivity. Geometric adaptivity of surface
            # should be on by default.

            tolerance = self._current_tols['surface']
            # Unlike the tolerance input to adaptive integration, we do not
            # multiply the tolerance with surface area, because the errors
            # computed in compute_data_error should be scaled by surface
            # area (or curve length) already.

            # Refinement error estimator function
            # data_func = MeshFunction( self._I )
            r_est_func = curve_compute_data_error
            r_est_func_params = {'data function': self._I }

            # Refinement marking function
            r_marking_func = equidistribution_marking
            r_marking_params = {'tolerance': tolerance,
                                'gamma_refine': 1.0,
                                'gamma_coarsen': 0.05 }

            refinement_functions = ( (r_est_func, r_est_func_params),
                                     (r_marking_func, r_marking_params) )

            # Coarsening error estimator function
            c_est_func = curve_compute_data_coarsening_error
            c_est_func_params = {'data function': self._I }

            # Coarsening marking function
            c_marking_func = equidistribution_marking
            c_marking_params = {'tolerance': tolerance,
                                'gamma_refine': 1.0,
                                'gamma_coarsen': 0.05 }

            coarsening_functions = ( (c_est_func, c_est_func_params),
                                     (c_marking_func, c_marking_params) )

            # Assign the adaptivity criteria to the surface.
            surface.set_data_adaptivity_criteria( refinement_functions,
                                                  coarsening_functions )

ChanVeseEnergy = PwConstRegionEnergy
MultiphaseRegionEnergy = PwConstRegionEnergy

#########################################################################
#####               The Isotropic Boundary Energy                   #####
#########################################################################


class IsotropicBoundaryEnergy(ShapeEnergy):
    """
    The isotropic boundary energy approximates the energy of a curve or
    surface given by the weighted boundary integral:

    .. math:: J(\Gamma) = \int_\Gamma g(x) d\Gamma,

    where :math:`\Gamma` is a surface or curve and :math:`g(x)` is a
    spatially-varying weight function.
    This class can compute the energy values, and the first and second
    shape derivatives.
    
    When combined with an image-based weight function :math:`g(x)`, such as
    the edge indicator function of an image intensity function :math:`I(x)`

    .. math::  g(x) = 1 / (1 + | \\nabla I(x)|^2 / \lambda^2), \lambda > 0

    this energy can be used to detect boundaries of objects in images,
    by fitting curves (or surfaces in 3d) to locations of high image
    gradient (see [Caselles1997]_). This energy implements the Lagrangian approach
    proposed in [Dogan2017]_.

    Parameters
    ----------
    parameters : dict
        A dictionary of parameters used to initialize and setup the energy
        object. It has the following keys:
        
            'weight function', a weight function :math:`g(x)` that depends on
            the spatial coordinates :math:`x`, an array of shape (dim x n_coord_pts)
            where dim = 2,3. It should return a 1d array of scalar values of
            length n_coord_pts.
            
            'basin width', (optional) float parameter indicating the width of
            the minima or valleys or basins. The default value is inherited
            from the image interpolant function, and can be inter-pixel dist.
            
            'adaptive integral', (optional) boolean parameter indicating whether
            the integration will be adaptive or not. The default value is
            False, so (non-adaptive) fixed quadrature integration is used.
            
            'integral tol', (optional) float parameter indicating the error
            tolerance of adaptive integration. The default value is 1e-4.

  
    Notes
    -----
    .. [Caselles1997] : Caselles, V.; Kimmel, R.; Sapiro, G. "Geodesic Active Contours."
         *International Journal of Computer Vision*, 22(1), 61-79 (1997).
    .. [Dogan2017] : Dogan, G. "An Efficient Lagrangian Algorithm for an Anisotropic
       Geodesic Active Contour Model."
       In *International Conference on Scale Space and Variational Methods
       in Computer Vision*, 408-420, Springer, Cham (2017).
       
    """

    def __init__(self, parameters):
        super( IsotropicBoundaryEnergy, self ).__init__()

        self._params = parameters

        self._edge_indicator = G = parameters['edge indicator']
        self._G = MeshFunction( G )

        dG_N = lambda x,y: np.sum( y * G.gradient(x), 0 )
        self._dG_N = AnisotropicMeshFunction( dG_N, normal_type='pwlinear' )

        self._using_adaptive_integral = parameters.get('adaptive integral',False)
        self._integral_tol = parameters.get('integral tol',1e-4)
        self._current_tol  = self._integral_tol

        if 'basin width' in parameters.keys():
            self._basin_width = parameters['basin_width']
        else:
            try:
                self._basin_width = G.transition_width()
            except AttributeError:
                self._basin_width = None
        if self._basin_width is None:
            self._basin_width = 1.0


    def basin_width(self):
        return self._basin_width

    def __call__(self, surface):
        if surface.size() == 0:  return 0.0

        int_G = self._get_from_cache( surface, 'surface integral' )
        if int_G is not None:  return int_G  # otherwise, compute int_G below

        # TODO: Check using fixed quadrature makes iterative shape optim.
        #       more robust.
        # The choice of fixed vs adaptive quadrature must optional
        # (default=fixed), set in energy parameters.

        if self._using_adaptive_integral:
            tol = self._current_tol * surface.surface_area()
            int_G, error = adapt_integrate( self._G, surface, tol=tol )
        else: # using fixed quadrature for integral
            int_G = integrate( self._G, surface, 2 )
            error = 0.0

        self._put_in_cache( surface, 'surface integral', int_G )
        self._put_in_cache( surface, 'surface integral error', error )

        return int_G

    def _compute_and_interp_K(self, surface, s):
        FEM = surface.FEM
        X_vec = surface.coords()
        dim = X_vec.shape[0]
        M = FEM.mass_matrix( surface )
        A = FEM.stiffness_matrix( surface )
        N = [ FEM.normal_matrix( surface, k ) for k in range(dim) ]

        K_vec = [ M.solve(A*x) for x in X_vec ]
        K = M.solve( sum(( N[k]*K_vec[k] for k in range(dim) )) )
        K_s = surface.ref_element.interpolate(K,s)
        return K_s

    def shape_gradient(self, surface, s=None, mask=None, coarsened=None, split=False):
        if surface.size() == 0:  return np.empty(0)

        G = self._G
        dG_N = self._dG_N

        if s is not None: # then we want shape gradient value at quad pt s
            K = surface.curvature(s, mask, coarsened)
            # K = self._compute_and_interp_K(surface,s)
            grad = K * G(surface,s,mask,coarsened) + dG_N(surface,s,mask,coarsened)
            return grad

        else: # we want the discretized vector:  dJ(surf;\phi_i)
            if split: # M_g = \int g(x)\phi_i\phi_j and f = \int dG_N\phi_i
                M_g = surface.FEM.weighted_mass_matrix( surface, f=G )
                f = surface.FEM.load_vector( surface, f=dG_N )
                return (M_g, f)
            else: # not split, need full shape gradient vector.
                grad = self._get_from_cache( surface, 'shape gradient' )
                if grad is not None:  return grad
                from numerics.integration import Quadrature1d
                # shape grad = G K + dG/dN => dJ_i = \int (GK+dG/dN) \phi_i
                grad = surface.FEM.load_vector( surface, f=self.shape_gradient, quad=Quadrature1d(degree=2) )
                self._put_in_cache( surface, 'shape gradient', grad )
                return grad

    def _mass_coef_func(self, surface, s):
        # TODO: The 3d version of the following needs principal curvatures K_i.
        threshold = self._hessian_threshold

        dG_N = self._dG_N(surface,s)
        d2G = self._G.hessian(surface,s)

        N = surface.normals( s, smoothness='pwlinear' )
        K = surface.curvature(s)

        indices = product( range(N.shape[0]), repeat=2 )
        d2G_NN = np.sum(( N[i] * d2G[i,j] * N[j] for i,j in indices ))

        result = d2G_NN + 2.0 * K * dG_N

        if not self._full_hessian:
            result[ result < threshold ] = threshold

        return result

    def shape_hessian(self, surface, full_hessian=None, threshold=None):
        if surface.size() == 0:  return np.empty((0,0))

        params = self._params
        if full_hessian is not None:
            self._full_hessian = full_hessian
        else:
            self._full_hessian = params.get('use full Hessian',
                                 params.get('use full hessian',False))
        if threshold is not None:
            self._hessian_threshold = threshold
        else:
            self._hessian_threshold = params.get('Hessian threshold',
                                      params.get('hessian threshold',1.0))

        A = surface.FEM.weighted_stiffness_matrix( surface, self._G )
        M = surface.FEM.weighted_mass_matrix( surface, self._mass_coef_func )
        return (A + M)

    def is_minimized(self, surface, parameters={}, history=None):
        abs_tol = parameters.get( 'absolute tolerance', 0.1 )
        rel_tol = parameters.get( 'relative tolerance', 0.01 )
        dG_scaling = 1.0 / self._edge_indicator.transition_width()
        tol = min( abs_tol, rel_tol*dG_scaling )
        norm_G = self.shape_gradient_norm( surface )
        small_shape_gradient = norm_G < tol * surface.surface_area()**0.5
        return small_shape_gradient

    def estimate_errors_from_tol(self, surface):
        raise NotImplementedError

    def estimate_errors(self, surface, integral_error=None):
        raise NotImplementedError

    def has_maximum_accuracy(self):
        return (self._current_tol['energy'] <= self._integral_tol)

    def set_accuracy(self, surface, factor=None, use_shape_gradient=False):
        if factor is not None:
            # Reduce or increase adaptivity tolerances by the given factor.
            # For example, factor=0.5 reduces all tolerances by half.

            if self._current_tol > self._integral_tol:

                self._current_tol = max( self._integral_tol,
                                         factor * self._current_tol )

                params = surface._adaptivity_parameters

                # Update the tolerance for refinement.
                criteria = params['errors and marking'][1]
                marking_params = criteria[1][1]
                marking_params['tolerance'] = self._current_tol
                # Update the tolerance for coarsening.
                criteria = params['coarsening errors and marking'][0]
                marking_params = criteria[1][1]
                marking_params['tolerance'] = self._current_tol

                # If factor < 1.0, then the tolerance for the surface integral
                # computation has been reduced, thus accuracy requirements
                # have been tightened; therefore, the old values in the cache
                # are not accurate enough any more and they should be removed.
                if factor < 1.0:  self._reset_cache()

        elif not use_shape_gradient: # and factor is None
            # No tolerance factor specified and not using the shape gradient
            # info yet, so we initialize the data-related parts of surface
            # adaptivity. Geometric adaptivity of surface should be on by default.

            tolerance = self._current_tol  # Surface integration tolerance

            # Refinement error estimator function
            r_est_func = curve_compute_data_error
            r_est_func_params = {'data function': self._G }

            # Refinement marking function
            r_marking_func = equidistribution_marking
            r_marking_params = {'tolerance': tolerance,
                                'gamma_refine': 1.0,
                                'gamma_coarsen': 0.05 }

            refinement_functions = ( (r_est_func, r_est_func_params),
                                     (r_marking_func, r_marking_params) )

            # Coarsening error estimator function
            c_est_func = curve_compute_data_coarsening_error
            c_est_func_params = {'data function': self._G }

            # Coarsening marking function
            c_marking_func = equidistribution_marking
            c_marking_params = {'tolerance': tolerance,
                                'gamma_refine': 1.0,
                                'gamma_coarsen': 0.05 }

            coarsening_functions = ( (c_est_func, c_est_func_params),
                                     (c_marking_func, c_marking_params) )

            # Assign the adaptivity criteria to the surface.
            surface.set_data_adaptivity_criteria( refinement_functions,
                                                  coarsening_functions )

        else: # use_shape_gradient is True, so we add the shape gradient
            # as additional data criterion.

            # Refinement error estimator function
            r_est_func = curve_compute_data_error
            r_est_func_params = {'data function': self.shape_gradient }

            # Refinement marking function
            r_marking_func = fixed_ratio_marking
            r_marking_params = {'ratio': 0.05}

            refinement_functions = ( (r_est_func, r_est_func_params),
                                     (r_marking_func, r_marking_params) )
            coarsening_functions = None

            # Assign the adaptivity criteria to the surface.
            surface.set_data_adaptivity_criteria( refinement_functions,
                                                  coarsening_functions )


GeodesicActiveContourEnergy = IsotropicBoundaryEnergy

