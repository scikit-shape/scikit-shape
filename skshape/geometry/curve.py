"""Curves, curve families, and relevant geometric functionality.

This module contains the Curve and CurveHierarchy classes, and the geometric
functionality, such as curvature, normal computations, etc.

"""

import numpy as np
import numba
from numba import jit
from collections import deque
from ._mesh import Mesh
from .domain import Domain2d
from .curve_adaptivity import geometric_adaptivity_parameters, copy_adaptivity_parameters, adapt as curve_adapt, refine_coarsen as curve_refine_coarsen
from ..numerics.fem import curve_fem as FEM


class Curve(Mesh):
    """Lagrangian curve discretized as a list of nodes, and geometric functions.

    Curve class encapsulating the data and functionality for a Lagrangian
    curve discretized as a list of nodes on the (x,y) plane.
    The geometric functions include normal, curvature computation, interior
    area and other functions.
    """

    def __init__(self, coords, adaptivity_parameters={}):
        """Initializes the Curve object with the given coords and adaptivity.

        Initializes the Curve object with the given coordinates and adaptivity
        parameters.

        Parameters
        ----------
        coords : Numpy array
            A (2,n) sized Numpy array of floats, storing the x,y
            coordinates of the nodes, in the order they are on the curve.
            A clockwise order corresponds a negative orientation and
            negative area.
        adaptivity_parameters : dictionary, optional
            A dictionary storing the parameters for adaptivity
            (how and where to add or subtract nodes).
            See geometry.curve_adaptivity.adapt() for more information
            on how to specify the parameters.
            The default value of adaptivity_parameters is {}, in which
            case only geometric adaptivity is turned on.
            To turn off adaptivity, set adaptivity_parameters=None.
        """
        super(Curve, self).__init__()

        try:
            if coords.shape[0] != 2:
                raise ValueError("The coordinate array should have size (2,n_pts).")
        except AttributeError:
            raise ValueError("Coords should be a NumPy array of size (2,n_pts).")

        if np.all(coords[:,0] == coords[:,-1]): # if 1st and last nodes are equal
            raise ValueError("Coordinates of consecutive nodes cannot be the same.")
        # If some of the intermediate consecutive nodes have the same coordinates
        if np.any( np.logical_and( coords[0,1:] == coords[0,:-1],
                                   coords[1,1:] == coords[1,:-1] )):
            raise ValueError("Coordinates of consecutive nodes cannot be the same.")

        self._coords = coords
        self.FEM = FEM
        self.ref_element = FEM.ReferenceElement()
        local_to_global = self.ref_element.local_to_global
        self.local_to_global = lambda s, mask, coarsened: \
                               local_to_global(s, self, mask, coarsened)
        self.reset_data()

        if adaptivity_parameters is None:
            self.mesh_adaptivity = False
            self._adaptivity_parameters = None
            self._geometric_adaptivity = None
        elif len(adaptivity_parameters) == 0: # An empty dictionary
            self.mesh_adaptivity = True
            self._adaptivity_parameters = p = geometric_adaptivity_parameters()
            self._geometric_adaptivity = {
                'errors and marking': p['errors and marking'],
                'coarsening errors and marking':p['coarsening errors and marking']}
            self.adapt()
        else: # User-specified adaptivity parameters
            self.mesh_adaptivity = True
            self._adaptivity_parameters = adaptivity_parameters
            self._geometric_adaptivity = None
            self.adapt()

    def copy(self, copy_cache_data=False):
        new_curve = Curve( self._coords.copy(), adaptivity_parameters=None )

        new_curve.timestamp = self.timestamp

        new_curve.mesh_adaptivity = self.mesh_adaptivity
        if self._adaptivity_parameters is not None:
            adapt_params = copy_adaptivity_parameters( self._adaptivity_parameters,
                                                       new_curve )
        else:
            adapt_params = None
        new_curve._adaptivity_parameters = adapt_params
        if self._geometric_adaptivity is not None:
            geom_adapt = copy_adaptivity_parameters( self._geometric_adaptivity )
        else:
            geom_adapt = None
        new_curve._geometric_adaptivity = geom_adapt

        if copy_cache_data:
            if self._el_sizes is not None:
                new_curve._el_sizes = self._el_sizes.copy()
            if self._el_sizes_sqr is not None:
                new_curve._el_sizes_sqr = self._el_sizes_sqr.copy()
            if self._normals is not None:
                new_curve._normals = self._normals.copy()
            if self._normal_smoothness is not None:
                new_curve._normal_smoothness = self._normal_smoothness
            if self._tangents is not None:
                new_curve._tangents = self._tangents.copy()
            if self._tangent_smoothness is not None:
                new_curve._tangent_smoothness = self._tangent_smoothness
            if self._curvature is not None:
                new_curve._curvature = self._curvature.copy()
            if self._bounding_box is not None:
                new_curve._bounding_box = self._bounding_box.copy()
            if self._length is not None:
                new_curve._length = self._length
            if self._area is not None:
                new_curve._area = self._area

        return new_curve

    def dim(self):
        return 1

    def dim_of_world(self):
        return 2

    def reset_data(self):
        self._el_sizes = None
        self._el_sizes_sqr = None
        self._normals = None
        self._normal_smoothness = None
        self._tangents = None
        self._tangent_smoothness = None
        self._curvature = None
        self._bounding_box = None
        self._length = None
        self._area = None
        self.timestamp += 1

    def set_geometry(self, coords, element_sizes=None,
                     normals=None, curvature=None):
        self.reset_data()
        self._coords = coords
        self._el_sizes = element_sizes
        self._normals = normals
        self._curvature = curvature

    def __repr__(self):
        str_list = []
        x = self._coords[0,:]
        y = self._coords[1,:]

        for i in range(len(x)):
            str_list.append('(' + repr(x[i]) + ',' + repr(y[i]) + ')\n')
        return ''.join(str_list)

    def size(self):
        return self._coords.shape[1]

    def show(self, format='b', factor=1.0):
        import matplotlib.pyplot as plt
        x,y = self.coords()
        x = np.hstack((factor*x,factor*x[0]))
        y = np.hstack((factor*y,factor*y[0]))
        plt.plot(x,y,format)
        plt.show()

    def coords(self):
        return self._coords

    def refine_coarsen(self, mask, data_vecs, refinement_method='curved'):
        if len(mask) == self.size():
            mask2 = mask
        else:
            mask2 = np.zeros( self.size(), dtype=int )
            mask2[mask] = 1
        new_data_vecs, new_indices = \
                       curve_refine_coarsen( self, mask2, data_vecs,
                                             refinement_method )
        self.reset_data()
        return (new_data_vecs, new_indices)

    def set_adaptivity(self, adaptivity=True):
        """Turns on/off mesh adaptivity.

        This function is used to switch mesh adaptivity on or off.

        Parameters
        ----------
        adaptivity : bool
            A boolean argument to set mesh adaptivity to True or False.
        """
        self.mesh_adaptivity = adaptivity

    def set_data_adaptivity_criteria(self, refinement_functions=None,
                                     coarsening_functions=None, append=False):
        """Sets data adaptivity criteria given for refinement and coarsening.

        Curves can be adapted spatially (refined or coarsened) with respect
        to given criteria. This function is used to set or add a refinement
        or coarsening pair (of error and marking functions) to the criteria
        list. The curve's criteria list includes the geometric criterion
        by default (if created with default arguments).

        The geometric criterion estimates the geometric error by examining
        the curvatures and element sizes of the curves. Then the marking
        function marks the elements with errors above a certain threshold
        for refinement. See the following function for more information:
            geometry.curve_adaptivity.compute_geometric_error(...)

        If new refinement or coarsening error functions are added or appended,
        then mesh_adaptivity is turned on. To turn it off, you need to call
        the function: set_adaptivity(False).

        Parameters
        ----------
        refinement_functions
             A pair of pairs storing the error estimation function and
            the corresponding marking function to drive refinement of
            elements. It has the following format:
                ( error_estimation_pair, marking_pair )
            where error_estimation_pair = (error_func, error_params),
                  marking_pair = (marking_func, marking_params).
            The error estimation function should have the curve as an
            argument and three other optional arguments: mask, a boolean
            NumPy array or an integer NumPy array of element indices, vec,
            a float NumPy array to store the error values, parameters,
            a dictionary of parameters needed for error estimation.
            The default for these three arguments can be None.
            An example is
               compute_geometric_error(curve, mask=None, vec=None,
                                       parameters=None)
            Marking functions are defined in the module geometry.marking.

        coarsening_functions
            A pair of pairs storing the error estimation function and
            the corresponding marking function to drive coarsening of
            elements. Its format is the same as that of refinement
            functions.

        append : bool, optiomal
            A boolean optional flag with default value equal to False.
            If it is True, the given criteria are appended to the
            existing list and all the criteria in the list are used
            together. If it is False, only the first geometric criteria
            (if they are available) are kept in the refinement and
            coarsening criteria list, the others in the lists are removed
            and the new ones are inserted.
        """
        params = self._adaptivity_parameters
        geom_criteria = self._geometric_adaptivity

        if (refinement_functions is None) and (coarsening_functions is None):
            # Resetting data adaptivity, i.e. no more data criteria in params.
            # But we don't change self.mesh_adaptivity (=True/False) in this case.
            if params is None:
                return  # No params to reset, so return.
            elif geom_criteria is not None: # Reset to only geometric criteria.
                params['errors and marking'] = \
                               geom_criteria['errors and marking']
                params['coarsening errors and marking'] = \
                               geom_criteria['coarsening errors and marking']
            else:
                self._adaptivity_parameters = None
            return # without changing the value of mesh.adaptivity

        # At this point, one of refinement_functions or coarsening_functions
        # should be given as a valid argument.

        if params is None: # Then initialize to an empty params dictionary.
            params = self._adaptivity_parameters = geometric_adaptivity_parameters()
            params['errors and marking'] = [] # No more geometric criteria.
            params['coarsening errors and marking'] = []

        if refinement_functions is not None:
            if append:
                params['errors and marking'].append( refinement_functions )
            elif geom_criteria is None:
                params['errors and marking'] = [ refinement_functions ]
            else: # geom_criteria already known (so it should come first).
                params['errors and marking'] = geom_criteria['errors and marking'] \
                                               + [ refinement_functions ]

        if coarsening_functions is not None:
            if append:
                params['coarsening errors and marking'].append( coarsening_functions )
            elif geom_criteria is None:
                params['coarsening errors and marking'] = [ coarsening_functions ]
            else: # geom_criteria already known (so it should come first).
                params['coarsening errors and marking'] = \
                             geom_criteria['coarsening errors and marking'] \
                             + [ coarsening_functions ]

        self.mesh_adaptivity = True

    def adapt(self, parameters=None):
        if parameters is None:  parameters = self._adaptivity_parameters
        changed = curve_adapt( self, parameters )
        if changed:  self.reset_data()
        return changed

    def bounding_box(self):
        if self._bounding_box is None:
            min_coord = np.min( self._coords, 1 )
            max_coord = np.max( self._coords, 1 )
            self._bounding_box = {'min':min_coord,'max':max_coord}
        return self._bounding_box

    def reverse_orientation(self):
        self._coords = np.fliplr( self._coords )
        self.reset_data()

    def set_orientation(self, new_orientation):
        if new_orientation not in [1,-1]:
            raise ValueError("New orientation can only be one of {-1,1}.")
        if new_orientation != self.orientation():
            self.reverse_orientation()

    def edges(self):
        n = self._coords.shape[1]
        s = np.empty( (2,n), dtype=int )
        s[0,0:n] = range(0,n)
        s[1,0:n-1] = range(1,n)
        s[1,n-1] = 0
        return s

    def length(self):
        if self._length is None:
            self._length = np.sum( self.element_sizes() )
        return self._length

    def surface_area(self):
        return self.length()

    def orientation(self):
        """Returns the orientation of the curve.

        Computes the signed area of the curve, and returns the orientation
        of the curve based on the area. If area > 0, then orientation is 1.
        If area < 0, then orientation is -1. If area = 0, then orientation is 0.

        Returns
        -------
        orientation_value : int, one of -1,0,1
        """
        area = self.area()
        if area > 0.: return  1
        if area < 0.: return -1
        return 0

    def area(self):
        if self._area is not None:   return self._area

        # We will need the normals for the area computation.
        # If pw const normals have already been computed, that's good.
        # Otherwise store old ones and compute pw const normals from scratch.
        if (self._normals is not None) and (self._normal_smoothness == 'pwconst'):
            old_normals = None
            normals = self._normals
        else:
            old_normals = self._normals
            old_normal_smoothness = self._normal_smoothness
            normals = self.normals( smoothness='pwconst' )

        # The area of the curve is computed using the divergence theorem
        #    \int_domain dx = 0.5 \int_domain div(x) dx
        #                   = 0.5 \int_curve x.n dS
        #                   = 0.5 \sum_i \int_element_i x.n_i dS
        #                   = 0.5 \sum_i (n_i,0 \int_i x0 dS +
        #                                 n_i,1 \int_i x1 dS
        #                   = 0.5 \sum_i l_i (n_i,0 (x(i,0) + x(i+1,0)) / 2 +
        #                                     n_i,1 (x(i,1) + x(i+1,1)) / 2)

        n = self._coords.shape[1]
        x = self._coords[0,:]
        y = self._coords[1,:]
        Nx, Ny = normals  # x,y coordinates of the normals

        el_sizes = self.element_sizes()

        area = 0.25 * np.sum( el_sizes[0:n-1] *
                              (Nx[0:n-1] * (x[0:n-1] + x[1:n]) +
                               Ny[0:n-1] * (y[0:n-1] + y[1:n])) )
        area += 0.25 * el_sizes[n-1] * (Nx[n-1] * (x[n-1] + x[0]) +
                                        Ny[n-1] * (y[n-1] + y[0]))
        self._area = area

        if old_normals is not None:
            self._normals = old_normals
            self._normal_smoothness = old_normal_smoothness

        return area

    def volume(self):
        return self.area()

    def interior_area(self, world_boundary=None):
        return abs(self.area())

    def interior_volume(self, world_boundary=None):
        return abs(self.area())

    def exterior_area(self, world_boundary=None):
        return abs(world_boundary.area()) - abs(self.area())

    def exterior_volume(self, world_boundary=None):
        return abs(world_boundary.area()) - abs(self.area())


    def move_back(self):
        prev = self._previous_state

        if prev is None:  return

        self.timestamp += 1

        self.mesh_adaptivity = prev.mesh_adaptivity
        self._adaptivity_parameters = prev._adaptivity_parameters
        self._geometric_adaptivity = prev._geometric_adaptivity

        self._coords = prev._coords
        self._el_sizes = prev._el_sizes
        self._el_sizes_sqr = prev._el_sizes_sqr
        self._normals = prev._normals
        self._normal_smoothness = prev._normal_smoothness
        self._tangents = prev._tangents
        self._tangent_smoothness = prev._tangent_smoothness
        self._curvature = prev._curvature
        self._bounding_box = prev._bounding_box
        self._length = prev._length
        self._area = prev._area

        self._previous_state = None


    def move(self, update, adaptivity=None, world_boundary=None):
        # If adaptivity argument is not specified (given as None),
        # then the default adaptivity behavior is dictated by the
        # variable Curve.mesh_adaptivity.

        if (update.shape[0] != 2) and (update.shape[1] != self.size()):
            raise Exception("The coordinate array should have the size (2,n_pts).")

        self._previous_state = self.copy()

        self._coords += update

        if (adaptivity is None and self.mesh_adaptivity) or adaptivity:
            self.adapt()

        self.reset_data() # Clear all cached data that depends on coord info.
        return True


    def element_sizes(self, mask=None, coarsened=False, squared=False):
        if mask is not None and len(mask) == 0:  return np.empty(0)

        if not coarsened and squared and (self._el_sizes_sqr is not None):
            if mask is None:
                return self._el_sizes_sqr.copy()
            else:
                return self._el_sizes_sqr[mask]

        if not coarsened and not squared and (self._el_sizes is not None):
            if mask is None:
                return self._el_sizes.copy()
            else:
                return self._el_sizes[mask]

        x,y = self._coords
        n = len(x)

        if not coarsened:

            if mask is None:
                d = np.empty(n)
                d[0:n-1] = (x[0:n-1] - x[1:n])**2 + (y[0:n-1] - y[1:n])**2
                d[n-1] = (x[n-1] - x[0])**2 + (y[n-1] - y[0])**2

            else: # mask given
                mask2 = (mask + 1) % n
                d = (x[mask2] - x[mask])**2 + (y[mask2] - y[mask])**2

        else: # coarsened elements sizes

            if mask is not None:
                mask2 = (mask + 2) % n
                d = (x[mask2] - x[mask])**2 + (y[mask2] - y[mask])**2

            else: # no mask, all elements are coarsened
                if (n % 2) == 0: # case of even number of elements
                    m = n/2
                    d = np.empty(m)
                    d[m-1] = (x[0] - x[n-2])**2 + (y[0] - y[n-2])**2
                else: # case of odd number of elements
                    m = n/2 + 1
                    d = np.empty(m)
                    d[m-1] = (x[0] - x[n-1])**2 + (y[0] - y[n-1])**2
                d[0:m-1] = (x[2:n:2] - x[0:n-2:2])**2 + (y[2:n:2] - y[0:n-2:2])**2

        if not coarsened and mask is None:
            self._el_sizes_sqr = d.copy()

        if not squared:
            d = np.sqrt(d)
            if not coarsened and mask is None:
                self._el_sizes = d.copy()

        return d


    def _normalize_vector_by_size(self, vec, vec_size=None, mask=None,
                                  coarsened=False, mask_extra=[]):
        n = vec.shape[1]
        vec_size0 = vec_size

        # Compute sizes of vectors if coarsened or given vec_size is None.
        if mask is not None:
            if coarsened or vec_size is None:
                vec_size = np.empty(n)
                vec_size[mask] = np.sqrt( vec[0,mask]**2 + vec[1,mask]**2 )
        elif coarsened:
            vec_size = np.empty(n)
            vec_size[0:n:2] = np.sqrt( vec[0,0:n:2]**2 + vec[1,0:n:2]**2 )
        else:
            vec_size = np.sqrt( vec[0]**2 + vec[1]**2 )

        if len(mask_extra) > 0:
            if vec_size0 is not None:
                vec_size[mask_extra] = vec_size0[mask_extra]
            else:
                vec_size[mask_extra] = np.sqrt( vec[0,mask_extra]**2 +
                                                vec[1,mask_extra]**2 )

        # Divide the tangents by their lengths to obtain unit length.
        if mask is not None:
            vec[0,mask] /= vec_size[mask]
            vec[1,mask] /= vec_size[mask]
        elif coarsened: # no mask
            vec[0,0:n:2] /= vec_size[0:n:2]
            vec[1,0:n:2] /= vec_size[0:n:2]
        else: # all elements
            vec[0,:] /= vec_size
            vec[1,:] /= vec_size
        if len(mask_extra) > 0:
            vec[0,mask_extra] /= vec_size[mask_extra]
            vec[1,mask_extra] /= vec_size[mask_extra]

        return (vec, vec_size)

    def _pwconst_tangents(self, coords, mask, coarsened,
                          mask_extra=[], normalize=True):
        n = coords.shape[1]
        tangents = np.empty_like(coords)

        if not coarsened:
            if mask is None: # tangents for all elements
                tangents[:,0:n-1] = coords[:,1:n] - coords[:,0:n-1]
                tangents[:,n-1] = coords[:,0] - coords[:,n-1]
            else: # tangents for elements indexed by mask
                mask2 = (mask + 1) % n
                tangents[:,mask] = coords[:,mask2] - coords[:,mask]

        else: # Tangents for coarsened elements:
              #   tangent[:,k] = coords[:,k+2] - coords[:,k]
            if mask is None: # all elements are coarsened
                tangents[:,0:n-2:2] = coords[:,2:n:2] - coords[:,0:n-2:2]
                if (n % 2) == 0: # Then the last element is also coarsened.
                    tangents[:,n-2] = coords[:,0] - coords[:,n-2]
                else: # n is odd, the last element is not coarsened.
                    tangents[:,n-1] = coords[:,0] - coords[:,n-1]
            else: # Tangents for coarsened elements indexed by mask
                mask2 = (mask + 2) % n
                tangents[:,mask] = coords[:,mask2] - coords[:,mask]
                # Will need normals of some uncoarsened extra elements too.
                if len(mask_extra) > 0:
                    mask3 = (mask_extra + 1) % n  # These are not coarsened.
                    tangents[:,mask_extra] = coords[:,mask3] - coords[:,mask_extra]

        if not normalize:
            tangent_size = None
        else: # normalize
            tangents, tangent_size = \
                      self._normalize_vector_by_size( tangents, self._el_sizes,
                                                      mask, coarsened, mask_extra )
        return (tangents, tangent_size)


    def _pwlinear_tangents(self, tangents0, mask, coarsened, neighbors=False):
        n = tangents0.shape[1]

        # If coarsened and mask is used, we copy tangent values to the
        # next locations to make the computations easier and consistent.
        # Also a given s coord requires tangents at nodes k & k+1.
        if mask is not None:
            if coarsened:
                mask_next = (mask + 1) % n
                tangents0[:,mask_next] = tangents0[:,mask]
                if neighbors:
                    mask = np.unique( np.hstack(( mask, (mask+2)%n )))
            elif neighbors: # and not coarsened
                mask = np.unique( np.hstack(( mask, (mask+1)%n )))

        # We compute pwlinear tangents by summing current & prev tangents.
        if mask is not None: # mask is given
            tangents = np.empty_like( tangents0 )
            mask_prev = (mask - 1) % n
            tangents[:,mask] = tangents0[:,mask] + tangents0[:,mask_prev]

        else: # mask is None
            tangents = tangents0.copy()
            if not coarsened:
                tangents[:,1:n] += tangents0[:,0:n-1]
                tangents[:,0] += tangents0[:,n-1]
            else: # coarsened
                tangents[:,2:n:2] += tangents0[:,0:n-2:2]
                tangents[:,0] += tangents0[:,n-2] if (n%2)==0 else tangents0[:,n-1]

        tangents = self._normalize_vector_by_size( tangents, None,
                                                   mask, coarsened )[0]
        return tangents


    def tangents(self, s=None, mask=None, coarsened=False, smoothness='pwconst'):
        """Computes the unit tangents of the curve.

        Returns the tangent vectors for the elements of a curve (the
        pwconst case) or for the nodes of the curve (the pwlinear case).
        In the piecewise constant case, the tangents are given by the
        difference of the first and second nodes of the element, i.e.
                T[i] = ( x[i+1] - x[i],  y[i+1] - y[i] )
        In the piecewise linear case, the tangents are computed as weighted
        averages of the element tangents.
        The tangent vectors are normalized to have unit length.

        Parameters
        ----------
        s : float, optional
            An optional local coordinate, a value between 0 and 1.
            This needs to be given if the value of the tangent is sought
            at a local position, say s=0.2, on all elements. Its default
            value is None.
        mask : NumPy array, optional
            A NumPy array of integer indices, indicating the elements
            for which the tangents are sought.
        coarsened : bool, optional
            True if the tangents are sought for coarsened elements,
            otherwise False.
        smoothness : str, optional
            An optional flag denoting the smoothness of the computed tangents.
            Its value can be 'pwconst', corresponding to piecewise constant
            tangents defined on the elements, (but not on the nodes) or
            its value can be 'pwlinear', corresponding to piecewise linear
            tangents, defined by coefficients on the nodes and by
            interpolation of the coefficients on the edges.

        Returns
        -------
        tangents : NumPy array
            A NumPy array of size 2xN storing the x and y components
            of the tangents.
        """
        if mask is not None and len(mask) == 0:  return np.empty((2,0))

        ##### Checking for cached tangents first #####

        if not coarsened and (self._tangents is not None) \
           and (smoothness == self._tangent_smoothness):

            if (s is None) or (smoothness == 'pwconst'):
                if mask is None:
                    return self._tangents.copy()
                else: # mask is given
                    return self._tangents[:,mask]

            else: # s is given and smoothness == 'pwlinear'
                tangents = self.ref_element.interpolate( self._tangents, s, mask )
                tangents = self._normalize_vector_by_size( tangents )[0]
                return tangents

        # If tangents were not cached and return, now we need to compute them.

        coords = self._coords
        n = coords.shape[1]

        ##### Figure out which elements to compute tangents #####

        mask0 = mask
        mask_extra = np.empty(0,dtype=int)
        if mask is not None and smoothness == 'pwlinear':

            if not coarsened:
                mask_prev = (mask0 - 1) % n
                if s is None: # We will use elements k-1, k only.
                    mask = np.unique( np.hstack(( mask_prev, mask0 )) )
                else: # Need pwlinear normal at s, using elements k-1, k, k+1.
                    mask_next = (mask0 + 1) % n
                    mask = np.unique( np.hstack(( mask_prev, mask0, mask_next )) )

            else: # if coarsened
                mask_extra = (mask0 - 1) % n  # This is mask_prev.
                if s is not None: # then we need mask_next also.
                    mask_extra = np.hstack(( mask_extra, (mask0 + 2) % n ))
                # already_marked are elements k & k+1 to be coarsened into one.
                already_marked = set( mask0 ).union( set( (mask0 + 1) % n ) )
                set_extra = set( mask_extra ).difference( already_marked )
                mask_extra = np.array( list(set_extra) )

        ##### The piecewise constant tangents #####

        normalize = (smoothness == 'pwconst')
        tangents, tangent_size = self._pwconst_tangents( coords, mask, coarsened,
                                                         mask_extra, normalize )
        if not coarsened and mask is None:
            self._el_sizes = tangent_size

        # Return if pwconst tangents are sought, otherwise continue
        # to compute a pwlinear reconstruction by weighted averaging.

        mask = mask0  # Back to original mask, where the values are sought.

        if smoothness == "pwconst":
            if mask is not None:
                return tangents[:,mask]
            elif coarsened:
                return tangents[:,0:n:2]
            else: # tangents for all elements (no mask and not coarsened)
                self._tangents = tangents.copy() # Store a copy in cache.
                self._tangent_smoothness = smoothness
                return tangents

        elif smoothness != "pwlinear":
            raise ValueError("Smoothness argument should be one of 'pwconst' or 'pwlinear'!")

        ##### The piecewise linear tangents #####

        mask0 = mask
        neighbors = ( s is not None )
        tangents = self._pwlinear_tangents(tangents, mask, coarsened, neighbors)
        mask = mask0
        # Store a copy of the tangents in cache.
        if mask is None and not coarsened:
            self._tangents = tangents if (s is not None) else tangents.copy()
            self._tangent_smoothness = smoothness

        ##### Return pw linear tangents, interpolate if s given #####

        if s is None:
            if mask is not None:
                tangents = tangents[:,mask]
            elif coarsened: # and mask is None
                tangents = tangents[:,0:n:2]
        else: # Interpolate tangents linearly at local coord s.
            tangents = self.ref_element.interpolate( tangents, s, mask, coarsened )
            tangents = self._normalize_vector_by_size( tangents )[0]

        return tangents


    def normals(self, s=None, mask=None, coarsened=False, smoothness='pwconst'):
        """Computes the unit normals of the curve.

        Returns the normal vectors for the elements of a curve (the
        pwconst case) or for the nodes of the curve (the pwlinear case).
        In the piecewise constant case, the normals are given by the
        difference of the first and second nodes of the element, i.e.
                N[i] = ( y[i] - y[i+1],  x[i+1] - x[i] )
        In the piecewise linear case, the normals are computed as weighted
        averages of the element normals.
        The normal vectors are normalized to have unit length.

        Parameters
        ----------
        s : float, optional
            An optional local coordinate, a value between 0 and 1.
            This needs to be given if the value of the normal is sought
            at a local position, say s=0.2, on all elements. Its default
            value is None.
        mask : NumPy array, optional
            A NumPy array of integer indices, indicating the elements
            for which the normals are sought.
        coarsened : bool, optional
            True if the normals are sought for coarsened elements,
            otherwise False.
        smoothness : str, optional
            An optional flag denoting the smoothness of the computed normals.
            Its value can be 'pwconst', corresponding to piecewise constant
            normals defined on the elements, (but not on the nodes) or
            its value can be 'pwlinear', corresponding to piecewise linear
            normals, defined by coefficients on the nodes and by
            interpolation of the coefficients on the edges.

        Returns
        -------
        normals : NumPy array
            A NumPy array of size 2xN storing the x and y components
            of the normals.

        """
        if mask is not None and len(mask) == 0:  return np.empty((2,0))

        # Check for cached normals first.
        if not coarsened and (self._normals is not None) \
           and (smoothness == self._normal_smoothness):

            if (s is None) or (smoothness == 'pwconst'):
                if mask is None:
                    return self._normals.copy()
                else: # mask is given
                    return self._normals[:,mask]

            else: # s is given and smoothness == 'pwlinear'
                normals = self.ref_element.interpolate( self._normals, s, mask )
                normals = self._normalize_vector_by_size( normals )[0]
                return normals

        # If normals not available in the cache, compute them from tangents.
        tangents = self.tangents( s, mask, coarsened, smoothness )
        normals = np.vstack(( tangents[1,:], -tangents[0,:] ))

        # Store the computed results in cache.
        if not coarsened and (mask is None):
            if (s is None) or (smoothness == 'pwconst'):
                self._normals = normals.copy()
                self._normal_smoothness = smoothness
            elif (self._tangents is not None) and \
                 (self._tangent_smoothness == smoothness):
                self._normals = np.vstack((  self._tangents[1,:],
                                            -self._tangents[0,:] ))
                self._normal_smoothness = smoothness
        return normals


    def curvature(self, s=None, mask=None, coarsened=False):
        """Computes the curvature of the curve.

        Computes the curvature at the node i by fitting a circle to nodes
        i-1, i, i+1 and computing its radius r. Then the curvature is given
        by 1/r. The curvature values between the nodes are given by linear
        interpolation.

        Parameters
        ----------
        s : float, optional
            An optional local coordinate, a value between 0 and 1.
            This needs to be given if the value of the curvature is sought
            at a local position, say s=0.2, on all elements. Its default
            value is None.
        mask : NumPy array, optional
            A NumPy array of integer indices, indicating the elements
            for which the curvature values are sought.
        coarsened : bool, optional
            True if the curvature values are sought for coarsened elements,
            otherwise False.

        Returns
        -------
        curvature : NumPy array
            A NumPy array storing the computed curvature values.

        """

        ##### There are 2 ways to compute curvature (see below) #####
        ##### We use the 2nd approach: circle fitting           #####

        ####### FEM-based method to compute curvature #############
        #
        #  # Solve for curvature vector:  M K_vec = A X
        #  K_vec = [ M.solve( A*x ) for x in coords ]
        #
        #  # Compute the scalar curvature: M K = N0 K0 + N1 K1 (+ N2 K2)
        #  rhs = np.sum(( N[k] * K_vec[k]  for k in range(dim) ))
        #  K = M.solve( rhs )
        #
        ###########################################################

        ####### Curvature by circle-fitting to P1,P2,P3 #############
        #
        #  To compute curvature K2 at point P2, we fit a circle
        #  to the three points P1=(x1,y1), P2=(x2,y2), P3=(x3,y3).
        #
        #         | x1 (y2 - y3) + x2 (y3 - y1) + x3 (y1 - y2) |
        #   K2 = -------------------------------------------------
        #         sqrt( |P2 - P1|^2 * |P3 - P1|^2 * |P3 - P2|^2 )
        #
        ###########################################################

        if (mask is not None) and (len(mask) == 0):  return np.empty(0)

        ##### Check for cached curvature first #####

        if not coarsened and self._curvature is not None:
            if s is not None:
                return self.ref_element.interpolate( self._curvature, s, mask )
            elif mask is None:
                return self._curvature.copy()
            else: # mask of indices given
                return self._curvature[mask]

        ##### Figure out which nodes to compute curvature #####

        n = self._coords.shape[1]
        mask0 = mask
        mask_extra = mask_extra_next = mask_extra_prev = np.array([],dtype=int)
        if mask is not None:
            if not coarsened:
                if s is not None:
                    mask = np.unique( np.hstack((mask0, (mask0 + 1) % n)) )
                mask_prev = (mask - 1) % n
                mask_next = (mask + 1) % n
            else:
                mask_prev = (mask - 1) % n
                mask_next = (mask + 2) % n
                if s is not None:
                    mask_extra = set( mask_next )
                    already_marked = set( mask0 ).union( set( (mask0 + 1) % n ))
                    set_extra = mask_extra.difference( already_marked )
                    mask_extra = np.array( list(set_extra) )
                    mask_extra_prev = (mask_extra - 1) % n
                    mask_extra_next = (mask_extra + 1) % n

        ##### Assign coordinates x,y #####

        m = n
        if not coarsened:
            x,y = self._coords
        elif mask is None: # and coarsened
            m = int(np.ceil( n / 2.0 ))
            x,y = self._coords[:,0:n:2]
        else: # coarsened and mask is given
            coords = np.empty_like( self._coords )
            for mask_i in [ mask, mask_prev, mask_next, mask_extra_next ]:
                coords[:, mask_i ] = self._coords[:, mask_i ]
            coords[:, (mask+1)%n ] = self._coords[:, mask ]
            x,y = coords

        ##### Compute the element sizes #####

        if mask is None:
            d = self.element_sizes( None, coarsened, squared=True )
        elif not coarsened: # and mask is given
            d = np.empty(n)
            all_mask = np.unique( np.hstack(( mask_prev, mask )) )
            d[all_mask] = self.element_sizes( all_mask, False, squared=True )
        else: # coarsened and mask is given
            d = np.empty(n)
            two_masks = np.unique( np.hstack(( mask_prev, mask_extra )) )
            d[two_masks] = self.element_sizes( two_masks, False, squared=True )
            d[mask] = self.element_sizes( mask, True, squared=True )
            d[(mask+1)%n] = d[mask]

        ##### Compute y1 = y[k+1]-y[k], y2 = y[k+1]-y[k-1], d2 =... #####

        y1, y2, d2 = np.empty(m), np.empty(m), np.empty(m)

        if mask is None:
            y1[0:m-1] = y[1:m] - y[0:m-1]
            y1[m-1] = y[0] - y[m-1]
            y2[1:m-1] = y[2:m] - y[0:m-2]
            y2[m-1] = y[0] - y[m-2]
            y2[0] = y[1] - y[m-1]
            d2[1:m-1] = y2[1:m-1]**2 + (x[2:m] - x[0:m-2])**2
            d2[m-1] = y2[m-1]**2 + (x[0] - x[m-2])**2
            d2[0] = y2[0]**2 + (x[1] - x[m-1])**2

        else: # mask is given
            for indx,prev,next in [(mask, mask_prev, mask_next),
                                   (mask_extra, mask_extra_prev, mask_extra_next)]:
                if len(indx) > 0:
                    y1[indx] = y[next] - y[indx]
                    y1[prev] = y[indx] - y[prev]
                    y2[indx] = y[next] - y[prev]
                    d2[indx] = y2[indx]**2 + (x[next] - x[prev])**2

        ##### Compute the curvature using x,y1,y2,d,d2 #####

        K = np.empty(m)
        if mask is None:
            bottom_sqr = d.copy()
            bottom_sqr[1:m-1] *= d[0:m-2] * d2[1:m-1]
            bottom_sqr[m-1] *= d[m-2] * d2[m-1]
            bottom_sqr[0] *= d[m-1] * d2[0]
            K[1:m-1] = x[0:m-2]*y1[1:m-1] - x[1:m-1]*y2[1:m-1] +x[2:m]*y1[0:m-2]
            K[m-1] = x[m-2]*y1[m-1] - x[m-1]*y2[m-1] + x[0]*y1[m-2]
            K[0] = x[m-1]*y1[0] - x[0]*y2[0] + x[1]*y1[m-1]
            K = -2.0 * K / np.sqrt( bottom_sqr )

        else: # mask is given
            for i,prev,next in [(mask, mask_prev, mask_next),
                                (mask_extra, mask_extra_prev, mask_extra_next)]:
                if len(i) > 0:
                    bottom_sqr = d[i] * d[prev] * d2[i]
                    K[i] = x[prev] * y1[i] - x[i] * y2[i] + x[next] * y1[prev]
                    K[i] = -2.0 * K[i] / np.sqrt( bottom_sqr )

        ##### Return the curvature values, interpolate if s given #####

        mask = mask0

        if s is not None: # Interpolate curvature linearly at local coord s.
            if mask is None:
                K = self.ref_element.interpolate( K, s ) # NO coarsen flag
            else: # mask is given
                K = self.ref_element.interpolate( K, s, mask, coarsened )
        elif mask is not None:
            K = K[mask]
        elif not coarsened:
            self._curvature = K.copy()

        return K


    def second_fund_form(self, s=None, mask=None, coarsened=False):
        """Estimates the second fundamental form on nodes of the curve.

        Estimates the second fundamental form on each element using
        the following formula:
            II(s) = K(s) T(s) T(s)^T
        where K(s) is the scalar curvature, interpolated linearly from
        nodal curvature values, T(s) is piecewise linear reconstruction
        of the unit tangent vector, T(s)^T is its transpose.
        The second fundamental curve can be computed for nodes of the
        curve or interior points of the elements denoted by s, on all
        elements or selected elements specified by mask. They can be
        computed for coarsened elements too.

        Parameters
        ----------
        s : float, optional
            An optional local coordinate, a value between 0 and 1.
            This needs to be given if the value of the second fundamental
            form is sought at a local position, say s=0.2, on all elements.
            Its default value is None.
        mask : NumPy array, optional
            A NumPy array of integer indices, indicating the elements
            for which the values of the second fundamental form are sought.
        coarsened : bool, optional
            True if the values of the second fundamental form are sought
            for coarsened elements, otherwise False.

        Returns
        -------
        II : NumPy array
            An array of size 2x2xN storing the components of the second
            fundamental form.

        """
        if mask is not None and len(mask) == 0:  return np.empty((2,2,0))

        T = self.tangents( s, mask, coarsened, smoothness='pwlinear' )
        K = self.curvature( s, mask, coarsened )
        II = np.empty((2,2,len(K)))

        for i,j in [(0,0),(0,1),(1,0),(1,1)]:
            II[i,j] = K * T[i] * T[j]
        return II

    def safe_step_size(self, V, tol=0.3):
        """The maximum safe step size to avoid local geometric distortions.

        If the spacing between the two nodes of an element changes too
        rapidly, this might create distortions in the geometric representation.
        This function computes the maximum safest step size to move the
        curve with the given velocity. The computation is based on the
        following inequality to be enforced

            (dx1 - dx0) / h = dt (V1 - V0).T / h < tol

        where dx1, dx0 denote the changes in the node locations,
        V1, V0 are their velocity, h is the size, T is the tangent
        vector of the element.

        Parameters
        ----------
        V : NumPy array
            The velocity vector, a NumPy array of size (2,n), where n
            is the number of nodes of the curve.
        tol : float
            Allowed ratio of relative tangential displacement with
            respect to the element size. The default value is 0.3.

        Returns
        -------
        dt : float
            The maximum safe step size allowed within the specified
            tolerance.
        """
        n = self.size()
        dV = V.copy()
        dV[:,0:n-1] -= V[:,1:n]
        dV[:,n-1] -= V[:,0]

        d = self.element_sizes()
        T = self.tangents( smoothness='pwconst' )

        # Compute the maximum relative tangential displacement
        max_rel_disp = np.max( np.abs(np.sum( dV*T, 0 )) / d )
        if max_rel_disp == 0.0:
            dt = np.inf
        else:
            dt = tol / max_rel_disp
        return dt


    def _ray_intersects_edge(self,pt,edge):
        x1, y1 = edge[:,0]
        x2, y2 = edge[:,1]

        # Edge crossing rules:
        # 1) An upward edge includes its start node, excludes its end node.
        # 2) A downward edge excludes its start node, includes its end node.
        # 3) Horizontal edges are excluded.
        # 4) The edge-ray intersection must be strictly right of the point P.

        if ((y1 < y2) and (y1 <= pt[1] < y2)) or \
           ((y1 > y2) and (y1 > pt[1] >= y2)):
            m = (x2 - x1) / (y2 - y1)
            x0 = x1 + m * (pt[1] - y1)
            if pt[0] < x0:
                return True

        return False


    def contains(self,obj): # obj can be a curve or a point

        this_bbox = self.bounding_box()
        try: # The input object might be a curve
            # Check if the input curve's bounding box intersects this curve's
            obj_bbox = obj.bounding_box()
            if obj_bbox['max'][0] < this_bbox['min'][0]: return False
            if obj_bbox['min'][0] > this_bbox['max'][0]: return False
            if obj_bbox['max'][1] < this_bbox['min'][1]: return False
            if obj_bbox['min'][1] > this_bbox['max'][1]: return False
            # If the bounding boxes don't "not intersect",
            # use the coordinates of first node for inclusion check
            coords = obj.coords()
            pt = coords[:,0]
        except: # If obj doesn't behave like a curve,
            pt = obj  # it is probably a point
            if pt[0] < this_bbox['min'][0]: return False
            if pt[0] > this_bbox['max'][0]: return False
            if pt[1] < this_bbox['min'][1]: return False
            if pt[1] > this_bbox['max'][1]: return False

        n = self._coords.shape[1]
        intersection_count = Curve._compute_edge_intersection_count( pt[0],pt[1], self._coords, n)
        return (intersection_count % 2 == 1)

    @jit( numba.int32(numba.float64, numba.float64, numba.float64[:,:], numba.int32),
          nopython=True )
    def _compute_edge_intersection_count(X, Y, coords, n):
        # Edge crossing rules:
        # 1) An upward edge includes its start node, excludes its end node.
        # 2) A downward edge excludes its start node, includes its end node.
        # 3) Horizontal edges are excluded.
        # 4) The edge-ray intersection must be strictly right of the point P.

        count = 0;
        for i in range(n-1):
            x1,y1 = coords[:,i]
            x2,y2 = coords[:,i+1]
            if ((y1 < y2) and (y1 <= Y) and (Y <  y2)) or \
               ((y1 > y2) and (y1  > Y) and (Y >= y2)):
                m = (x2 - x1) / (y2 - y1)
                x0 = x1 + m * (Y - y1)
                if (X < x0):  count = count + 1

        x1,y1 = coords[:,n-1]
        x2,y2 = coords[:,0]
        if ((y1 < y2) and (y1 <= Y) and (Y <  y2)) or \
           ((y1 > y2) and (y1  > Y) and (Y >= y2)):
            m = (x2 - x1) / (y2 - y1)
            x0 = x1 + m * (Y - y1)
            if (X < x0):  count = count + 1

        return count


class CurveHierarchy(Mesh):
    """Hierarchy of nonintersecting simple curves on the (x,y) plane.

    This class stores the hierarchy and spatial positioning of a set of
    nonintersecting simple curves on (x,y) planes. It stores the list of
    the Curve objects as well. In this way, it can perform the geometric
    computations, e.g. curvature, normals for all the curves.
    """

    def __init__(self, curve_list, world_boundary=None, curve_hierarchy=None,
                 adaptivity_parameters={}):
        """Initializes the CurveHierarchy object with the given curve list.

        Initializes the CurveHierarchy object with the given list of Curve
        objects and optionally the world boundary, curve hierarchy and
        adaptivity parameters.

        Parameters
        ----------
        curve_list : list
            A list of curve objects that defines the CurveHierarchy.
        world_boundary : Curve class, optional
            An optional Curve object of four nodes that defines
            the domain/world boundary as a rectangle.
        curve_hierarchy: tuple, optional
            An optional triplet ( topmost_curve_ids, parent, children )
            of three lists defined as follows:
            parent[curve_id] is the id of the immediate (enclosing)
                parent curve of a given curve specified with curve_id,
            children[curve_id] is a set of the ids of the curves
                enclosed by the curve specified with curve_id,
            topmost_curve_ids is a list of curve ids of the curves
                that have parent=None, namely they are not enclosed
                by any other curve.
        adaptivity_parameters: dictionary, optional
            A dictionary storing the parameters for adaptivity (how and
            where to add or subtract nodes).
            See geometry.curve_adaptivity.adapt() for more information
            on how to specify the parameters.
            The default value of adaptivity_parameters is {}, in which
            case only geometric adaptivity is turned on.
            To turn off adaptivity, set adaptivity_parameters=None.

        """
        super(CurveHierarchy, self).__init__()

        from ._curve_topology import adjust_curves_for_boundary, update_topology, compute_intersections, compute_boundary_intersections
        self._adjust_curves = adjust_curves_for_boundary
        self._update_topology = update_topology
        self._compute_intersections = compute_intersections
        self._compute_boundary_intersections = compute_boundary_intersections

        self._curve_list = curve_list
        self._boundary = world_boundary
        if len(curve_list) > 0:
            self.FEM = curve_list[0].FEM
            self.ref_element = curve_list[0].ref_element

        if curve_hierarchy is not None:
            topmost_curves, parent, children = curve_hierarchy
        elif len(curve_list) > 0:
            topmost_curves, parent, children \
                            = self._create_curve_hierarchy( curve_list )
        else:
            topmost_curves, parent, children = [], [], []

        self._topmost_curve_ids = topmost_curves
        self._parent = parent
        self._children = children
        if len(topmost_curves) > 0:
            self._topmost_orientation = curve_list[ topmost_curves[0] ].orientation()
        else:
            self._topmost_orientation = 1

        self.topology_changes = True

        self.reset_data()

        if adaptivity_parameters is None:
            self.mesh_adaptivity = False
            self._adaptivity_parameters = None
            self._geometric_adaptivity = None
        elif len(adaptivity_parameters) == 0: # An empty dictionary
            self.mesh_adaptivity = True
            self._adaptivity_parameters = p = geometric_adaptivity_parameters()
            self._geometric_adaptivity = {
                'errors and marking': p['errors and marking'],
                'coarsening errors and marking': p['coarsening errors and marking'] }
        else: # User-specified adaptivity parameters
            self.mesh_adaptivity = True
            self._adaptivity_parameters = adaptivity_parameters
            self._geometric_adaptivity = None

        if self.mesh_adaptivity:  self.adapt()


    def copy(self, copy_cache_data=False):
        new_curve_list = [ curve.copy( copy_cache_data=copy_cache_data )
                           for curve in self._curve_list ]

        if self._boundary is None:
            new_boundary = None
        else:
            new_boundary = self._boundary.copy()

        topmost_curves = list( self._topmost_curve_ids )
        parent = list( self._parent )
        children = [ child_set.copy() for child_set in self._children ]
        curve_hierarchy = ( topmost_curves, parent, children )

        new_curves = CurveHierarchy( new_curve_list, new_boundary, curve_hierarchy,
                                  adaptivity_parameters=None )

        new_curves.timestamp = self.timestamp
        new_curves.ref_element = self.ref_element

        new_curves.topology_changes = self.topology_changes
        new_curves.mesh_adaptivity = self.mesh_adaptivity
        if self._adaptivity_parameters is not None:
            adapt_params = copy_adaptivity_parameters( self._adaptivity_parameters,
                                                       new_curves )
        else:
            adapt_params = None
        new_curves._adaptivity_parameters = adapt_params
        if self._geometric_adaptivity is not None:
            geom_adapt = copy_adaptivity_parameters( self._geometric_adaptivity )
        else:
            geom_adapt = None
        new_curves._geometric_adaptivity = geom_adapt

        if copy_cache_data:
            if self._coords is not None:
                new_curves._coords = self._coords.copy()
            if self._el_sizes is not None:
                new_curves._el_sizes = self._el_sizes.copy()
            if self._el_sizes_sqr is not None:
                new_curves._el_sizes_sqr = self._el_sizes_sqr.copy()
            if self._normals is not None:
                new_curves._normals = self._normals.copy()
            if self._normal_smoothness is not None:
                new_curves._normal_smoothness = self._normal_smoothness
            if self._tangents is not None:
                new_curves._tangents = self._tangents.copy()
            if self._tangent_smoothness is not None:
                new_curves._tangent_smoothness = self._tangent_smoothness
            if self._curvature is not None:
                new_curves._curvature = self._curvature.copy()
            if self._2nd_fund_form is not None:
                new_curves._2nd_fund_form = self._2nd_fund_form.copy()
            if self._edges is not None:
                new_curves._edges = self._edges.copy()
            if self._hole_pts is not None:
                new_curves._hole_pts = self._hole_pts.copy()
            if self._interior_domain is not None:
                new_curves._interior_domain = self._interior_domain.copy()
            if self._exterior_domain is not None:
                new_curves._exterior_domain = self._exterior_domain.copy()
            if self._regions is not None:
                new_curves._regions = [ region.copy() for region in self._regions ]

        return new_curves


    def dim(self):
        return 1

    def dim_of_world(self):
        return 2

    def curve_list(self):
        return self._curve_list

    def submeshes(self):
        return self._curve_list

    def has_submeshes(self):
        return True

    def get_curve_object(self, curve_id):
        return self._curve_list[ curve_id ]

    def reset_data(self):
        self._coords = None
        self._el_sizes = None
        self._el_sizes_sqr = None
        self._normals = None
        self._normal_smoothness = None
        self._tangents = None
        self._tangent_smoothness = None
        self._curvature = None
        self._2nd_fund_form = None
        self._edges = None
        self._hole_pts = None
        self._interior_domain = None
        self._exterior_domain = None
        self._regions = None
        self.timestamp += 1

    def refine_coarsen(self, mask, data_vecs, refinement_method='curved'):
        mask2 = np.zeros( self.size(), dtype=int )
        mask2[mask] = 1

        all_indices = []
        all_vecs = [ [] for k in range(len(data_vecs)) ]
        offset = 0
        for curve in self._curve_list:
            end = offset + curve.size()
            curve_mask = mask2[offset:end]
            data_vecs_in = [ vec[offset:end] for vec in data_vecs ]

            if np.any(curve_mask): # if any of curve_mask entries is nonzero.
                data_vecs_out, indices = \
                               curve.refine_coarsen( curve_mask, data_vecs_in,
                                                     refinement_method )
            else:
                data_vecs_out, indices = data_vecs_in, np.array([],dtype=int)

            all_indices.append( indices )
            for k,vec in enumerate(data_vecs_out):
                all_vecs[k].append(vec)
            offset = end

        new_offset = self._curve_list[0].size()
        for curve, indices in zip(self._curve_list[1:], all_indices[1:]):
            indices += new_offset
            new_offset += curve.size()

        new_indices = np.hstack( all_indices )
        new_data_vecs = [ np.hstack(vecs) for vecs in all_vecs ]

        self.reset_data()
        return (new_data_vecs, new_indices)


    def set_topology_changes(self, value=True):
        self.topology_changes = value


    def set_adaptivity(self, adaptivity=True):
        """Turns on/off mesh adaptivity.

        This function is used to switch mesh adaptivity on or off.

        Parameters
        ----------
        adaptivity : bool
            A boolean arguments to set mesh adaptivity to True or False.

        """
        self.mesh_adaptivity = adaptivity


    def set_data_adaptivity_criteria(self, refinement_functions=None,
                                     coarsening_functions=None, append=False):
        """Sets data adaptivity criteria given for refinement and coarsening.

        Curves can be adapted spatially (refined or coarsened) with respect
        to given criteria. This function is used to set or add a refinement
        or coarsening pair (of error and marking functions) to the criteria
        list. The curve's criteria list includes the geometric criterion
        by default (if created with default arguments).

        The geometric criterion estimates the geometric error by examining
        the curvatures and element sizes of the curves. Then the marking
        function marks the elements with errors above a certain threshold
        for refinement. See the following function for more information:
            geometry.curve_adaptivity.compute_geometric_error(...)

        If new refinement or coarsening error functions are added or appended,
        then mesh_adaptivity is turned on. To turn it off, you need to call
        the function: set_adaptivity(False).

        Parameters
        ----------
        refinement_functions
            A pair of pairs storing the error estimation function
            and the corresponding marking function to drive
            refinement of elements. It has the following format:
                ( error_estimation_pair, marking_pair )
            where error_estimation_pair = (error_func, error_params),
                  marking_pair = (marking_func, marking_params).
            The error estimation function should have the curve as an
            argument and three other optional arguments: mask, a boolean
            NumPy array or an integer NumPy array of element indices, vec,
            a float NumPy array to store the error values, parameters,
            a dictionary of parameters needed for error estimation.
            The default for these three arguments can be None.
            An example is
               compute_geometric_error(curve, mask=None, vec=None,
                                       parameters=None)
            Marking functions are defined in the module geometry.marking.
        coarsening_functions
            A pair of pairs storing the error estimation function and
            the corresponding marking function to drive coarsening
            of elements. Its format is the same as that of refinement
            functions.
        append : bool, optional
            A boolean optional flag with default value False.
            If it is True, the given criteria are appended to the
            existing list and all the criteria in the list are used
            together. If it is False, only the first geometric criteria
            (if they are available) are kept in the refinement and
            coarsening criteria list, the others in the lists are removed
            and the new ones are inserted.

        """
        params = self._adaptivity_parameters
        geom_criteria = self._geometric_adaptivity

        if (refinement_functions is None) and (coarsening_functions is None):
            # Resetting data adaptivity, i.e. no more data criteria in params.
            # But we don't change self.mesh_adaptivity (=True/False) in this case.
            if params is None:
                return  # No params to reset, so return.
            elif geom_criteria is not None: # Reset to only geometric criteria.
                params['errors and marking'] = \
                               geom_criteria['errors and marking']
                params['coarsening errors and marking'] = \
                               geom_criteria['coarsening errors and marking']
            else:
                self._adaptivity_parameters = None
            return # without changing the value of mesh.adaptivity

        # At this point, one of refinement_functions or coarsening_functions
        # should be given as a valid argument.

        if params is None: # Then initialize to an empty params dictionary.
            params = self._adaptivity_parameters = geometric_adaptivity_parameters()
            params['errors and marking'] = [] # No more geometric criteria.
            params['coarsening errors and marking'] = []

        if refinement_functions is not None:
            if append:
                params['errors and marking'].append( refinement_functions )
            elif geom_criteria is None:
                params['errors and marking'] = [ refinement_functions ]
            else: # geom_criteria already known (so it should come first).
                params['errors and marking'] = geom_criteria['errors and marking'] \
                                               + [ refinement_functions ]

        if coarsening_functions is not None:
            if append:
                params['coarsening errors and marking'].append( coarsening_functions )
            elif geom_criteria is None:
                params['coarsening errors and marking'] = [ coarsening_functions ]
            else: # geom_criteria already known (so it should come first).
                params['coarsening errors and marking'] = \
                             geom_criteria['coarsening errors and marking'] \
                             + [ coarsening_functions ]

        self.mesh_adaptivity = True


    def adapt(self, parameters=None):
        if parameters is None:  parameters = self._adaptivity_parameters
        curve_family_changed = False
        for curve in self._curve_list:
            curve_changed = curve.adapt( parameters )
            curve_family_changed = curve_family_changed or curve_changed
        if curve_family_changed:  self.reset_data()
        return curve_family_changed

    def topmost_curve_ids(self):
        return self._topmost_curve_ids

    def curve_hierarchy(self):
        return (self._topmost_curve_ids, self._parent, self._children)

    def _create_curve_hierarchy(self, curve_list):
        """Computes the curve hierarchy information from a given curve list.

        Given a list of curve objects, this function computes the curve
        hierarchy information, namely, the information of which curve encloses
        a given curve. This information is returned as triplet of three lists:
         ( topmost_curve_ids, parent, children )

        Parameters
        ----------
        curve_list : list
            A list of curve objects, for which the curve hierarchy information
            will be computed. The curves are assumed not to be intersecting
            each other. Intersections might create inconsistencies in the curve
            hierarchy calculations.

        Returns
        -------
        curve_hierarchy
            triplet ( topmost_curve_ids, parent, children ) of three lists:
            parent[curve_id] is the id of the immediate (enclosing)
                parent curve of a given curve specified with curve_id,
            children[curve_id] is a set of the ids of the curves
                enclosed by the curve specified with curve_id,
            topmost_curve_ids is a list of curve ids of the curves
                that have parent=None, namely they are not enclosed
                by any other curve.

        """
        if curve_list is None: return None

        # Sort all the curves with descending area; the curves will
        # be examined in this order to create the curve hierarchy
        curve_sort_key = lambda c: np.abs(c.area())
        curve_list.sort( key=curve_sort_key, reverse=True )

        # Initialize the parent and children set lists.
        n_curves = len( curve_list )
        parent = [ None ]*n_curves
        children = [ set() for i in range(n_curves) ]

        # If there is only one curve in the curve family, then return;
        # the curve hierarchy will consist of a single topmost curve.
        if len(curve_list) == 1: return ([0], parent, children)

        # Go through all the curves in descending order to see if
        # the curve is contained in an already-processed larger curve.
        for curve_id, curve in enumerate( curve_list ):
            for possible_parent in range(curve_id-1,-1,-1):
                possible_parent_curve = curve_list[ possible_parent ]
                if possible_parent_curve.contains( curve ):
                    parent[ curve_id ] = possible_parent
                    children[ possible_parent ].add( curve_id )
                    break

        topmost_curve_ids = [ curve_id
                              for curve_id, parent_id in enumerate(parent)
                              if parent_id is None ]

        return (topmost_curve_ids, parent, children)


    def regions(self):
        if self._regions is not None:  return self._regions

        curves = self._curve_list
        regions = []
        for k, curve in enumerate( curves ):
            curve_children = [ curves[child].copy() for child in self._children[k] ]
            n_children = len( curve_children )

            region_curves = [ curve.copy() ] + curve_children
            if curve.orientation() < 0.0:
                for crv in region_curves:  crv.reverse_orientation()

            topmost_curves = [ 0 ]
            parent = [ None ] + [ 0 ]*n_children
            children = [ set(range(1,n_children+1)) ] + [ set() ]*n_children
            curve_hierarchy = ( topmost_curves, parent, children )

            region = CurveHierarchy( region_curves, curve_hierarchy=curve_hierarchy,
                                  adaptivity_parameters=None )
            regions.append( region )

        self._regions = regions
        return regions

    def region_areas(self, world_boundary=None):
        if world_boundary is not None:  self.set_boundary( world_boundary )
        boundary = self._boundary

        if self._regions is not None:
            regions = self._regions
        else:
            regions = self.regions()

        n_areas = len(regions) if boundary is None else (len(regions)+1)
        areas = np.empty( n_areas )

        for k, region in enumerate(regions):
            areas[k] = abs( region.interior_volume( boundary ) )

        if boundary is not None: # Compute the area outside the regions.
            hole_ids = self._topmost_curve_ids
            N = len(hole_ids)
            parent = [ None ]*N
            children = [ set() ]*N
            curves = [ regions[i]._curve_list[0] for i in hole_ids ]
            curve_hierarchy = ( range(N), parent, children )
            exclusions = CurveHierarchy( curves, curve_hierarchy=curve_hierarchy,
                                      adaptivity_parameters=None )
            areas[-1] = exclusions.exterior_area( boundary )

        return areas

    def region_volumes(self, world_boundary=None):
        return self.region_areas( world_boundary )


    def size(self):
        total_size = 0
        for curve in self._curve_list:
            total_size += curve.size()
        return total_size

    def orientation(self):
        return self._topmost_orientation

    def reverse_orientation(self):
        self.set_orientation( -self._topmost_orientation )

    def set_orientation(self, new_orientation):
        if new_orientation not in [-1,1]:
            raise ValueError("New orientation can only be one of {1,-1}.")
        elif new_orientation != self._topmost_orientation:
            for curve in self._curve_list:
                curve.reverse_orientation()
            self._topmost_orientation = -self._topmost_orientation
            self.reset_data()

    def length(self):
        total_len = 0.0
        for curve in self._curve_list:
            total_len += curve.length()
        return total_len

    def surface_area(self):
        return self.length()

    def area(self):
        return self.interior_area()

    def volume(self):
        return self.area()

    def interior_area(self, world_boundary=None):
        if world_boundary is not None:  self.set_boundary( world_boundary )
        boundary = self._boundary

        if boundary is None:
            if self.orientation() < 0:
                raise ValueError("Curve family has negative orientation. World boundary should be specified!")
            curve_list = self._curve_list
        else: # boundary is defined (it is not None).
            new_curves = self._adjust_curves( self, boundary )
            curve_list = new_curves.curve_list()

        total_area = sum(( curve.area() for curve in curve_list ))
        return total_area

    def interior_volume(self, world_boundary=None):
        return self.interior_area( world_boundary )

    def exterior_area(self, world_boundary=None):
        if world_boundary is not None:  self.set_boundary( world_boundary )
        boundary = self._boundary

        inverted_curves = self.copy( copy_cache_data=False )
        inverted_curves.reverse_orientation()
        if boundary is None:
            if self.orientation() > 0:
                raise ValueError("Positive orientation. World boundary should be specified!")
            return inverted_curves.interior_area()

        new_curves = self._adjust_curves( inverted_curves, boundary )
        return new_curves.interior_area()

    def exterior_volume(self, world_boundary=None):
        return self.exterior_area( world_boundary )

    def show(self,format='b',factor=1.0):
        import matplotlib.pyplot as plt
        for curve in self._curve_list:
            x,y = curve.coords()
            x = np.hstack((factor*x,factor*x[0]))
            y = np.hstack((factor*y,factor*y[0]))
            plt.plot(x,y,format)
        plt.show()

    def coords(self):
        if self._coords is not None: return self._coords

        n = self.size()
        coords = np.empty((2,n))

        offset = 0
        for curve in self._curve_list:
            curve_size = curve.size()
            coords[:, offset:offset+curve_size] = curve.coords()
            offset += curve_size
        self._coords = coords

        return coords

    def edges(self):
        if self._edges is not None: return self._edges

        n = self.size()
        edges = np.empty((2,n),dtype=int)

        start = 0
        for curve in self._curve_list:
            end = start + curve.size()
            edges[0,start:end] = np.arange(start,end)
            edges[1,start:end-1] = np.arange(start+1,end)
            edges[1,end-1] = start
            start = end
        self._edges = edges

        return edges

    def _compute_hole_point(self, curve_id):
        # A hole point (or a point inside the curve) will be computed
        # using the first edge of the curve.
        # The point is obtained by projecting the midpoint of the first
        # edge along the edge normal at a distance of half
        # the edge size. If the hole point turns out to be outside
        # curve or inside one of the interior curves (children),
        # then we halve the distance (so that the hole point is closer
        # to the edge) and try again.
        curve_list = self._curve_list
        children = self._children
        curve  = curve_list[ curve_id ]
        coords = curve.coords() # to calculate midpoint of first edge
        normal = curve.normals()[:,0] # normal of the first edge
        normal *= -curve.orientation() # make sure normal points inwards
        mid_pt = 0.5 * (coords[:,0] + coords[:,1])
        edge_size = np.sum( (coords[:,0] - coords[:,1])**2 )**0.5
        alpha  = 0.5 * edge_size

        hole_pt = mid_pt + alpha * normal
        hole_pt_is_ok = True
        if not curve.contains( hole_pt ): # pt is outside the curve
            hole_pt_is_ok = False
        else: # pt is inside the curve, but it shouldn't be inside the children
            for child_id in children[ curve_id ]:
                child_curve = curve_list[ child_id ]
                if child_curve.contains( hole_pt ):
                    hole_pt_is_ok = False
                    break

        while not hole_pt_is_ok:
            alpha /= 2.0
            if alpha < 1e-14 * edge_size: # Abort if alpha is too small
                raise Exception("Cannot compute hole point, alpha is too small!")
            hole_pt = mid_pt + alpha * normal
            hole_pt_is_ok = True # true for now
            if not curve.contains( hole_pt ): # pt is outside the curve
                hole_pt_is_ok = False
            else: # if pt is inside the curve, it shouldn't be inside the children
                for child_id in children[ curve_id ]:
                    child_curve = curve_list[ child_id ]
                    if child_curve.contains( hole_pt ):
                        hole_pt_is_ok = False
                        break
        return hole_pt

    def hole_points(self):

        children = self._children
        hole_pts = [] # one point per a "void" of the curve family
        hole_curves = deque([]) # queue of curves to compute the hole points

        for top_curve in self._topmost_curve_ids:
            hole_curves.extend( children[ top_curve ] )

        while len( hole_curves ) > 0:
            # pop the first curve from the queue
            curve_id = hole_curves.popleft()
            # compute a hole point that is inside this curve
            pt = self._compute_hole_point( curve_id )
            hole_pts.append( pt )
            # if the curve has grandchildren, their hole pts should be computed
            for child_id in children[ curve_id ]:
                hole_curves.extend( children[ child_id ] ) # grandchildren

        self._hole_pts = np.array( hole_pts ).T
        return self._hole_pts


    def set_boundary(self, world_boundary):

        if self._boundary is None:
            self._boundary = world_boundary
            self._interior_domain = None
            self._exterior_domain = None
            self._regions = None

        else: # Check if given world_boundary is the same as self._boundary.
            difference = np.sum( (self._boundary.coords()
                                  - world_boundary.coords())**2 )**0.5
            # If the difference is too large, update self._boundary
            if difference > 1e-14:
                self._boundary = world_boundary
                self._interior_domain = None
                self._exterior_domain = None
                self._regions = None


    def interior_domain(self, world_boundary=None):
        if world_boundary is not None:  self.set_boundary( world_boundary )

        if self._interior_domain is None:
            boundary = self._boundary

            if boundary is None:
                if self.orientation() < 0:
                    raise ValueError("Curve family has negative orientation. World boundary should be specified!")
                self._interior_domain = Domain2d( self )

            else: # Boundary is defined (is not None).
                new_curves = self._adjust_curves( self, boundary )
                self._interior_domain = Domain2d( new_curves )

        return self._interior_domain


    def exterior_domain(self, world_boundary=None):
        if world_boundary is not None:  self.set_boundary( world_boundary )

        if self._exterior_domain is None:
            inverted_curves = self.copy( copy_cache_data=False )
            inverted_curves.reverse_orientation()

            boundary = self._boundary
            if boundary is None:
                if self.orientation() > 0:
                    raise ValueError("Curve family has positive orientation. World boundary should be specified!")
                self._exterior_domain = Domain2d( inverted_curves )

            else: # Boundary is defined (is not None).
                new_curves = self._adjust_curves( inverted_curves, boundary )
                self._exterior_domain = Domain2d( new_curves )

        return self._exterior_domain


    def move_back(self):
        prev = self._previous_state

        if prev is None:  return

        self.timestamp += 1

        self._curve_list = prev._curve_list
        self._boundary = prev._boundary
        self._topmost_orientation = prev._topmost_orientation
        self._topmost_curve_ids = prev._topmost_curve_ids
        self._parent = prev._parent
        self._children = prev._children

        self.topology_changes = prev.topology_changes
        self.mesh_adaptivity = prev.mesh_adaptivity
        self._adaptivity_parameters = prev._adaptivity_parameters
        self._geometric_adaptivity = prev._geometric_adaptivity

        self._coords = prev._coords
        self._el_sizes = prev._el_sizes
        self._el_sizes_sqr = prev._el_sizes_sqr
        self._normals = prev._normals
        self._normal_smoothness = prev._normal_smoothness
        self._tangents = prev._tangents
        self._tangent_smoothness = prev._tangent_smoothness
        self._curvature = prev._curvature
        self._2nd_fund_form = prev._2nd_fund_form
        self._edges = prev._edges
        self._hole_pts = prev._hole_pts
        self._interior_domain = prev._interior_domain
        self._exterior_domain = prev._exterior_domain
        self._regions = prev._regions

        self._previous_state = None


    def move(self, update, adaptivity=None, world_boundary=None):
        if (update.shape[0] != 2) and (update.shape[1] != self.size()):
            raise Exception("The coordinate array should have the size (2,n_pts).")

        # Store a copy of the current state (to be used by move_back()).
        self._previous_state = self.copy()

        # The coordinate update may flip some of the curves, for example
        # from +1 orientation to -1 orientation or vice versa. This should
        # not be allowed. So compute orientations before and after the update
        # and take back the update and return failure if the orientations
        # don't match.
        old_orientations = [ c.orientation() for c in self._curve_list ]

        offset = 0
        for curve in self._curve_list:
            n = curve.size()
            curve.move( update[:,offset:offset+n], adaptivity=False )
            offset += n

        new_orientations = [ c.orientation() for c in self._curve_list ]

        inverted_component = max(( o1*o2 < 0 for o1,o2
                                   in zip(old_orientations,new_orientations) ))

        if inverted_component:
            self.move_back()
            return False


        # Perform topological surgery (reconnect curves) if necessary.
        if self.topology_changes:
            self._update_topology( self, update )


        # If world_boundary is given, check if some curves are now completely
        # outside the world_boundary. Remove the outside curves.
        if world_boundary is not None:
            intersections, outside_curve_ids = \
                self._compute_boundary_intersections( self._curve_list,
                                                      world_boundary )
            self.add_remove_curves( outside_curve_ids, [] )

        if self.size() == 0:
            self.move_back()
            return False

        # Adapt (refine/coarsen) the curves.
        if (adaptivity is None and self.mesh_adaptivity) or adaptivity:
            self.adapt()

            # Are there new intersections requiring more topology surgery?
            if self.topology_changes:
                intersection_info = self._compute_intersections( self )
                # Repeat checks/adjustments until no more intersections.
                while len(intersection_info[0]) > 0:
                    self._update_topology( self, update )
                    self.adapt()
                    intersection_info = self._compute_intersections( self )

        # Reset cached data because the curves have moved.
        self.reset_data()
        return True  # => successful move


    def _partition_mask(self, mask):
        """Partitions a given mask into local masks for curves.

        Partitions the given mask indices to curves and maps
        them to local mask indices for individual curves.
        Say, we have two curves of sizes 3,5 respectively.
        We are given a mask = [0,7,4,2]. This should be split
        into two masks: [0,2] => [0,2], [4,7] => [1,4].

        Parameters
        ----------
        mask : NumPy array
            NumPy array of integers denoting indices to curve nodes

        Returns
        -------
        curve_masks : list
            List of NumPy arrays, each of which corresponds to a mask
            for a curve in the CurveHierarchy.

        """
        n_curves = len( self._curve_list )
        if mask is None: return [None]*n_curves
        mask.sort()
        curve_sizes = [0] + [ curve.size() for curve in self._curve_list ]
        bounds = np.cumsum( curve_sizes )  # bounds within mask array
        indx = mask.searchsorted( bounds ) # where to place bounds in mask
        curve_masks = [ mask[ indx[k]:indx[k+1] ] - bounds[k]
                        for k in range(n_curves) ]
        return curve_masks


    def local_to_global(self, s, mask=None, coarsened=False):
        if mask is not None and len(mask) == 0:  return np.empty(0)

        curve_masks = self._partition_mask( mask )

        coords = [ curve.local_to_global( s, cmask, coarsened )
                   for curve,cmask in zip(self._curve_list, curve_masks) ]

        return np.hstack( coords )


    def element_sizes(self, mask=None, coarsened=False, squared=False):
        if mask is not None and len(mask) == 0:  return np.empty(0)
        if len(self._curve_list) == 0:  return np.empty(0)

        if not coarsened and squared and self._el_sizes_sqr is not None:
            if mask is None:
                return self._el_sizes_sqr.copy()
            else:
                return self._el_sizes_sqr[mask]

        if not coarsened and not squared and self._el_sizes is not None:
            if mask is None:
                return self._el_sizes.copy()
            else: # mask is given
                return self._el_sizes[mask]

        curve_masks = self._partition_mask( mask )

        el_sizes = [ curve.element_sizes( cmask, coarsened, squared )
                     for curve,cmask in zip(self._curve_list, curve_masks)]

        el_sizes = np.hstack( el_sizes )

        if not coarsened and mask is None:
            if squared:
                self._el_sizes_sqr = el_sizes.copy()
            else:
                self._el_sizes = el_sizes.copy()

        return el_sizes


    def normals(self, s=None, mask=None, coarsened=False, smoothness='pwconst'):
        """Computes the unit normals of the curve family.

        Returns the normal vectors for the elements of the curves (the
        pwconst case) or for the nodes of the curves (the pwlinear case).
        In the piecewise constant case, the normals are given by the
        difference of the first and second nodes of the element, i.e.
                N[i] = ( y[i] - y[i+1],  x[i+1] - x[i] )
        In the piecewise linear case, the normals are computed as weighted
        averages of the element normals.
        The normal vectors are normalized to have unit length.

        Parameters
        ----------
        s : float, optional
            An optional local coordinate, a value between 0 and 1.
            This needs to be given if the value of the normal is sought
            at a local position, say s=0.2, on all elements. Its default
            value is None.
        mask : NumPy array, optional
            A NumPy array of integer indices, indicating the elements
            for which the normals are sought.
        coarsened : bool, optional
            True if the normals are sought for coarsened elements,
            otherwise False.
        smoothness : str, optional
            An optional flag denoting the smoothness of the computed normals.
            Its value can be 'pwconst', corresponding to piecewise constant
            normals defined on the elements, (but not on the nodes) or
            its value can be 'pwlinear', corresponding to piecewise linear
            normals, defined by coefficients on the nodes and by
            interpolation of the coefficients on the edges.

        Returns
        -------
        normals : NumPy array
            A NumPy array of size 2xN storing the x and y components
            of the normals.
        """
        if mask is not None and len(mask) == 0:  return np.empty((2,0))

        if not coarsened and (self._normals is not None) \
           and (self._normal_smoothness == smoothness):

            if (s is None) or (smoothness == 'pwconst'):
                if mask is None:
                    return self._normals.copy()
                else: # mask is given
                    return self._normals[:,mask]

        curve_masks = self._partition_mask( mask )

        normals = [ curve.normals( s, cmask, coarsened, smoothness )
                    for curve,cmask in zip(self._curve_list, curve_masks)]

        normals = np.hstack( normals )

        if s is None and mask is None and not coarsened:
            self._normals = normals.copy()
            self._normal_smoothness = smoothness

        return normals


    def tangents(self, s=None, mask=None, coarsened=False, smoothness='pwconst'):
        """Computes the unit tangents of the curve family.

        Returns the tangent vectors for the elements of the curves (the
        pwconst case) or for the nodes of the curves (the pwlinear case).
        In the piecewise constant case, the tangents are given by the
        difference of the first and second nodes of the element, i.e.
                T[i] = ( x[i+1] - x[i],  y[i+1] - y[i] )
        In the piecewise linear case, the tangents are computed as weighted
        averages of the element tangents.
        The tangent vectors are normalized to have unit length.

        Parameters
        ----------
        s : float, optional
            An optional local coordinate, a value between 0 and 1.
            This needs to be given if the value of the tangent is sought
            at a local position, say s=0.2, on all elements. Its default
            value is None.
        mask : NumPy array, optional
            A NumPy array of integer indices, indicating the elements
            for which the tangents are sought.
        coarsened : bool, optional
            True if the tangents are sought for coarsened elements,
            otherwise False.
        smoothness : str, optional
            An optional flag denoting the smoothness of the computed tangents.
            Its value can be 'pwconst', corresponding to piecewise constant
            tangents defined on the elements, (but not on the nodes) or
            its value can be 'pwlinear', corresponding to piecewise linear
            tangents, defined by coefficients on the nodes and by
            interpolation of the coefficients on the edges.

        Returns
        -------
        tangents : NumPy array
            A NumPy array of size 2xN storing the x and y components
            of the tangents.

        """
        if mask is not None and len(mask) == 0:  return np.empty((2,0))

        if not coarsened and (self._tangents is not None) \
           and (self._tangent_smoothness == smoothness):

            if (s is None) or (smoothness == 'pwconst'):
                if mask is None:
                    return self._tangents.copy()
                else: # mask is given
                    return self._tangents[:,mask]

        curve_masks = self._partition_mask( mask )

        tangents = [ curve.tangents( s, cmask, coarsened, smoothness )
                     for curve,cmask in zip(self._curve_list, curve_masks)]

        tangents = np.hstack( tangents )

        if s is None and mask is None and not coarsened:
            self._tangents = tangents.copy()
            self._tangent_smoothness = smoothness

        return tangents

    def curvature(self, s=None, mask=None, coarsened=False):
        """Computes the curvature of the curves in the CurveHierarchy.

        Computes the curvature at the node i by fitting a circle to nodes
        i-1, i, i+1 and computing its radius r. Then the curvature is given
        by 1/r. The curvature values between the nodes are given by linear
        interpolation.

        Parameters
        ----------
        s : float, optional
            An optional local coordinate, a value between 0 and 1.
            This needs to be given if the value of the curvature is sought
            at a local position, say s=0.2, on all elements. Its default
            value is None.
        mask : NumPy array, optional
            A NumPy array of integer indices, indicating the elements
            for which the curvature values are sought.
        coarsened : bool, optional
            True if the curvature values are sought for coarsened elements,
            otherwise False.

        Returns
        -------
        curvature : NumPy array
            A NumPy array storing the computed curvature values.

        """
        if mask is not None and len(mask) == 0:  return np.empty(0)

        if not coarsened and (s is None) and (self._curvature is not None):
            if mask is None:
                return self._curvature.copy()
            else: # mask is given
                return self._curvature[:,mask]

        curve_masks = self._partition_mask( mask )

        curvature = [ curve.curvature( s, cmask, coarsened )
                      for curve,cmask in zip(self._curve_list, curve_masks)]

        curvature = np.hstack( curvature )

        if s is None and mask is None and not coarsened:
            self._curvature = curvature.copy()

        return curvature

    def second_fund_form(self, s=None, mask=None, coarsened=False):
        """Estimates the second fundamental form on nodes of the curves.

        Estimates the second fundamental form on each element using
        the following formula:
            II(s) = K(s) T(s) T(s)^T
        where K(s) is the scalar curvature, interpolated linearly from
        nodal curvature values, T(s) is piecewise linear reconstruction
        of the unit tangent vector, T(s)^T is its transpose.
        The second fundamental curve can be computed for nodes of the
        curve or interior points of the elements denoted by s, on all
        elements or selected elements specified by mask. They can be
        computed for coarsened elements too.

        Parameters
        ----------
        s : float, optional
            An optional local coordinate, a value between 0 and 1.
            This needs to be given if the value of the second fundamental
            form is sought at a local position, say s=0.2, on all elements.
            Its default value is None.
        mask : NumPy array, optional
            A NumPy array of integer indices, indicating the elements
            for which the values of the second fundamental form are sought.
        coarsened : bool, optional
            True if the values of the second fundamental form are sought
            for coarsened elements, otherwise False.

        Returns
        -------
        II : NumPy array
            An array of size 2x2xN storing the components of the second
            fundamental form.

        """
        if mask is not None and len(mask) == 0:  return np.empty((2,2,0))

        if not coarsened and (s is None) and (self._2nd_fund_form is not None):
            if mask is None:
                return self._2nd_fund_form.copy()
            else: # mask is given
                return self._2nd_fund_form[:,mask]

        curve_masks = self._partition_mask( mask )

        II = np.dstack(( curve.second_fund_form( s, cmask, coarsened )
                         for curve,cmask in zip(self._curve_list, curve_masks)))

        if s is None and mask is None and not coarsened:
            self._2nd_fund_form = II.copy()

        return II

    def safe_step_size(self, V, tol=0.3):
        """The maximum safe step size to avoid local geometric distortions.

        If the spacing between the two nodes of an element changes too
        rapidly, this might create distortions in the geometric representation.
        This function computes the maximum safest step size to move the
        curve with the given velocity. The computation is based on the
        following inequality to be enforced

            (dx1 - dx0) / h = dt (V1 - V0).T / h < tol

        where dx1, dx0 denote the changes in the node locations,
        V1, V0 are their velocity, h is the size, T is the tangent
        vector of the element.

        Parameters
        ----------
        V : NumPy array
            The velocity vector, a NumPy array of size (2,n), where n
            is the number of nodes of the curve.
        tol : float
            Allowed ratio of relative tangential displacement with
            respect to the element size. The default value is 0.3.

        Returns
        -------
        dt : float
            The maximum safe step size allowed within the specified
            tolerance.
        """
        dt = np.inf  # initialize dt with inf (to be replaced in the loop)
        offset = 0
        for curve in self._curve_list:
            n = curve.size()
            dt = min( dt, curve.safe_step_size( V[:,offset:offset+n], tol) )
            offset += n
        return dt

    def _enforce_consistency(self):

        curve_list = self._curve_list
        topmost_curve_ids = self._topmost_curve_ids
        parent = self._parent
        children = self._children

        valid_curve_ids = [ curve_id for curve_id in topmost_curve_ids
                            if curve_list[curve_id].orientation() == self._topmost_orientation ]

        # If none of the topmost curves are valid, return by emptying
        # the curve family.
        if len(valid_curve_ids) == 0:
            self._curve_list = []
            self._topmost_curve_ids = []
            self._parent = []
            self._children = []
            self.reset_data()
            return

        # New topmost_curve_ids are only those from valid_curve_ids
        topmost_curve_ids = list( valid_curve_ids )

        # Initialize check_queue with the childen of the topmost curves.
        check_queue = deque( children[ topmost_curve_ids[0] ] )
        for curve_id in topmost_curve_ids[1:]:
            check_queue.extend( children[ curve_id ] )

        # For each curve in check_queue, check if its orientation is
        # the opposite of its parent curve. If not, remove it from
        # its parent's children (thus from the curve hierarchy).
        while len(check_queue) > 0:
            curve_id = check_queue.popleft()
            curve = curve_list[ curve_id ]
            parent_curve = curve_list[ parent[curve_id] ]
            if curve.orientation() != parent_curve.orientation():
                valid_curve_ids.append( curve_id )
                check_queue.extend( children[curve_id] )
            else:
                children[ parent[curve_id] ].remove( curve_id )

        # If the current configuration is consistent, no curves removed, return.
        if len(valid_curve_ids) == len(curve_list):
            return

        # If not all curves are valid, then create the new curve list
        # and the new curve hierarchy as follows.

        valid_curve_ids.sort()

        new_id_map = dict( ( (old_id, new_id)
                             for new_id, old_id in enumerate(valid_curve_ids) ) )
        new_id_map[None] = None

        new_curve_list = [ curve_list[curve_id]
                           for curve_id in valid_curve_ids ]
        self._curve_list = new_curve_list

        new_parent = [ new_id_map[ parent[old_id] ]
                       for old_id in valid_curve_ids ]
        self._parent = new_parent

        new_children = [ set( ( new_id_map[child]
                                for child in children[curve_id] ) )
                         for curve_id in valid_curve_ids ]
        self._children = new_children

        self._topmost_curve_ids = [ new_id_map[curve_id]
                                    for curve_id in topmost_curve_ids ]

        # Reset other curve-related data
        self.reset_data()

        topmost_curve = self._topmost_curve_ids[0]
        self._topmost_orientation = self._curve_list[topmost_curve].orientation()


    def add_remove_curves(self, remove_curve_ids, new_curve_list,
                          recompute_parent_list=[], enforce_consistency=True):

        if (len(remove_curve_ids) == 0) and (len(new_curve_list) == 0):  return

        old_curve_list = self._curve_list
        parent = self._parent
        children = self._children

        # First remove/invalidate the references from parent and
        # children set lists to remove_curve_ids.
        for cid in remove_curve_ids:
            if (parent[cid] is not None) and (parent[cid] >= 0):
                children[ parent[cid] ].remove( cid )
            for child in children[cid]:
                parent[child] = -1  # needs to be updated/recomputed

        for cid in recompute_parent_list:
            parent[cid] = -1

        # Create the list of tuples of
        #    (curve_object, old_id, parent, children_set)
        # from the remaining old curves and the new curves.
        #
        # For curves in recompute_parent_list, the last three entries
        # are (old_id, -1, children set).
        #
        # For the new curves, the last three entries are (-1,-1,set()).
        # Sort the list w.r.t. areas of the curves as a first step
        # to create the curve hierarchy.

        # Add the information for the unaffected curves.
        curve_info_list = [ (curve, cid, parent[cid], children[cid])
                            for cid, curve in enumerate(old_curve_list)
                            if cid not in remove_curve_ids ]
        # Add the new curves to the curve_info list.
        curve_info_list.extend( ( (curve, -1, -1, set())
                                  for curve in new_curve_list) )
        # Sort the curve info list w.r.t. the magnitude of the curve area.
        curve_sort_key = lambda curve_info: np.abs( curve_info[0].area() )
        curve_info_list.sort( key=curve_sort_key, reverse=True )

        # After sorting, the order of the old curves in the new curve
        # list has changed; therefore, their ids have changed as well,
        # so we define the new id map that gives the new ids from
        # the old ids of the old curves.

        new_id_map = dict( ( (old_id, new_id)
                             for new_id, (curve, old_id, parent, children)
                             in enumerate(curve_info_list)
                             if old_id != -1) ) # new curves have old_id == -1
        new_id_map[None] = None
        new_id_map[-1] = -1

        # Define the new curve_list, parent and children set lists.
        # Parent and children will need to be updated afterwards
        # to accommodate for the new curves and those of the old ones
        # that have lost their parents.

        curve_list = [ info[0] for info in curve_info_list ]
        self._curve_list = curve_list

        parent = [ new_id_map[ info[2] ] for info in curve_info_list ]
        self._parent = parent

        children = [ set( (new_id_map[child] for child in info[3]) )
                     for info in curve_info_list ]
        self._children = children

        # Now create the list of curves to be updated (from the new ones
        # and the old ones). These have parent == -1 (i.e. unknown parent).

        update_curves = ( (curve_id, curve_info[0]) # the curve object and its id
                          for curve_id, curve_info in enumerate(curve_info_list)
                          if curve_info[2] == -1 ) # if parent unknown

        # Iterate over all the update curves, check with those that have
        # larger area (preceding in the curve_list) to see if the update
        # curve is contained in the one with larger area.

        for curve_id, curve in update_curves:
            parent[ curve_id ] = None # default: topmost curve, no parent
            for possible_parent in range(curve_id-1,-1,-1):
                possible_parent_curve = curve_list[ possible_parent ]
                if possible_parent_curve.contains( curve ):
                    parent[ curve_id ] = possible_parent
                    children[ possible_parent ].add( curve_id )
                    break

        self._topmost_curve_ids = [ curve_id
                                    for curve_id, parent_id in enumerate(parent)
                                    if parent_id is None ]

        if len(self._topmost_curve_ids) > 0:
            curve_id = self._topmost_curve_ids[0]
            self._topmost_orientation = curve_list[ curve_id ].orientation()

        # At point, the new curve list, parent list and children set list
        # have been created. If needed, consistency of this new hierarchy
        # can be examined and corrected by removing the inconsistent curves.
        # Inconsistent curves are those that have the same orientation
        # with their parents.

        if enforce_consistency:  self._enforce_consistency()
        self.reset_data()
