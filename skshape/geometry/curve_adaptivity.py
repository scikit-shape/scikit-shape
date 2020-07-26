"""Functions to estimate errors and adjust curve node sampling.

This module contains functions for curve adaptivity, based on geometric
and data-based error estimation, implemented via curve element/edge
refinement or coarsening.

"""

import numpy as np
from numpy import nan, logical_and as np_AND, logical_not as np_NOT
from ..numerics.integration import Quadrature1d
from ..numerics import marking
from copy import copy, deepcopy


##############  Error estimation functions  ####################

def compute_geometric_error(curve, mask=None, coarsened=False,
                            vec=None, parameters=None):
    """Computes the discretization error for a given Curve object.

    This function takes a Curve object as input and returns an estimate
    of the maximum pointwise error per element, incurred by the geometric
    discretization represented by the Curve object. To be specific,
    the output is the error vector defined as follows

         error[i] = max|X(s) - x(s)| for all s on element[i],

    where X(s) is the true curve coordinate and x(s) is the coordinate
    given by the discretized curve representation.
    The actual error is computed by the following estimate

         error[i] = max{K1,K2} h^2,

    where K1,K2 are the curvature values at the 1st and 2nd nodes of
    the element[i] respectively, and h is the size of the element.

    If a mask vector is given, the error is computed only for the elements
    specified by mask.

    If the argument vec is given, the error is returned in vec.
    Otherwise a new Numpy array is allocated as the error vector.

    If coarsen is set to be True, then the error values are not computed
    for the existing elements of the curve. They are computed for the elements
    that would result from the coarsening of the current elements. For example,
    if the current element is given by (X[i],X[i+1]) and its next element is
    (X[i+1],X[i+2]) (X being the coordinates of the nodes), then the coarsened
    element would be (X[i],X[i+2]), and these are element coordinates used for
    computation when coarsened=True.

    Parameters
    ----------
    curve : :obj:'Curve'
        A instance of a Curve class, storing the curve geometry.
    mask : NumPy array, optional
        An optional argument to be provided if the error is computed
        for some of the elements only. It can be a boolean NumPy array
        of the same size as the curve or it can be an integer NumPy array
        of element indices.
    coarsened : bool, optional
        True if the errors are being computed for coarsened elements,
        otherwise False.
    vec : NumPy array, optional
        An optional output vector. If it is provided, then the computed
        error values are written to vec, and vec is returned is the error
        vector.
    parameters : dict, optional
        An optional and unused argument, included in the function definition
        for compatibility.

    Returns
    -------
    error : NumPy array
        A NumPy array of real numbers, storing the computed error values
        for all elements. It is the same size as the curve.

    """
    el_sizes = curve.element_sizes( mask, coarsened )
    K = curve.curvature( None, mask, coarsened )
    m = len(K)
    n = curve.size()

    if (mask is not None) and (mask.dtype == bool):  mask = mask.nonzero()[0]
    if (mask is not None) and (len(mask) == 0):  return np.empty(0)

    if mask is None:
        error = np.abs(K)
        error[0:m-1] = np.maximum( error[0:m-1], np.abs( K[1:m] ) )
        error[m-1] = max( error[m-1], np.abs(K[0]) )
    else: # mask is given
        mask2 = (mask + 1) % n
        if not coarsened:
            K2 = curve.curvature( None, mask2, coarsened=False )
        else: # coarsened
            K2 = curve.curvature( 1.0, mask, coarsened=True )
        error = np.maximum( np.abs(K), np.abs(K2) )

    error *= el_sizes**2

    n = curve.size()
    if vec is None:  vec = np.empty(n)

    if mask is not None:
        vec[mask] = error
        if coarsened:  vec[mask2] = error
    else: # mask is None
        if not coarsened:
            vec[:] = error
        else:
            vec[0:n-2:2] = error[0:m-1]
            vec[1:n-1:2] = vec[0:n-2:2]
            vec[n-1] = error[m-1]
            if (n % 2) == 1:  vec[n-2] = vec[n-1]

    return vec


def compute_data_error(curve, mask=None, coarsened=False,
                       vec=None, parameters=None):
    """Computes the integration error for given f on each element.

    This function computes an approximation to the integration error on
    each element for a given mesh function f. The integration error is
    estimated by performing high and low order quadratures on each element,
    multiplying the quadrature sums by element sizes and taking the difference
    of the results for the high and low orders (default values 2 and 3
    respectively). The error estimates are normalized by dividinng by
    curve length.

    If the mask vector is given, then error is computed for only the elements
    specified by mask.

    If vec is provided the user, the computed values are stored in vec and
    vec is returned as the error vector. Otherwise a new Numpy array is
    allocated as the error vector.

    The parameters for this function have to be specified and should include
    the data function f that is being integrated. The function f can be
    a MeshFunction defined on the curve.

    If coarsen is set to be True, then the error values are not computed
    for the existing elements of the curve. They are computed for the elements
    that would result from the coarsening of the current elements. For example,
    if the current element is given by (X[i],X[i+1]) and its next element is
    (X[i+1],X[i+2]) (X being the coordinates of the nodes), then the coarsened
    element would be (X[i],X[i+2]), and these are element coordinates used for
    computation when coarsened=True.

    Parameters
    ----------
    curve : :obj:'Curve'
    mask : boolean or integer NumPy array, optional
        An optional argument to be provided if the error is computed
        for some of the elements only. It can be a boolean NumPy array
        of the same size as the curve or it can be an integer NumPy array
        of element indices.
    coarsened : bool, optional
        An optional boolean value that indicates whether the errors
        are being computed for the current elements (coarsened=False),
        or for those obtained by coarsening the current elements
        (coarsened=True).
    vec : NumPy array, optional
        An optional output vector. If it is provided, then the computed
        error values are written to vec, and vec is returned is the error
        vector.
    parameters : dict
        An parameter dictionary that has to be provided. Its format is:
        {'data function':f,'low quadrature':order0,'high quadrature':order1},
        where f is an instance of a MeshFunction; f should be able to take
        the arguments in the form f(curve,s,mask,coarsened), where s is
        the local parametric coordinate on the element in the interval [0,1].
        The quadrature orders, order0 and order1, are optional integer values.

    Returns
    -------
    error : NumPy array
        The computed error values for all elements. It is the same size
        as the curve.
    """
    f = parameters['data function']

    if (mask is not None) and (mask.dtype == bool):  mask = mask.nonzero()[0]
    if (mask is not None) and (len(mask) == 0):  return np.empty(0)

    quad0 = Quadrature1d( order = parameters.get('low quadrature', 2) )
    quad1 = Quadrature1d( order = parameters.get('high quadrature',3) )

    error  = np.abs( sum(( weight * f(curve, pt, mask, coarsened)
                           for pt, weight in quad0.iterpoints() )) +
                     sum(( (-weight) * f(curve, pt, mask, coarsened)
                           for pt, weight in quad1.iterpoints() )) )

    el_sizes = curve.element_sizes( mask, coarsened )
    error *= el_sizes
    error /= curve.length()  # normalization for curves of different length

    n = curve.size()
    if vec is None:  vec = np.empty(n)

    if mask is not None:
        vec[mask] = error
        if coarsened:  vec[(mask+1)%n] = error
    else: # mask is None
        if not coarsened:
            vec[:] = error
        else:
            m = len(error)
            vec[0:n-2:2] = error[0:m-1]
            vec[1:n-1:2] = vec[0:n-2:2]
            vec[n-1] = error[m-1]
            if (n % 2) == 1:  vec[n-2] = vec[n-1]

    return vec


def compute_data_coarsening_error(curve, mask=None, vec=None, parameters=None):
    """Estimates the integration error on each element in the case of coarsening.

    This function computes an approximation to the integration error
    on each element for a given mesh function f after coarsening the curve.
    For example, if the current element is given by (X[i],X[i+1]) and its
    next element is (X[i+1],X[i+2]) (X being the coordinates of the nodes),
    then the coarsened element would be (X[i],X[i+2]), and these are element
    coordinates used for computation. The integration error is estimated
    by performing quadratures of order 0 and 1 on each coarsened element,
    scaling the quadrature sums by element sizes and taking the difference
    of the results for 0 and 1.

    If the mask vector is given, then error is computed for only the elements
    specified by mask.

    If vec is provided the user, the computed values are stored in vec and
    vec is returned as the error vector.

    The parameters for this function have to be specified and should include
    the data function f that is being integrated.

    Parameters
    ----------
    curve : :obj:'Curve
    mask : boolean or integer NumPy array, optional
        An optional argument to be provided if the error is computed
        for some of the elements only. It can be a boolean NumPy array
        of the same size as the curve or it can be an integer NumPy array
        of element indices.
    vec : NumPy array, optional
        An optional output vector. If it is provided, then the computed
        error values are written to vec, and vec is returned is the error
        vector.
    parameters : dict
        An parameter dictionary that has to be provided. Its format is:
        {'data function':f}, where f is an instance of a MeshFunction;
        f should be able to take the arguments in the form
        f(curve,s,mask,coarsened), where s is the local parametric
        coordinate on the element in the interval [0,1].

    Returns
    -------
    error : NumPy array
        The computed error values for all elements. It is the same size
        as the curve.
    """
    return compute_data_error( curve, mask, True, vec, parameters )


def estimates_and_markers(curve, current_info, parameters):
    """Estimates the errors of elements and marks them for refinement/coarsening.

    This function executes a set of error estimator functions specified
    in the parameters and computes the error values for each element.
    Once each error estimator is executed, the corresponding marking
    function is also executed. The elements with high errors are marked
    as 1 for refinement and those with too low error are marked as -1
    for coarsening (the remaining elements are marked as 0).

    A set of functions for coarsening error (and their corresponding marking
    functions) can also be provided in parameters. These function are executed
    only on the elements marked for coarsening. If the coarsening error is too
    high, these elements are unmarked, i.e. the corresponding locations in
    the markers array are set to 0.

    Parameters
    ----------
    curve : :obj:'Curve'
    current_info : tuple of list of NumPy arrays
        A pair of a list of error vectors and a NumPy array of
        integer indices indicating which elements have been updated.
        If this information is not available, its value is None.
        The error vectors should be as many as the error functions
        specified in parameters. Each error vector is a NumPy array of
        real values with a length equal to the size of the curve.
        The error vectors can contain previously computed error values
        or 'nan' values indicating invalid or uncomputed values.
    parameters : dict
        A dictionary of parameters that should contain the error
        functions and the corresponding marking functions (from
        geometry.marking), paired with their parameters. Optionally
        it also contains the coarsening error function. The following
        is an example dictionary of parameters:

        pair1 = ( (compute_geometric_error, None),
                  (fixed_el_tol_marking, {'tolerance': 0.01,
                                          'gamma_coarsen': 0.2} ) )

        pair2 = ( (compute_data_error, {'data function':f}),
                  (equidistribution_marking, {'tolerance':    0.0001,
                                              'gamma_refine': 0.7,
                                              'gamma_coarsen':0.05} ) )

        pair3 = ( (compute_coarsen_error,{'data function':f}),
                  (equidistribution_marking, {'tolerance':    0.0001,
                                              'gamma_refine': 0.7,
                                              'gamma_coarsen':0.05} ) )

        params = { 'errors and marking': [ pair1, pair2 ],
                   'coarsening errors and marking': [ pair3 ] }

    Returns
    -------
    markers : NumPy array
        Array of integer markers with length equal to curve size.
        If element[i] is to be refined, then markers[i] = 1.
        If element[i] is to be coarsened, then markers[i] = -1.
        Otherwise markers[i] = 0.
    """
    # Process each error array in the errors list (of arrays) using
    # the corresponding marking strategy.
    # Then use a conservative strategy to combine all the markers.
    # If any of the strategies says the element should be refined
    # (marker has a positive int value), then it is marked to be refined.
    # An element is marked to be coarsened only if all the markers
    # say that it should be coarsened (marker has a negative int value).

    n = curve.size()

    ####  Error estimation and marking based on error estimates  ####

    n_err_func = len( parameters['errors and marking'] ) \
                 + len( parameters['coarsening errors and marking'] )

    ####  First error vector and corresponding markers computed  ####

    # Get marking strategy and the error function for the first error vector.
    (error_func, error_params), (marking_func, marking_params) \
                 = parameters['errors and marking'][0]

    # If errors is None, we need to compute the errors from scratch.
    # We start by computing all the error values for all elements using
    # the first error function. The result is stored in error_vecs[0].
    # If errors is not None and stores the error vectors with possible
    # nan values at some locations, we recompute those for the first
    # error vector.

    if current_info is not None and current_info[0] is not None:
        error_vecs, indices = current_info
        # Compute the missing values of the first error vector.
        if indices is None: indices = np.nonzero(np.isnan( error_vecs[0] ))[0]
        error_func( curve, indices, False, error_vecs[0], error_params )
    else: # errors is None, need to initialize all the error vectors.
        # Compute the first error vector from scratch.
        error_vecs = [ error_func( curve, parameters=error_params ) ]
        # Initialize the other error vectors with nan, to be computed later.
        error_vecs.extend( (np.empty(n)+nan for k in range(n_err_func-1)) )

    # We initialize the markers vectors using the first error vector
    # and the corresponding marking strategy.
    r_markers, c_markers = marking_func( error_vecs[0], marking_params )

    ####  Remaining error vectors and corresponding markers computed  ####

    ####  Here is how it works in the following:                      ####
    ####  - If at least one error func says refine, then we refine.   ####
    ####  - If all error funcs say coarsen (no refine), then coarsen, ####

    for k, ((error_func, error_params), (marking_func, marking_params)) \
            in enumerate( parameters['errors and marking'][1:] ):
        error_vec = error_vecs[k+1]
        # Compute the errors only for the elements not marked to be refined.
        mask = np.isnan( error_vec ) # indices of missing error estimates
        mask[ r_markers ] = False # no need to compute for those marked already
        mask = mask.nonzero()[0]
        error_func( curve, mask, False, error_vec, error_params )
        # Now corresponding marking strategy to mark elements for refinement.
        r_indices, c_indices = marking_func( error_vec, marking_params, mask )
        r_markers = np.hstack(( r_markers, r_indices ))

    offset = len( parameters['errors and marking'] )
    for k0, ((error_func, error_params), (marking_id, marking_params)) \
            in enumerate( parameters['coarsening errors and marking'] ):
        error_vec = error_vecs[ offset + k0 ]
        # Compute the errors only for those marked for coarsening.
        mask = c_markers[ np.isnan(error_vec[c_markers]) ]
        error_func( curve, mask, error_vec, error_params )
        # Use the corresponding marking strategy to identify the elements
        # with high coarsening error (marking_func gives marker indices).
        r_indices, c_markers = marking_func(error_vec, marking_params, c_markers)

    markers = np.zeros( n, dtype=int )
    markers[ c_markers ] = -1  #!!! c_markers and r_markers may share indices,
    markers[ r_markers ] =  1  #!!! therefore r_markers should come after
                               #!!! c_markers, b/c refine overrides coarsen.

    # If an element size is too large, it should be refined, regardless
    # of the value of the marker. Similarly if an element size is too small,
    # it should be coarsened.

    el_sizes = curve.element_sizes()
    min_size = parameters['minimum element size']
    max_size = parameters['maximum element size']
    refine_indices  = np.nonzero( el_sizes > max_size )[0]
    coarsen_indices = np.nonzero( el_sizes < min_size )[0]
    coarsen_indices = np.hstack(( coarsen_indices, (coarsen_indices+1) % n ))
    markers[ refine_indices ] = 1    # !!! in the case of very small elements,
    markers[ coarsen_indices ] = -1  # !!! coarsen overrides refine.

    # If elements k & k+1 are to be coarsened and combined into
    # a single element, only element k should be marked with -1,
    # element k+1 should be marked with 0.

    has_neg_neighbor = np.zeros( n, dtype=bool )
    coarsen_indices = np.nonzero( markers < 0 )[0]
    even = coarsen_indices[ coarsen_indices % 2 == 0 ]
    odd = coarsen_indices[ coarsen_indices % 2 == 1 ]
    next_of_even = (even + 1) % n
    next_of_odd = (odd + 1) % n
    prev_of_odd = (odd - 1) % n
    has_neg_neighbor[even] = markers[next_of_even] < 0
    has_neg_neighbor[odd] = np_AND( np_NOT( has_neg_neighbor[prev_of_odd] ),
                                    np_AND( markers[next_of_odd] < 0,
                                            np_NOT(has_neg_neighbor[next_of_odd]) ))
    # The element and its immediate neighbor should be marked for coarsening.
    # If not, reset that element's negative marker to zero.
    markers[ np_AND(markers < 0, np_NOT(has_neg_neighbor) ) ] = 0

    if markers[0] < 0 and markers[n-1] < 0:  markers[n-1] = 0

    if n <= 4:  markers[ markers < 0 ] = 0  # Don't coarsen small curves.

    return (markers, error_vecs)


def _refinement_new_nodes(curve, refine_indices, refinement_method='curved'):
    coords = curve.coords()
    n = coords.shape[1]
    j = refine_indices

    if refinement_method == 'linear':
        # For the linear method, the new node is given by the midpoint
        # of the element, i.e. the mean of the its nodes.
        new_nodes = 0.5*(coords[:,j] + coords[:,(j+1)%n])
        return new_nodes

    elif refinement_method == 'curved':
        # For 'curved' refinement, the midpoint of the element is taken,
        # and projected along the normal to a fictitious curve matching
        # the curvature, such that the curvature of the new node is
        # the average of those of the two old nodes.
        d = curve.element_sizes()
        N = curve.normals( smoothness='pwconst' )
        K = curve.curvature()
        # mid_pt = (pt0 + pt1) / 2
        new_nodes = 0.5*(coords[:,j] + coords[:,(j+1)%n])
        # k = (k0 + k1)/2
        k = 0.5 * (K[j] + K[(j+1)%n])
        # a = r - sqrt(r^2 - d^2/4) = 0.25 d^2 k / (1 + sqrt(1 - d^2 k^2/4))
        a = 0.25 * d[j] * d[j] * k
        a /= 1.0 + np.sqrt(1.0 - a*k)
        # new_node = mid_pt + a * N
        new_nodes[0,:] += a * N[0,j]
        new_nodes[1,:] += a * N[1,j]
        return new_nodes
    else:
        raise ValueError('Invalid refinement method!')


def _create_curve_segments(curve, data_vectors, markers,
                           refinement_method='curved'):

    coords = curve.coords()
    n = coords.shape[1]

    # Create some dummy nan-valued arrays, to be used as fillers
    # for the data vector pieces of the new elements.
    filler_vec1 = { 1: (nan*np.ones(1)), 2: (nan*np.ones((2,1))) }
    filler_vec2 = { 1: (nan*np.ones(2)), 2: (nan*np.ones((2,2))) }

    # Initialize curve_segments, segment_start_nodes & _end_nodes
    curve_segments = {}
    segment_start_nodes = set()
    segment_end_nodes = set()

    # The new elements and the corresponding data vector pieces
    # created by coarsening.
    coarsen_indices = np.nonzero( markers < 0 )[0]
    # Coarsening markers have to be followed by 0 markers
    if np.sum(np.abs( markers[(coarsen_indices+1)%n] )):
        raise ValueError("Invalid marker array!")

    if len(coarsen_indices) > 0:
        vecs = [ filler_vec1[vec.ndim] for vec in data_vectors ]

        curve_segments.update(( (k, ((k+2)%n, coords[:,k].reshape(2,1), vecs))
                                for k in coarsen_indices ))
        segment_start_nodes.update( (coarsen_indices + 2) % n )
        segment_end_nodes.update( coarsen_indices )

    # The new elements and the corresponding data vector pieces
    # created by refinement.
    refine_indices = np.nonzero( markers > 0 )[0]

    if len(refine_indices) > 0:
        new_nodes = _refinement_new_nodes( curve, refine_indices,
                                           refinement_method )
        vecs = [ filler_vec2[vec.ndim] for vec in data_vectors ]

        for k0,k in enumerate(refine_indices):
            new_coords = np.hstack((coords[:,k].reshape(2,1),
                                    new_nodes[:,k0].reshape(2,1)))
            curve_segments[k] = ( (k+1)%n, new_coords, vecs )

        segment_start_nodes.update( (refine_indices + 1) % n )
        segment_end_nodes.update( refine_indices )

    # The start and end nodes for curve segments
    shared_nodes = segment_start_nodes.intersection( segment_end_nodes )
    segment_start_nodes -= shared_nodes
    segment_end_nodes   -= shared_nodes
    segment_start_nodes = sorted( segment_start_nodes )
    segment_end_nodes   = sorted( segment_end_nodes )

    # Segment boundaries: list of pairs (start,end) of a segment
    if (len(segment_start_nodes) == 0) or (len(segment_end_nodes) == 0):
        segment_boundaries = []
    elif segment_start_nodes[0] < segment_end_nodes[0]:
        segment_boundaries = zip( segment_start_nodes, segment_end_nodes )
    else:
        segment_boundaries = zip( segment_start_nodes[:-1], segment_end_nodes[1:] )
        segment_boundaries.append((segment_start_nodes[-1], segment_end_nodes[0]))


    # Curve segments: a dictionary where the key is the start node
    # and the value is the end node, coords and vector pieces of
    # nodes from start to end-1.

    if len(segment_boundaries) > 0:

        for start, end in segment_boundaries[:-1]: # all pairs except the last
            coord_piece = coords[:,start:end]
            coord_piece.reshape( 2, end-start )
            curve_segments[start] = (end, coord_piece, [])
        # The last (start,end) pair might need special handling.
        start, end = segment_boundaries[-1]
        if start < end:
            coord_piece = coords[:,start:end].reshape( 2, end-start )
        else:
            coord_piece = np.hstack(( coords[:,start:], coords[:,:end] ))
        curve_segments[start] = (end, coord_piece, [])

        # Now the pieces of the data vectors.
        for vec in data_vectors:
            for start, end in segment_boundaries[:-1]:
                if   vec.ndim == 1:  vec_piece = vec[start:end]
                elif vec.ndim == 2:  vec_piece = vec[:,start:end]
                elif vec.ndim == 3:  vec_piece = vec[:,:,start:end]
                curve_segments[start][2].append( vec_piece )
            # The last (start,end) pair is a special case.
            start, end = segment_boundaries[-1]
            if start < end:
                if   vec.ndim == 1:  vec_piece = vec[start:end]
                elif vec.ndim == 2:  vec_piece = vec[:,start:end]
                elif vec.ndim == 3:  vec_piece = vec[:,:,start:end]
            else:
                if vec.ndim == 1:
                    vec_piece = np.hstack(( vec[start:], vec[:end] ))
                elif vec.ndim == 2:
                    vec_piece = np.hstack(( vec[:,start:], vec[:,:end] ))
                elif vec.ndim == 3:
                    vec_piece = np.dstack(( vec[:,:,start:], vec[:,:,:end] ))
            curve_segments[start][2].append( vec_piece )

    return curve_segments


def _stack_vec_list(vec_list):
    ndims = vec_list[0].ndim
    if ndims == 1:
        length = np.sum(( vec.shape[0] for vec in vec_list ))
        new_vec = np.empty(length)
        offset = 0
        for vec in vec_list:
            vec_size = vec.shape[0]
            new_vec[offset:offset+vec_size] = vec
            offset = offset + vec_size
    elif ndims == 2:
        length = np.sum(( vec.shape[1] for vec in vec_list ))
        new_vec = np.empty((vec_list[0].shape[0],length))
        offset = 0
        for vec in vec_list:
            vec_size = vec.shape[1]
            new_vec[:,offset:offset+vec_size] = vec
            offset = offset + vec_size
    else:
        raise Exception("Cannot handle data vectors of ndim > 3!")

    return new_vec


def _reassemble_curve_and_data(curve_segments):

    key, value = curve_segments.popitem()
    curve_start = start = key
    end, coords, vector_pieces = value
    data_vectors = [ [vec] for vec in vector_pieces ]
    data_vectors.append( [coords] )

    while end != curve_start:
        start = end
        end, coords, vector_pieces = curve_segments.pop( start )
        for i, vec in enumerate(vector_pieces):
            data_vectors[i].append( vec )
        data_vectors[-1].append( coords )

    new_data_vectors = [ _stack_vec_list(veclist) for veclist in data_vectors ]

    return new_data_vectors


def _compute_element_sizes(coords, j, element_sizes):
    n = coords.shape[1]

    x0, y0 = coords[0,j], coords[1,j]
    x1, y1 = coords[0,(j+1)%n], coords[1,(j+1)%n]

    element_sizes[j] = np.sqrt( (x0 - x1)**2 + (y0 - y1)**2 )

    return element_sizes


def _compute_element_normals(coords, element_sizes, j, normals):
    n = coords.shape[1]

    x0, y0 = coords[0,j], coords[1,j]
    x1, y1 = coords[0,(j+1)%n], coords[1,(j+1)%n]

    normals[0,j] = y1 - y0
    normals[1,j] = x0 - x1

    if element_sizes is not None:
        normal_size = element_sizes[j]
    else:
        normal_size = np.sqrt( normals[0,j]**2 + normals[1,j]**2 )

    normals[0,j] /= normal_size
    normals[1,j] /= normal_size

    return normals


def _compute_element_curvatures(coords, j, K):
    n = coords.shape[1]

    j1, j2 = j, (j+1)%n
    x0, y0 = coords[0,(j-1)%n], coords[1,(j-1)%n]
    x1, y1 = coords[0, j1],     coords[1, j1]
    x2, y2 = coords[0, j2],     coords[1, j2]
    x3, y3 = coords[0,(j+2)%n], coords[1,(j+2)%n]

    bottom_sqr = ((x2 - x1)**2 + (y2 - y1)**2) * \
                 ((x2 - x0)**2 + (y2 - y0)**2) * \
                 ((x1 - x0)**2 + (y1 - y0)**2)

    K[j1] = x0 * (y1 - y2) + x1 * (y2 - y0) + x2 * (y0 - y1)
    K[j1] = 2.0 * K[j1] / np.sqrt( bottom_sqr )

    bottom_sqr = ((x2 - x1)**2 + (y2 - y1)**2) * \
                 ((x2 - x3)**2 + (y2 - y3)**2) * \
                 ((x1 - x3)**2 + (y1 - y3)**2)

    K[j2] = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
    K[j2] = 2.0 * K[j2] / np.sqrt( bottom_sqr )

    return K


def refine_coarsen(curve, markers, data_vectors, refinement_method='curved'):
    """Refines/coarsens elements of a curve following a given markers array.

    Given a curve object and an array of markers, this function refines
    (adds nodes) and coarsens (removes nodes) the elements of the curve.
    Adding and removing nodes changes the ordering of the coordinates of
    the curve's nodes. For example, if the old node indices are 0,1,2,3,4
    and the elements are (0,1),(1,2),(2,3),(3,4),(4,0), then with a markers
    array of [1,-1,0,1,0], the changes in the elements is:
       (0,1) refine => (0,1.5),(1.5,1) remapped to (0,1),(1,2)
       (1,2),(2,3) coarsen => (1,3) remapped to (2,3)
       (3,4) refine => (3,3.5),(3.5,4) remapped to (3,4),(4,5)
       (4,0) no change => (4,0) remapped to (5,0)
    The nodes of the new curve are 0,1,2,3,4,5 and the elements are
    (0,1),(1,2),(2,3),(3,4),(4,5),(5,0).

    If some data is associated with the old elements (specified in the list
    of arrays, data_vectors), it can be reassigned to the new corresponding
    elements. The data_vectors are recreated as new_data_vectors and the data
    is copied for the unchanged elements, the values for the new elements
    resulting from refinement or coarsening are set to nan.

    The user can also control where the nodes created by specifying the
    refinement_method. If refinement_method is 'linear', then the new node
    is created between the two nodes of the element. If it is 'curved',
    then the new node is created off the element along the normal from
    the midpoint of the element in a way to match the average curvature
    of the element.

    Parameters
    ----------
    curve : :obj:'Curve'
    markers : NumPy array
        An array of integers, the same length as the curve size.
        It specifies if an element will be refined or coarsened.
        If markers[i] = 1, then element[i] will be refined.
        If markers[i] = -1, then element[i] will be coarsened.
        If markers[0] = 0, then element[i] will remain the same.
    data_vectors : list of NumPy arrays
        Each of the arrays in the list should be the same length
        as the size of the curve, for example, the arrays can be
        of size C or (n1,C) or (n1,n2,C), C being the curve size.
        The data vectors represent data associated with elements.
        They will remapped and reassigned to match the new indices
        of the elements.
    refinement_method : str
        The refinement method specifies where the new node will be
        created. Its value is one of 'linear' or 'curved'.
        If refinement_method is 'linear', then the new node is created
        between the two nodes of the element.
        If refinement_method is 'curved', then the new node is created
        off the element along the normal from the midpoint of the element
        in a way to match the average curvature of the element.

    Returns
    -------
    new_data_vectors : list of NumPy arrays
        The remapped data vectors created from the original data vectors,
        a list of NumPy arrays, storing the data associated with the new
        data. The values of the unchanged elements are copied to the new
        corresponding element indices. The data values for the new elements
        resulting from refinement and coarsening are set to nan.
    new_indices : NumPy array
        An array of integers giving the indices/locations of the new elements.

    """
    # Return if any element is NOT marked for refinement or coarsening.
    if not markers.any():  return data_vectors

    # Check if the markers vector and data_vectors are the correct size.
    n = curve.size()
    if len(markers) != n:
        raise ValueError("Markers vector should be the same size as the curve.")
    for data_vec in data_vectors:
        if data_vec.shape[0] != n:
            raise ValueError("Data vectors should be the same size as the curve.")

    # Add el sizes, normals and curvature vectors to data_vectors,
    # because they will also be split into pieces and reassembled.
    data_vectors.append( curve.element_sizes() )
    data_vectors.append( curve.normals( smoothness='pwconst' ) )
    data_vectors.append( curve.curvature() )

    # Delete elements for coarsening and refinement and create new elements.
    # This separates the curve into untouched and new segments.
    curve_segments = _create_curve_segments( curve, data_vectors, markers,
                                             refinement_method )

    # Put the curve segments together and create the new coordinates
    # and the data vectors.
    data_vectors = _reassemble_curve_and_data( curve_segments )

    new_el_sizes, new_normals, new_curvature, new_coords = data_vectors[-4:]

    # Compute the missing parts of the geometry (el_sizes, normals, curvature).
    indices = np.nonzero( np.isnan( new_el_sizes ) )[0]

    _compute_element_sizes( new_coords, indices, new_el_sizes )
    _compute_element_normals( new_coords, new_el_sizes, indices, new_normals )
    _compute_element_curvatures( new_coords, indices, new_curvature )

    curve.set_geometry( new_coords, new_el_sizes, new_normals, new_curvature )

    # Compute the missing parts of the given data vectors
    new_data_vectors = []
    for i, vec in enumerate(data_vectors[:-4]):
        # data_func, params = data_funcs[i]
        # new_vec = data_func( curve, indices, vec, params )
        new_vec = vec
        new_data_vectors.append( new_vec )

    return (new_data_vectors, indices)


##############  The main adaptivity function  ####################

marking_default_parameters = {'maximum':{'tolerance': 0.01,
                                         'gamma_refine': 0.7,
                                         'gamma_coarsen': 0.2 },

                              'fixed element tol':{'tolerance': 0.01,
                                                   'gamma_coarsen': 0.2 },

                              'equidistribute':{'tolerance': 0.01,
                                                'gamma_refine': 0.7,
                                                'gamma_coarsen': 0.05 }
                              }

## default_adaptivity_parameters = \
##     {
##     'maximum iterations': 5,

##     'minimum element size': 1e-8,

##     'maximum element size': 1000.0,

##     'errors and marking': [
##             ( (compute_geometric_error, None),
##               ('fixed element tol',
##                marking_default_parameters['fixed element tol']) ),

##             ( (compute_data_error, {'data function':None}),
##               ('equidistribute',   {'tolerance':    0.01,
##                                     'gamma_refine': 1.0,
##                                     'gamma_coarsen':0.05} ) )
##             ],

##     'coarsening errors and marking': [
##             ( (compute_data_coarsening_error,{'data function':None}),
##               ('equidistribute', {'tolerance':    0.01,
##                                   'gamma_refine': 1.0,
##                                   'gamma_coarsen':0.05} ) ) ]
##     }


_geometric_adaptivity_parameters = \
    {
    'maximum iterations': 5,

    'minimum element size': 1e-8,

    'maximum element size': 1000.0,

    'errors and marking': [
            ( (compute_geometric_error, None),
              (marking.fixed_el_tol_marking,
               marking_default_parameters['fixed element tol']) )
            ],

    'coarsening errors and marking': [ ],

    'refinement method': 'curved'
    }

def copy_adaptivity_parameters(params=None, new_curves=None):
    if params is None:  params = _geometric_adaptivity_parameters

    new_params = {}
    if 'maximum iterations' in params:
        new_params['maximum iterations'] = params['maximum iterations']
    if 'minimum element size'  in params:
        new_params['minimum element size'] = params['minimum element size']
    if 'maximum element size' in params:
        new_params['maximum element size'] = params['maximum element size']
    if 'refinement method' in params:
        new_params['refinement method'] = params['refinement method']

    def copy_error_param(param_in,curve):
        if param_in is None:  return None
        param_out = {}
        for key,item in param_in.iteritems():
            if key == 'data function':
                item2 = copy(item) # because item also will have data
            else:                  # which we don't want to copy.
                item2 = deepcopy(item)
            param_out[key] = item2
        return param_out

    if 'errors and marking' in params:
        new_params['errors and marking'] = \
            [ ( (func1, copy_error_param(param1, new_curves)),
                (func2, deepcopy(param2)) )
              for (func1,param1),(func2,param2) in params['errors and marking'] ]

    if 'coarsening errors and marking' in params:
        new_params['coarsening errors and marking'] = \
            [ ( (func1, copy_error_param(param1, new_curves)),
                (func2, deepcopy(param2)) )
              for (func1,param1),(func2,param2) in params['coarsening errors and marking'] ]

    return new_params

def geometric_adaptivity_parameters():
    """Returns a new copy of the geometric adaptivity parameters for curves.

    Returns a new copy of the geometric adaptivity parameters for curves.
    This is a dictionary with the following content.

    Returns
    -------
    params : dict
        A copy of the following dictionary of parameters:
        { 'maximum iterations': 5,
          'minimum element size': 1e-8,
          'maximum element size': 1000.0,
          'refinement method':'curved'
          'errors and marking': [
                ( (compute_geometric_error, None),
                  (fixed el tol_marking,
                   marking_default_parameters['fixed element tol']) ) ],
          'coarsening errors and marking': []  }

    """
    return copy_adaptivity_parameters( _geometric_adaptivity_parameters )


def adapt( curve, parameters=geometric_adaptivity_parameters() ):
    """Adapts a given curve by refinement and coarsening of the curve elements.

    Given a list of error estimator and marking function pairs (and their
    parameters) defined in the parameters argument, this function adapts
    the given curve, namely it refines and coarsens elements of the curve
    by adding or removing nodes. The adaptation is performed iteratively
    (up to a maximum number of iterations), by estimating element errors
    and marking the elements for refinement or coarsening according to the
    error and marking criteria. Minimum and maximum element sizes are also
    specified in the parameters so that the elements are not refined beyond
    the minimum element size or coarsened beyond the maximum element size.

    Parameters
    ----------
    curve : :obj:'Curve
    parameters : dict
        A dictionary of the parameters for adaptivity. It includes
        * a list of the error estimation and marking functions pairs and
        their parameters,
        * a list of the coarsening error estimation and marking function
        pairs and their parameters,
        * the maximum number of adaptation iterations,
        * the minimum element size,
        * the maximum element size,
        * the type of the refinement method ('curved' or 'linear')
        Its default value is geometric_adaptivity_parameters defined
        in this file. An example set of parameters is as follows:

        pair1 = ( (compute_geometric_error, None),
                  (fixed_el_tol_marking, {'tolerance': 0.01,
                                          'gamma_coarsen': 0.2} ) )

        pair2 = ( (compute_data_error, {'data function':f}),
                  (equidistribution_marking, {'tolerance':    0.0001,
                                              'gamma_refine': 0.7,
                                              'gamma_coarsen':0.05} ) )

        pair3 = ( (compute_data_coarsening_error,{'data function':f}),
                  (equidistribution_marking, {'tolerance':    0.0001,
                                              'gamma_refine': 0.7,
                                              'gamma_coarsen':0.05} ) )

        params = { 'errors and marking': [ pair1, pair2 ],
                   'coarsening errors and marking': [ pair3 ],
                   'maximum iterations': 5,
                   'minimum element size': 0.0001,
                   'maximum element size': 1.0,
                   'refinement method':'curved' }

        Each pair defined above includes two pairings. One is the error
        estimation function and its parameters, the other is the marking
        function id and its parameters.
        The error estimation functions should have the curve as an argument
        and three other optional arguments: mask, a boolean NumPy array
        or an integer NumPy array of element indices, vec, a float NumPy
        array to store the error values, parameters, a dictionary of
        parameters needed for error estimation. The default for these
        three arguments can be None. An example is
           compute_geometric_error(curve, mask=None, vec=None, parameters=None)
        defined in the file.

    Returns
    -------
    adapted : bool
        A boolean value that is True if the curve has been adapted
        (refined or coarsened), False if the curve has not been adapted.
    """
    errors, new_indices = None, None  # initially no errors, no new indices

    refinement_method = parameters.get( 'refinement method', 'curved' )

    max_iter = parameters['maximum iterations']
    i = 0
    while i < max_iter:
        # Use the error estimator arrays and the corresponding marking
        # strategies to come up with the final marker array.
        current_info = (errors, new_indices)
        markers, errors = estimates_and_markers( curve, current_info,
                                                 parameters )

        if np.all( markers == 0 ):  break  # no more marked elements

        # Nodes are added or removed based on the values of the element markers.
        errors, new_indices = refine_coarsen( curve, markers, errors,
                                              refinement_method )
        i = i+1

    curve_changed = (i > 0)

    return curve_changed
