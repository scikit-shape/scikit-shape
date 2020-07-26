"""Functionality for statistical computations needed by other functions.

This module contains functions for statistical computations. These are
mostly auxiliary functions used by other image and shape analysis functions.

"""

import numpy as np
from ..numerics.function import ImageFunction, MeshFunction
from ..numerics.fem.domain2d_fem import ReferenceElement as DomainRefElement
from ..numerics.fem.curve_fem import ReferenceElement as SurfaceRefElement


def two_clusters(I):
    """Computes two clusters from given vector with k-means algorithm.

    Given a one-dimensional array of real numbers, this function computes
    and returns two clusters using the k-means algorithm. For the method
    to work well, the data should have a clear bimodal distribution.

    Parameters
    ----------
    I : NumPy array
        A one-dimensional Numpy array of real numbers, which is assumed
        to have two clusters.

    Returns
    -------
    clusters : tuple
        A pair of tuples characterizing the two clusters computed.
        The tuples have the following form:
               (center, deviation, share, mask),
        where the center is the center (or average) of the cluster
        computed, deviation is the median of the distances of the
        points in that cluster to the center, share is the ratio
        of the points in that cluster to the total number of points,
        mask is a boolean array of the same size as I and is True
        for points belonging to the clusters, is False otherwise.
    d : NumPy array
        Array of distances for each point in I to its cluster center.
    """

    tol = 1.0 / 255.0
    c0, c1 = np.min(I), np.max(I)
    old_c0 = c0 + 2.0*tol
    old_c1 = c1 + 2.0*tol

    k = 1
    while abs(old_c0 - c0) + abs(old_c1 - c1) > tol:
        d0 = np.abs( I - c0 )
        d1 = np.abs( I - c1 )
        mask0 = d0 < d1
        mask1 = ~mask0
        n0 = np.count_nonzero( mask0 )
        n1 = len(I) - n0
        old_c0, old_c1 = c0, c1
        c0 = np.sum( I[mask0] ) / n0
        c1 = np.sum( I[mask1] ) / n1
        k = k+1

    d = np.empty( len(I) )
    d[mask0] = np.abs( I[mask0] - c0 )
    d[mask1] = np.abs( I[mask1] - c1 )

    deviation0 = np.median( d[mask0] )
    deviation1 = np.median( d[mask1] )

    share0 = 1.0 * n0 / len(I)
    share1 = 1.0 * n1 / len(I)

    clusters = [ ( c0, deviation0, share0, mask0 ),
                 ( c1, deviation1, share1, mask1 ) ]

    return (clusters, d)


def estimate_integration_errors(I, resolution=None):
    """Estimates the integration errors for elements of about pixel size.

    The given image is sampled by domain elements (e.g. triangles),
    and surface elements (e.g. curve edges), with each element being
    roughly the pixel size. Quadrature error is evaluated on each
    element and used to estimate the average error for a unit square
    or a curve of length one on the image. The purpose is to see what
    the average error would be if the unit square or the curve is meshed
    fine enough to resolve the image pixels with small elements.

    Parameters
    ----------
        I : ImageFunction
            An image function that returns the image values for a given
            2xN or 3xN array of point coordinates. It can also have the
            method, resolution(), which returns a tuple consisting of the
            number of pixels along each axes. This is used if resolution
            is not given as a separate argument.
        resolution : tuple, optional
            A pair or triple of integers defining the grid size to infer
            the minimum size of mesh elements to be used in integration.

    Returns
    -------
        domain_int_error : float
            The average integration error on a unit area of the image
            when the integration is performed on triangles, each roughly
            the size of a pixel.
        surface_int_error : float
            The average integration error on a unit length curve on the
            image when the curve consists of elements of roughly the size
            of a pixel.
        sample_values : NumPy array
            A one-dimensional NumPy array containing all of the random image
            values used for the computations.
    """
    try:
        if resolution is not None:
            grid_size = resolution
        else:
            grid_size = I.resolution()
    except AttributeError:
        print("Need the resolution parameter to continue!")
        return None

    ref_element = DomainRefElement()
    quad0 = ref_element.quadrature( order=2 )
    quad1 = ref_element.quadrature( order = quad0.order()+1 )

    domain = ref_element.random_elements( grid_size, I.diameter() )

    func = MeshFunction( I, caching=False )

    pts0, pts1 = quad0.points(), quad1.points()
    f0_list = [ func( domain, pts0[:,k] ) for k in range(pts0.shape[1]) ]
    f1_list = [ func( domain, pts1[:,k] ) for k in range(pts1.shape[1]) ]

    error  = sum(( weight*f0 for f0,weight in zip(f0_list, quad0.weights()) ))
    error -= sum(( weight*f1 for f1,weight in zip(f1_list, quad1.weights()) ))
    error[:] = np.abs( error[:] )

    el_sizes = domain.element_sizes()
    error *= el_sizes
    avg_error = np.mean( error )
    avg_el_size = np.mean( el_sizes )

    normalization = 1.0 / avg_el_size
    domain_int_error = normalization * avg_error

    # Compute the integration error expected for curve elements
    # at pixel level.

    ref_element = SurfaceRefElement()
    quad0 = ref_element.quadrature( order=2 )
    quad1 = ref_element.quadrature( order = quad0.order()+1 )

    surface = ref_element.random_elements( grid_size, I.diameter() )

    func = MeshFunction( I, caching=False )

    f3_list = [ func( surface, pt ) for pt in quad0.points() ]
    f4_list = [ func( surface, pt ) for pt in quad1.points() ]

    error  = sum(( weight*f3 for f3,weight in zip(f3_list, quad0.weights()) ))
    error -= sum(( weight*f4 for f4,weight in zip(f4_list, quad1.weights()) ))
    error[:] = np.abs( error[:] )

    el_sizes = surface.element_sizes()
    error *= el_sizes
    avg_error = np.mean( error )
    avg_el_size = np.mean( el_sizes )

    normalization = 1.0 / avg_el_size
    surface_int_error = normalization * avg_error

    # Put sample image values in a single vector and return.
    sample_values = np.hstack( f0_list + f1_list + f3_list + f4_list )

    return (domain_int_error, surface_int_error, sample_values)
