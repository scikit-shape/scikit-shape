"""Collection of segmentation functions.

Segmentation is the task of finding the objects, regions or their boundaries
in given images. This module contains several segmentation functions, each
of which use different strategies to segment images.

"""

import numpy as np
from numpy.linalg import norm
from numpy.random import rand, randn
from numpy import inf, pi, arange, sin, cos
from scipy.fftpack import fft2, ifft2
from .shape_energy import PwConstRegionEnergy, IsotropicBoundaryEnergy
from .shape_optimization import default_optimization_parameters as default_shape_optim_parameters
from .shape_optimization import optimize as shape_optimize
from .topology_optimization import optimize as topology_optim


default_shape_energy_parameters = {'mu':0.001,
                             'domain integral tol':1e-2,
                             'surface integral tol':1e-2,
                             'min beta':0.01,
                             'use full hessian': True,
                             }

def segment_boundaries(image, initial_curves, method='multiphase',
                       parameters=( default_shape_energy_parameters,
                                    default_shape_optim_parameters ) ):
    """This function segments a given image starting with given initial curves.

    This function takes an image and a given set of initial curves,
    performs shape optimization on the surfaces based on a specified
    shape energy, thus performs image segmentation and returns the final
    set of surfaces delineating the region or object boundaries.

    Parameters
    ----------
    image : NumPy array or ImageFunction object
        A NumPy array storing the image pixel values, or an image function
        that returns the values of the image at the points given by a Numpy
        coordinate array size 2 x n_pts. The image function can be an instance
        of class :class:skshape.numerics.function.ImageFunction.
    initial_curves : CurveHierarchy object
        An instance of a CurveHierarchy that specify the collection of
        initial curves to initialize the curve evolution to compute the
        segmentation boundaries.
    method : str, optional
        One of 'multiphase' (or 'piecewise-constant'), 'two-phase' (or 'Chan-Vese'),
        'GAC' (or 'Geodesic-Active-Contours').
        Default value is 'multiphase'.
    parameters : tuple of dict, optional
        A pair of parameter dictionaries:
        (energy parameters, optimization parameters)
        Examples are in
        :data:skshape.image.segmentation.shape_energy.default_energy_parameters
        and :data:image.segmentation.shape_optim.default_parameters.

    Returns
    -------
    final_curves : CurveHierarchy object
        The initial curves are evolved by performing shape optimization
        and returned as final curves.
    """

    energy_parameters, optim_parameters = parameters

    energy_parameters['image function'] = image

    if method in ['multiphase','piecewise-constant','two-phase','Chan-Vese']:
        energy = PwConstRegionEnergy( energy_parameters )

    elif method in ['GAC','Geodesic-Active-Contours']:
        energy = IsotropicBoundaryEnergy( energy_parameters )
    else:
        raise ValueError("Invalid choice for method! The following are the available methods: 'multiphase' ('piecewise-constant'), 'GAC' ('Geodesic Active Contours') ")

    final_surfaces = shape_optimize( energy, initial_surfaces,
                                     optim_parameters, False )

    return final_surfaces


def segment_by_topology(image, n_phases, mu=30, sigma=6, init_method='zero', gamma=0.5, epsilon=0):
    """Label image pixels into separate regions by topology optimization.

    This functions segments an image by topology optimization of a statistical
    label energy. The goal is to label each image pixel with a region label,
    hence identify the regions of the image. The iterative topology optimization
    procedure first initializes the label array using init_method, then computes
    topological derivative matrix and switches labels iteratively until topological
    derivative is nonegative, and meets termination criterion.

    Parameters
    ----------
    image : NumPy ndarray
        Array of image values.
    n_phases : int
        Number of regions.
    init_method : str, optional
        Choose an initilization method from 'zero', 'rand', 'chk', 'grid',
        namely, zero or random or checkerboard or grid initialization.
        Default value is 'zero'.
    mu : float, optional
        Parameter mu for weight of nonlocal region regularization. Default value is 30.
    sigma : int, optional
        Parameter sigma for distance computation (in terms of number of pixels).
        Default value is 6.
    gamma : float, optional
        Parameter gamma for stopping criterion. Default value is 0.5.
    epsilon : float, optional
        Parameter epsilon for stopping criteria. Default value is 0.

    Returns
    -------
    labels : NumPy ndarray
        Final array of region labels.
    new_image : NumPy array
        New image formed by coloring with region averages within  each region.

    """

    labels, new_image = topology_optim( image, n_phases=4, mu=30, sigma=6, init_method='zero', gamma=0.5, epsilon=0)
    return labels, new_image


def segment_phase_field(image, u0=None, data_weight=500, s=0.85, a=0.0, eps=None,
                        step_size=None, max_iter=1000, termination_tol=0.1,
                        verbose=False):
    """Segment image to foreground/background using phase field representation.

    This function uses the fractional phase field equation to segment
    the given image into foreground objects and a background region.

    Parameters
    ----------
    image : NumPy ndarray
        A 2d gray-scale image array or the name of an image file.
    u0 : NumPy array or str or tuple, optional
        An initialization array for the phase field functions,
        or an initialization pattern 'rand', 'sine', 'cosine',
        'checkerboard', 'circles' or a cell record indicating
        the pattern and the number of repetitions in x,y dirs,
        e.g. ('sine',(4,6)), ('circles',(5,3)).
        The default value is None, it results in 'sine' pattern
        with (8,8) repetitions.
    data_weight : float, optional
        The weight of the data fidelity term. Default value is 500.
    s : float, optional
        The fractional exponent of the Laplacian, a value in (0.5,1.0].
        Default value is 0.85.
    a : float, optional
        The fractional exponent in the (H^-1)^a gradient descent term,
        a value in [0.0,1.0]. Default value is 0.0.
    epsilon : float, optional
        The order parameter for the phase field transition width.
        Default value is None, so that it is automatically assigned
        based on grid spacing.
    step_size : float, optional
        The step size for the phase field evolution. Default value
        is None.
    max_iter : int, optional
        Maximum number of iterations for the phase field evolution.
        Default value os 1000.
    termination_tol : float, optional
        Stopping tolerance on the norm of the change between the
        Fourier coefficients of the phase field U:

            norm( U2_tilde - U1_tilde ) < TOL

        Default value is 0.1.
    verbose : bool, optional
        True (default) or False, to print additional information
        at each iteration. Default value is False.

    Returns
    -------
    segmentation: NumPy array
        A boolean array storing the binary mask to represent the
        segmentation.

    """

    I,nx,ny = _check_image_for_phase_field( image )

    if (eps is None) or (eps <= 0):
        eps = 4 * 2*pi / min(nx,ny)

    if step_size is None:
        step_size = 0.25 / data_weight

    Kx = np.hstack([ arange(nx/2), arange(-nx/2,0) ]) # [0:nx/2-1,-nx/2:-1]
    Ky = np.hstack([ arange(ny/2), arange(-ny/2,0) ]) # [0:ny/2-1,-ny/2:-1]
    K1,K2 = np.meshgrid(Kx,Ky)
    K1 = K1.T;  K2 = K2.T

    A = (K1**2 + K2**2)**s
    B = (K1**2 + K2**2)**a
    A = B * A

    u, u_tilde = _initialize_phase_field( u0, nx,ny )
    u0_tilde = u_tilde


    dt = step_size
    for k in range(max_iter):
        region1 = 0.5*(1+u)
        region2 = 0.5*(1-u)
        area1 = region1.sum()
        area2 = region2.sum()
        c1 = ( region1 * I ).sum() / area1
        c2 = ( region2 * I ).sum() / area2

        I_force = eps**(-1)*data_weight * (region1*(I - c1)**2 - region2*(I - c2)**2)
        I_tilde = (2*pi/nx)*(2*pi/ny) * fft2( I_force )

        w = -4*u / (1 + u**2)**2
        w_tilde = (2*pi/nx)*(2*pi/ny) * fft2(w)

        u_tilde = (u_tilde  -  dt*eps**(-2)* B * w_tilde  -  dt * B * I_tilde) \
                    / (1 + dt*A + dt*eps**(-2) * B)

        u = ((2*pi/nx)*(2*pi/ny))**(-1) * np.real( ifft2( u_tilde ) )

        change_in_u = norm( u_tilde - u0_tilde, 'fro' )
        u0_tilde = u_tilde

        grad_norm = norm( A * u_tilde + eps**(-2) * (u_tilde + w_tilde) + I_tilde, 'fro' )
        if verbose:
            print('  Iter # %d / %d:  |u1-u0| = %4.2e,  |G| = %4.2e)' %
                  (k+1, max_iter, change_in_u, grad_norm))
        if change_in_u < termination_tol:  break

    region1 = 0.5*(1+u)

    ## _show_and_save_array( region1, k, max_iter, True,
    ##                       output_file, show_figures, save_frequency )
    return (u > 0)


def _check_image_for_phase_field( image ):

    if image.ndim != 2:
        raise ValueError('The image has to be 2d grayscale, given as a 2d NumPy array.')
    else:
        I = image

    max_I = max( I.max(), -I.min() )
    if max_I > 0:  I = np.double(I) / max_I

    nx,ny = I.shape

    # For now, we work with even dimensions only.

    if (np.mod(nx,2) == 1) or (np.mod(nx,2) == 1):
        printf("Warning: Current version of this function works with images of \
        even number of pixels, so the last row/column will be removed.")

    if np.mod(nx,2) == 1:  I = I[:nx-1,:]
    if np.mod(ny,2) == 1:  I = I[:,:ny-1]

    nx,ny = I.shape

    return I, nx, ny


def _initialize_phase_field(u0, nx, ny):

    if type(u0) is np.ndarray:

        if np.mod(u0.shape[0],2) == 1:  u0 = u0[:-1,:]
        if np.mod(u0.shape[1],2) == 1:  u0 = u0[:,:-1]

        if (u0.shape[0] != nx) or (u0.shape[1] != ny):
            raise ValueError('The initial phase field array u0 should have ' +
                             'the same size as the image array.')
        u0 = np.double(u0)
        u0_min = u0.min()
        u0_max = u0.max()
        u = -1.0 + 2.0 * (u0 - u0_min) / (u0_max - u0_min)

    elif u0 == 'rand':
        u = 2.0 * (rand(nx,ny) - 0.5)

    elif (u0 is not None) and (type(u0) not in [str, tuple, np.ndarray]):
        raise ValueError('The initial phase U0 can be specified as a numerical array ' +
                         'with the same size as the image array, or as a pattern function string:' +
                         '("rand","sine","cosine","circles"), ' +
                         'or a pair including the pattern function name and the number ' +
                         '(k1,k2) of alternations or subregions in (x,y) directions ' +
                         'e.g. ("sine",(4,6)).')
    else:
        if u0 is not None:
            pattern = 'sine'
            k1 = 8; k2 = 8
        elif is_string_like(u0):
            pattern = u0
            k1 = 8; k2 = 8
        else:
            pattern = u0[0]
            k1,k2 = u0[1]

        x1 = (2*pi/nx) * np.arange(nx)
        x2 = (2*pi/ny) * np.arange(ny)
        X1,X2 = np.meshgrid( x1, x2 )
        X1 = X1.T;  X2 = X2.T

        if pattern == 'sine':
            u = np.sin(k1*X1) * np.sin(k2*X2)

        elif pattern == 'cosine':
            u = np.cos(k1*X1) * np.cos(k2*X2)

        elif pattern == 'circles':
            h1 = 2*pi / k1
            h2 = 2*pi / k2
            r = 0.25 * sqrt( h1^2 + h2^2 )
            u = -inf * np.ones((nx,ny))
            for center1 in np.arange( h1/2, 2*pi, h1 ):
                for center2 in np.arange( h2/2, 2*pi, h2 ):
                    circle = r**4 - ( (X1 - center1)**2 + (X2 - center2)**2 )**2
                    u = np.maximum( u, circle )
            max_u = u.max()
            u[ u < -max_u ] = -max_u
            u /= max_u

        else:
            raise ValueError('The pattern name specified by u0 or u0[0] should be one of ' +
                             '"rand", "sine", "cosine", "circles".')

    u[ u < -1.0 ] = -1.0
    u[ u >  1.0 ] =  1.0

    u_tilde = (2*pi/nx)*(2*pi/ny) * fft2(u)

    return (u, u_tilde)
