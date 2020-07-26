"""Image enhancement module for image preprocessing and enhancement.

The enhancement module contains image preprocessing funtionality to improve
the quality of the image, for example, to smooth an image selectively
by preserving features like edges or corners.

"""

import numpy as np
import ..numerics.fem.grid_fem as fem
from ..geometry.grid import Grid2d
from ..numerics.function import EdgeIndicatorFunction
from scipy.ndimage.filters import gaussian_gradient_magnitude


def weighted_smoothing(image, diffusion_weight=1e-4, data_weight=1.0,
                       weight_function_parameters={}):
    """Weighted smoothing of images: smooth regions, preserve sharp edges.

    Parameters
    ----------
    image : NumPy array
    diffusion_weight : float or NumPy array, optional
        The weight of the diffusion for smoothing. It can be provided
        as an array the same shape as the image, or as a scalar that
        will multiply the default weight matrix obtained from the edge
        indicator function.
    data_weight : float or NumPy array, optional
        The weight of the image data to preserve fidelity to the image.
        It can be provided as an array the same shape as the image, or
        as a scalar that will multiply the default weight matrix obtained
        by subtracting the edge indicator function from 1.0.
    weight_function_parameters : dict, optional
        The parameters sigma and rho for the edge indicator function.
        Sigma is the standard deviation for the gaussian gradient
        applied to the image: dI = N * gaussian_gradient_magnitude(image),
        where N = min(image.shape) - 1, and rho is the scaling weight in
        the definition of the edge indicator function: 1/(1 + (dI/rho)**2)
        The default values are: {'sigma': 3.0, 'rho': None}.
        If rho is None, it is calculated by rho = 0.23 * dI.max().

    Returns
    -------
    smoothed_image : NumPy array

    Raises
    ------
    ValueError
        If diffusion_weight or data_weight have the wrong type.

    """

    if type( diffusion_weight ) is np.ndarray:
        beta = diffusion_weight

    elif not np.isscalar( diffusion_weight ):
        raise ValueError("data_weight can only be None or a scalar number "\
                         "or a NumPy array the same shape as the image.")
    else:
        sigma = weight_function_parameters.get('sigma', 3.0)
        rho = weight_function_parameters.get('rho', None)
        if rho is None:
            N = min(image.shape) - 1.0
            dI = N * gaussian_gradient_magnitude( image, sigma, mode='nearest' )
            rho = 0.23 * dI.max()
            g = EdgeIndicatorFunction( image, rho, sigma )
            G = g._g
        beta = (G - G.min()) / (G.max() - G.min())
        beta *= diffusion_weight

    if type(data_weight) is np.ndarray:
        alpha = data_weight

    elif np.isscalar(data_weight):
        alpha = data_weight * (beta.max() - beta) / (beta.max() - beta.min())
    else:
        raise ValueError("data_weight can only be None or a scalar number "\
                         "or a NumPy array the same shape as the image.")

    rhs = alpha * image

    grid = Grid2d( image.shape )

    smooth_image = fem.solve_elliptic_pde( grid, alpha, beta, rhs )

    return smooth_image
