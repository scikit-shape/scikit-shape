"""Labeling pixels into separate groups/regions by topology optimization.

This module includes functions to compute topological derivatives of
region-regularized statistical label energies, and to minimize these
energies iteratively to assign optimal region labels to image pixels.
Using this approach, an image can be segmented, and a label image
showing the distinct regions in the image can be obtained.

"""

from __future__ import division
import numpy as np
from numba import jit
from numpy import absolute


@jit(nopython = True)
def _weights(nx, ny, sigma):
    """Weight matrix for distance computation.

    Computes weight matrix for distance computation in the region
    regularization functions.

    Parameters
    ----------
    nx : int
        Number of columns of the image.
    ny : int
        Number of rows of the image.
    sigma : float
        Parameter that defines the threshold for distance computation.

    Returns
    -------
    W : Numpy ndarray
        Weight matrix used to compute distance.

    """

    W = np.zeros((nx, ny))
    for i in range(nx):
        for j in range(ny):
            W[i,j] = np.exp( (-i**2 - j**2) / sigma**2 ) / (2*np.pi*sigma**2)
    return W


@jit(nopython = True)
def _region_regularization_pp(n_phases, label, threshold, W):
    """Distance matrix between pixels in the same region.

    Computes distance matrix between every pixel and all pixels in the
    same region within the threshold.

    Parameters
    ----------
    n_phases : int
        Number of regions.
    label : NumPy ndarray
        Array of region labels.
    threshold : int
        Threshold to include pixels in the distance computation.
    W : NumPy ndarray
        Weight matrix to compute distance.

    Returns
    -------
    D_pp : NumPy ndarray
        Distance matrix between every pixel and all pixels in different
        regions within the threshold.

    """

    nx,ny = label.shape
    D_pp = np.zeros((nx, ny))
    for i in range(nx):
        for j in range(ny):
            p = label[i,j]
            for k in range(max(0, i-threshold), min(nx, i+threshold)):
                for l in range(max(0, j-threshold), min(ny, j + threshold)):
                    if label[k,l] == p:
                        D_pp[i,j] += W[ abs(i-k), abs(j-l) ]
    return D_pp


@jit(nopython = True)
def _region_regularization_pq(n_phases, label, threshold, W):
    """Distance matrix between pixels in the different regions.

    Computes distance matrix between every pixel and all pixels in
    different region within the threshold.

    Parameters
    ----------
    n_phases : int
        Number of regions.
    label : NumPy ndarray
        Array of region labels.
    threshold : int
        Threshold to include pixels in the distance computation.
    W : NumPy ndarray
        Weight matrix to compute distance.

    Returns
    -------
    D_pq : NumPy ndarray
        Distance matrix between every pixel and all pixels in different
        regions within the threshold.

    """

    nx,ny = label.shape
    D_pq = np.zeros((nx, ny, n_phases))
    for i in range(nx):
        for j in range(ny):
            p = label[i,j]
            for q in range(n_phases):
                if q != p:
                    for k in range(max(0, i-threshold), min(nx, i+threshold)):
                        for l in range(max(0, j-threshold), min(ny, j + threshold)):
                            if label[k,l] == q:
                                D_pq[i,j,q] += W[abs(i-k), abs(j-l)]
    return D_pq


def _update_region_statistics(image, label, region_no, model, intensity, std, count):
    k = region_no
    region_k = (label == k)
    count[k] = np.sum( region_k )
    intensity[k] = std[k] = 0

    if count[k] > 0:

        if model == 'pwconst':
            intensity[k] = np.mean( image[region_k], axis=0 )
            std[k] = 1.0

        elif model == 'Gaussian':
            intensity[k] = np.mean( image[region_k], axis=0 )
            std[k] = np.std( image[region_k], axis=0 )

        elif model == 'Laplace':
            intensity[k] = np.median( image[region_k], axis=0 )
            std[k] = np.mean( absolute( image[region_k] - intensity[k] ), axis=0 )

        else:
            raise ValueError("Model should be one of 'pwconst', 'Gaussian', 'Laplace'!")


def _auto_initialize(image, n_phases, model, threshold, W, init_method):
    """Different auto-initialization when strategies.

    Label all pixels 0, and switch pixels in regions 0 to p to region (p+1)
    if the data topological derivative is negative

    Parameters
    ----------
    image : NumPy ndarray
        Array of image values.
    n_phases : int
        Number of regions.
    model : string
        One of 'pwconst', 'Laplace', 'Gaussian'.
    threshold : int
        Threshold to include pixels in the distance computation.
    W : NumPy ndarray
        Weight matrix to compute distance
    init_method : string
        One of 'zero', 'rand', 'chk', 'grid'.

    Returns
    -------
    T : NumPy ndarray
        Topological derivative
    avg_I : NumPy ndarray
        Average intensity for each region
    labels : NumPy ndarray
        Array of region labels

    """

    nx, ny, n_channels = image.shape

    if init_method == 'zero':
        label = np.zeros((nx,ny),dtype=int)
        init_I = np.zeros(label.shape,dtype=int)

    elif init_method == 'rand':
        label = np.random.random_integers(0,n_phases-1,(nx,ny))
        init_I = np.zeros(label.shape,dtype=int)

    elif init_method == 'chk':
        label = np.zeros((nx,ny),dtype=int)
        t1 = np.linspace(0.,2*np.pi,nx)
        t2 = np.linspace(0.,2*np.pi,ny)
        x,y = np.meshgrid(t2,t1)
        S = np.sin(10*x)*np.sin(20*y)
        init_I = np.zeros(label.shape,dtype=int)
        for p in range(1, n_phases):
            init_I[S > (-1 + 2/n_phases*p)] = p
        label[:,:] = init_I

    elif init_method == 'grid':
        label = np.zeros((nx, ny), dtype = int)
        block = 5
        for i in range(nx//block):
            for j in range(ny//block):
                label[i*block:(i+1)*block,j*block:(j+1)*block] = (i+j) % n_phases
        init_I = np.zeros(label.shape,dtype=int)
    else:
        raise ValueError("Initialization method should be one of 'zero', 'rand', 'chk', 'grid'")


    init_I[:,:] = label
    intensity = np.zeros((n_phases,n_channels))
    std = np.zeros((n_phases,n_channels))
    count = np.zeros(n_phases)
    T = np.ones((n_phases,n_phases,nx, ny))*0
    T_min = np.zeros((n_phases, n_phases))

    for p in range((n_phases-1)):
        for q in range(p+1):

            _update_region_statistics( image, label, q, model, intensity, std, count )

            for i in range(nx):
                for j in range(ny):
                        if label[i,j] == q:
                            token1 = 0
                            token2 = 0
                            for c in range(n_channels):
                                if std[q,c] == 0:
                                    token1 = 0
                                else:
                                    token1 += -np.log(std[q,c])
                                    if model == 'Laplace':
                                        token1 += -absolute(image[i,j,c] - intensity[q,c]) / std[q,c]
                                    elif model in ['pwconst','Gaussian']:
                                        token1 += -0.5*((image[i,j,c] - intensity[q,c]) / std[q,c])**2
                                if std[p+1,c] == 0:
                                    token2 = 0
                                else:
                                    token2 += np.log(std[p+1,c])
                                    if model == 'Laplace':
                                        token2 += absolute(image[i,j,c] - intensity[p+1,c]) / std[p+1,c]
                                    elif model in ['pwconst','Gaussian']:
                                        token2 += 0.5*((image[i,j,c] - intensity[p+1,c]) / std[p+1,c])**2

                            T[q,(p+1),i,j] = token1 + token2

            T_min[q, (p+1)] = np.min(T[q,(p+1),:,:])

            for i in range(nx):
                for j in range(ny):
                    if T[q,(p+1),i,j] < 0:#gamma*T_min[q,(p+1)]:
                        label[i,j] = p+1

            _update_region_statistics( image, label, p+1, model, intensity, std, count )

    for k in range(n_phases):
        _update_region_statistics( image, label, k, model, intensity, std, count )

    return T, intensity, label, std, count, init_I


@jit(nopython = True)
def _derivative_computation(n_phases, label, image, model_no, intensity, std,
                            count, mu, sigma,threshold, D_pp, D_pq):
    """Compute topological derivative for a given label matrix.

    Parameters
    ----------
    n_phases : int
        Number of regions.
    label : NumPy ndarray
        Array of region labels.
    image : NumPy ndarray
        Array of image value.
    model_no : int
        0 for 'pwconst', 1 for 'Laplace', 2 for 'Gaussian'.
    intensity : NumPy ndarray
        Intensity matrix.
    mu : int
        Parameter mu for regularization.
    sigma : int
        Parameter sigma for distance computation.
    threshold : int
        Threshold to include pixels in the distance computation.
    D_pp : ndarray
        Distance matrix between pixel and the pixels in the same region.
    D_pq : ndarray
        Distance matrix between pixel and the pixels in different regions.

    Returns
    -------
    T : NumPy ndarray
        Topological derivative.
    T_min : NumPy ndarray
        Topological derivative aggregated for i^th region.
    T_i_min : NumPy ndarray
        Minimum of topological derivative for T_i, i=0,...,n_phases.

    """

    nx, ny, n_channels = image.shape

    T = np.zeros((n_phases,n_phases,nx, ny))
    T_i = np.zeros((n_phases, nx, ny))
    T_i_min = np.zeros(n_phases)

    for p in range(n_phases):
        for q in range(n_phases):
            if p != q:
                for i in range(nx):
                    for j in range(ny):
                        if label[i,j] == p:
                            token1 = 0
                            token2 = 0
                            for c in range(n_channels):
                                if std[q,c] == 0:
                                    token1 = 0
                                else:
                                    token1 += np.log(std[q,c])
                                    if model_no == 2: # model == 'Laplace'
                                        token1 += absolute( image[i,j,c]
                                                            - intensity[q,c]) / std[q,c]
                                    elif model_no in [0,1]: # model in ['pwconst','Gaussian']
                                        token1 += 0.5*((image[i,j,c] - intensity[q,c]) / std[q,c])**2

                                if std[p,c] == 0:
                                    token2 = 0
                                else:
                                    token2 += -np.log(std[p,c])
                                    if model_no == 2:
                                        token2 += -absolute( image[i,j,c]
                                                             - intensity[p,c] ) / std[p,c]
                                    elif model_no in [0,1]:
                                        token2 += -0.5*((image[i,j,c] - intensity[p,c]) / std[p,c])**2

                            token = token1 + token2
                            token += mu * ( D_pp[i,j] - D_pq[i,j,q] )
                            T[p,q,i,j] = token
                            if token < T_i[p, i, j]:
                                T_i[p,i,j] = token
                                if token < T_i_min[p]:
                                    T_i_min[p] = token
    return T, T_i, T_i_min


@jit(nopython = True)
def _label_switch(T, T_i, T_i_min, label, n_phases, gamma, D_pp, D_pq, threshold, W):
    """Switch region labels when topological derivative is negative enough.

    Change label when topological derivative is negative enough and
    update distance matrix when label changes.

    Parameters
    ----------
    T : NumPy ndarray
        Topological derivative.
    T_min : NumPy ndarray
        Topological derivative aggregated for i^th region.
    T_i_min : NumPy ndarray
        Minimum of topological derivative for T_i, i=0,...,n_phases.
    label : NumPy ndarray
        Array of region labels.
    n_phases : int
        Number of regions.
    gamma : int
        Parameter gamma for label change.
    threshold : int
        Threshold to include pixels in the distance computation.
    D_pp : NumPy ndarray
        Distance matrix between pixel and the pixels in the same region.
    D_pq : NumPy ndarray
        Distance matrix between pixel and the pixels in different regions.
    W : NumPy ndarray
        Weight matrix for distance computation.

    Returns
    -------
    label : NumPy ndarray
        Array of region labels.
    num_label_change : int
        Number of labels changed
    D_pp : NumPy ndarray
        Contribution to regularization from the same region pairs.
    D_pq : NumPy ndarray
        Contribution to regularization from the pairs of different regions.

    """

    nx,ny = label.shape

    num_label_change = 0
    for i in range(nx):
        for j in range(ny):
            p = label[i,j]
            for q in range(n_phases):
                if p != q and T[p,q,i,j] == T_i[p,i,j] and T_i[p,i,j] < gamma*T_i_min[p]:
                    label[i,j] = q
                    D_pp[i,j] = D_pq[i,j,q]
                    D_pq[i,j,q] = 0
                    for k in range(max(0, i-threshold), min(nx, i+threshold)):
                        for l in range(max(0, j-threshold), min(ny, j + threshold)):
                            if label[k,l] == p:
                                D_pp[k,l] -= W[abs(i-k), abs(j-l)]
                                D_pq[k,l,q] += W[abs(i-k), abs(j-l)]
                            elif label[k,l] == q:
                                D_pp[k,l] += W[abs(i-k), abs(j-l)]
                                D_pq[k,l,p] -= W[abs(i-k), abs(j-l)]
                            else:
                                D_pq[k,l,p] -= W[abs(i-k), abs(j-l)]
                                D_pq[k,l,q] += W[abs(i-k), abs(j-l)]
                    num_label_change += 1
                    break

    return label, num_label_change, D_pp, D_pq


@jit(nopython = True)
def _result_imaging(intensity, label, n_channels, n_phases):
    """Compute segmented intensity matrix.

    Parameters
    ----------
    intensity : NumPy ndarray
        Average intensity for regions
    label : NumPy ndarray)
        Array of region labels.
    n_channels : int
        Number of image channels.
    n_phases : int
        Number of regions.

    Returns
    -------
    avg_intensity : NumPy ndarray
        Averaged intensity map.

    """

    nx, ny = label.shape

    result = np.zeros( (nx, ny, n_channels) )

    for m in range(nx):
        for n in range(ny):
            for j in range(n_phases):
                if label[m,n] == j:
                    result[m,n] = intensity[j]
    return result


def optimize(image, n_phases, model, mu, sigma, init_method, gamma, epsilon):
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
    model : string
        One of 'pwconst', 'Laplace', 'Gaussian'.
    mu : int
        Parameter mu for regularization.
    sigma : int
        Parameter sigma for distance computation.
    init_method : str
        Choose an initilization method from 'zero', 'rand', 'chk', 'grid'.
    gamma : int
    epsilon : int
        Parameter epsilon for stopping criteria.

    Returns
    -------
    label : NumPy ndarray
        Final array of region labels.
    new_image : NumPy array
        New image formed by coloring with region averages within
        each region.

    """

    threshold = sigma*3 # int(np.floor(3*sigma*nx))

    if type(image) is not np.ndarray:
        try:
            image = image.pixels
        except:
            raise ValueError("Image argument should either be a NumPy array or ImageFunction defined with a NumPy array!")

    if image.ndim == 2:
        nx, ny = image.shape
        n_channels = 1
        image = image.reshape( (nx, ny, n_channels) )
    else:
        nx, ny, n_channels = image.shape

    W = _weights(nx, ny, sigma)

    # Step 1: initialization

    T, intensity, label, std, count, init_I = \
           _auto_initialize( image, n_phases, model, threshold, W, init_method )

    D_pp = _region_regularization_pp( n_phases, label, threshold, W )
    D_pq = _region_regularization_pq( n_phases, label, threshold, W )

    # Step 2: update iterations

    num_ite = 0
    num_label_change = []
    model_no = {'pwconst':0, 'Gaussian':1, 'Laplace':2}

    while np.any( T < epsilon ): # stopping criteria
        num_ite = num_ite + 1

        T, T_i, T_i_min = \
           _derivative_computation( n_phases, label, image, model_no[model], intensity, std,
                                    count, mu, sigma, threshold, D_pp, D_pq )

        label, num, D_pp, D_pq = _label_switch( T, T_i, T_i_min, label, n_phases,
                                                gamma, D_pp, D_pq, threshold, W )
        num_label_change.append(num)

        for i in range(n_phases):
            _update_region_statistics( image, label, i, model, intensity, std, count )

    new_image = _result_imaging( intensity, label, n_channels, n_phases )
    new_image = new_image.squeeze()

    return label, new_image
