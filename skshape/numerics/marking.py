"""Marking strategies for adaptation of elements based on errors.

This module includes function implementing different marking strategies
to decide which elements of a mesh to refine or to coarsen based on the
element errors computed.

**Summary of marking strategies:**

+-------------------+-------------------------+-------------------------+
|                   |  refinement criterion   |  coarsening criterion   |
+===================+=========================+=========================+
|  maximum strategy | error > gamma_r * tol   |  error < gamma_c * tol  |
+-------------------+-------------------------+-------------------------+
|  fixed ratio      |given ratio of sorted err|     no coarsening       |
+-------------------+-------------------------+-------------------------+
|  fixed el tol     |       error > tol       |  error < gamma_c * tol  |
+-------------------+-------------------------+-------------------------+
|  equidistribution | error > gamma_r * tol/n | error < gamma_c * tol/n |
|  (n = len(error)) |                         |                         |
+-------------------+-------------------------+-------------------------+


"""

import numpy as np


def maximum_strategy_marking(error, parameters, mask=None):
    """Marking the indices of the error vector using the maximum strategy.

    Marks the indices of the error vector using the maximum strategy:

    * the indices of errors larger than  tol*gamma_r  are returned for refinement.
    * the indices of errors smaller than  tol*gamma_c  are returned for coarsening.

    If gamma_c is not defined, then the array of coarsening indices is empty.

    Parameters
    ----------
    error : NumPy array
        A float array of error values.
    parameters : dict
        A parameter dictionary of the form
        ''{ 'tolerance':tol, 'gamma_refine':gamma_r, 'gamma_coarsen':gamma_c }''
        tol is the error tolerance, gamma_r is a ratio for refinement,
        gamma_c is a ratio for coarsening.
    mask : NumPy array
        A boolean Numpy array of the same size as error or an array of
        integer indices indicating which errors should be checked.
        Its default value is None, in which case all the error values
        are checked.

    Returns
    -------
    refine_indices : NumPy array
        An array of integers denoting the error positions marked for refinement.
    coarsen_indices : NumPy array
        An array of integers denoting the error positions marked for coarsening.

    """
    if (mask is not None) and (mask.dtype == bool):  mask = mask.nonzero()[0]

    tol = parameters['tolerance']

    gamma_r = tol * parameters['gamma_refine']

    if mask is None:
        refine_indices = np.nonzero( error > gamma_r )[0]
    else: # mask is given, check those errors only
        refine_indices = mask[ error[mask] > gamma_r ]

    gamma_c = tol * parameters.get('gamma_coarsen',-1.0)

    if gamma_c <= 0.0:
        coarsen_indices = np.array( [], dtype=int )
    elif mask is None:
        coarsen_indices = np.nonzero( error < gamma_c )[0]
    else: # mask is given, check those errors only
        coarsen_indices = mask[ error[mask] < gamma_c ]

    return (refine_indices, coarsen_indices)


def fixed_ratio_marking(error, parameters, mask=None):
    """Marking a fixed ratio of the errors in descending order.

    Marks a fixed ratio of error positions when the errors are sorted in
    decreasing order, for example, the highest 0.2 (20%) of all errors.

    Parameters
    ----------
    error : NumPy array
        A float array of error values.
    parameters : dict
        A dictionary of the form ''{ 'tolerance':tol, 'ratio':ratio }''.
        Errors < ratio*tol are marked for refinement.
    mask : NumPy array
        A boolean array of the same size as error or an array of integer
        indices indicating which errors should be checked. Its default
        value is None, in which case all the error values are checked.

    Returns
    -------
    refine_indices : NumPy array
        An array of integers denoting the error positions marked for refinement.
    coarsen_indices : NumPy array
        An empty array of integers (none marked for coarsening).

    """
    if (mask is not None) and (mask.dtype == bool):  mask = mask.nonzero()[0]

    tol = parameters['tolerance']
    ratio = parameters['ratio']
    if ratio <= 0.0 or 1.0 < ratio:
        raise ValueError("parameters['ratio'] must be a real number in (0.0,1.0].")

    if mask is None:
        sorted_indices = np.argsort( error )
        last = int( ratio * len(sorted_indices) )
        refine_indices = sorted_indices[-last:]
    else: # mask is given, check those errors only
        sorted_indices = np.argsort( error[mask] )
        last = int( ratio * len(sorted_indices) )
        refine_indices = mask[ sorted_indices[-last:] ]

    coarsen_indices = np.array([],dtype=int)

    return (refine_indices, coarsen_indices)


def fixed_el_tol_marking(error, parameters, mask=None):
    """Marking errors above a fixed tol for marking and some for coarsening.

    Marks positions with error above a fixed tol for refinement and those
    with error below tol*gamma_c for coarsening.
    If gamma_c is not defined, then the array of coarsening indices is empty.

    Parameters
    ----------
    error : NumPy array
        A float array of error values.
    parameters : dict
        A parameter dictionary of the form
        ''{ 'tolerance':tol, 'gamma_coarsen':gamma_c }''
    mask : NumPy array
        A boolean array of the same size as error or an array of integer
        indices indicating which errors should be checked. Its default
        value is None, in which case all the error values are checked.

    Returns
    -------
    refine_indices : NumPy array
        An array of integers denoting the error positions marked for refinement.
    coarsen_indices : Numpy array
        An array of integers denoting the error positions marked for coarsening.

    """
    if (mask is not None) and (mask.dtype == bool):  mask = mask.nonzero()[0]

    tol = parameters['tolerance']

    if mask is None:
        refine_indices = np.nonzero( error > tol )[0]
    else: # mask is given, check those errors only
        refine_indices = mask[ error[mask] > tol ]

    gamma_c = tol * parameters.get('gamma_coarsen',-1.0)

    if gamma_c <= 0.0:
        coarsen_indices = np.array([],dtype=int)
    elif mask is None:
        coarsen_indices = np.nonzero( error < gamma_c )[0]
    else: # mask is given, check those errors only
        coarsen_indices = mask[ error[mask] < gamma_c ]

    return (refine_indices, coarsen_indices)


def equidistribution_marking(error, parameters, mask=None):
    """Marking errors above gamma_r*tol/n for refinement (some for coarsening).

    Marks positions with error above  gamma_r*tol/n  for refinement and those
    with error below  gamma_c*tol/n  for coarsening, where n is len(error).
    If gamma_c is not defined, then the array of coarsening indices is empty.

    Parameters
    ----------
    error : NumPy array
        A float array of error values.
    parameters : dict
        A parameter dictionary of the form
        ''{ 'tolerance':tol, 'gamma_refine':gamma_r, 'gamma_coarsen':gamma_c }''
    mask : NumPy array
        A boolean array of the same size as error or an array of integer
        indices indicating which errors should be checked. Its default
        value is None, in which case all the error values are checked.

    Returns
    -------
    refine_indices : Numpy array
        An array of integers denoting the error positions marked for refinement.
    coarsen_indices : Numpy array
        An array of integers denoting the error positions marked for coarsening.

    """
    if (mask is not None) and (mask.dtype == bool):  mask = mask.nonzero()[0]

    tol = parameters['tolerance']
    n_elements = len(error)
    tol = tol / n_elements

    gamma_r = tol * parameters['gamma_refine']

    if mask is None:
        refine_indices = np.nonzero( error > gamma_r )[0]
    else: # mask is given, check those errors only
        refine_indices = mask[ error[mask] > gamma_r ]

    gamma_c = tol * parameters.get('gamma_coarsen',-1.0)

    if gamma_c <= 0.0:
        coarsen_indices = np.array( [], dtype=int )
    elif mask is None:
        coarsen_indices = np.nonzero( error < gamma_c )[0]
    else: # mask is given, check those errors only
        coarsen_indices = mask[ error[mask] < gamma_c ]

    return (refine_indices, coarsen_indices)
