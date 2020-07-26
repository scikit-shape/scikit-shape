"""Numerical integration functions incl. fixed & adaptive quadrature.

This module includes numerical integration procedures to approximate
the integral of a given function on the given mesh. The functions
can be integrated using fixed Gaussian quadrature on the elements
of the mesh, or using adaptive integration either by refining mesh
elements or increasing order of quadrature based on error criteria.

"""

import numpy as np
from numpy import array
from itertools import product
from .function import MeshFunction, MeshFunction2
from .marking import fixed_ratio_marking as marked_elements

MAX_ADAPT_ITERATIONS = 20

# TODO: Modify all integration functions, so that they work with
# vector-valued and matrix-valued functions too.


def integrate(f, mesh, quadrature_degree=None):
    """Integrate a function on a mesh using fixed Gaussian quadrature.

    This procedure takes a function and a mesh, and estimates the
    integral of f on mesh numerically using Gaussian quadrature.
    The degree of the quadrature can be specified. If it is not given,
    then the degree of the function f is used if available, otherwise
    quadrature degree is 2.

    Parameters
    ----------
    f : MeshFunction or function_like
        An instance of MeshFunction or a function of X where X is
        a dxn array of n coordinate points in d-space).
    mesh : Mesh
        An instance of a mesh (possibly Curve, CurveHierarchy, Domain2d),
        that gives the domain of integration.
    quadrature_degree : int, optional
        The degree of Gaussian quadrature to be used for integration.

    Returns
    -------
    integral : float
        The computed integral of f on given mesh.
    """

    if quadrature_degree is not None:
        quad_degree = quadrature_degree
    else:
        try:
            quad_degree = f.degree()
        except AttributeError:
            quad_degree = 2  # assume quadratic polynomials by default

    quad = mesh.ref_element.quadrature( quad_degree )

    try:
        integral = sum(( weight * f(mesh,pt) for pt,weight in quad.iterpoints() ))
    except TypeError:
        f = MeshFunction(f)
        integral = sum(( weight * f(mesh,pt) for pt,weight in quad.iterpoints() ))

    integral *= mesh.element_sizes()

    return np.sum(integral)


def double_integrate(f, mesh1, mesh2, quadrature_degree=None):
    """Double integral of function on two meshes with Gaussian quadrature.

    This procedure takes a function and a mesh, and estimates the double
    integral of f on mesh1 and mesh2 numerically using Gaussian quadrature.
    The degree of the quadrature can be specified. If it is not given,
    then the degree of the function f is used if available, otherwise
    quadrature degree is 2.

    Parameters
    ----------
    f : MeshFunction or function_like
        An instance of MeshFunction or a function of X where X is
        a dxn array of n coordinate points in d-space).
    mesh1 : Mesh
        An instance of a mesh (possibly Curve, CurveHierarchy, Domain2d),
        that gives the first domain of integration.
    mesh2 : Mesh
        An instance of a mesh (possibly Curve, CurveHierarchy, Domain2d),
        that gives the second domain of integration.
    quadrature_degree : int, optional
        The degree of Gaussian quadrature to be used for integration.

    Returns
    -------
    integral : float
        The computed integral of f on given mesh.
    """

    if quadrature_degree is not None:
        quad_degree = quadrature_degree
    else:
        try:
            quad_degree = f.degree()
        except AttributeError:
            quad_degree = 2  # assume quadratic polynomials by default

    quad1 = mesh1.ref_element.quadrature( quad_degree )
    quad2 = mesh2.ref_element.quadrature( quad_degree )

    quad_pts = product( quad1.iterpoints(), quad2.iterpoints() )

    try:
        integral = sum(( w1*w2 * f(mesh1,mesh2,pt1,pt2)
                         for (pt1,w1),(pt2,w2) in quad_pts ))
    except TypeError:
        f = MeshFunction2(f)
        integral = sum(( w1*w2 * f(mesh1,mesh2,pt1,pt2)
                         for (pt1,w1),(pt2,w2) in quad_pts ))

    integral *= np.outer( mesh1.element_sizes(), mesh2.element_sizes() )

    return np.sum(integral)


def adapt_integrate_tolerance(f, mesh, quad0=None, quad1=None):
    """Estimates a minimum tolerance for adaptive integration for (f, mesh).

    This procedure takes a function and a mesh, and estimates a minimum
    tolerance for adaptive integration. The function is used to specify
    a minimum length scale, which we do not want to overresolve in adaptive
    integration. For example, if the function is based on an ImageFunction,
    the ImageFunction provides its resolution and diameter, from which we can
    compute the pixel size. The variations in function value at a scale of
    the pixel size determines the minimum tolerance; we should not use many
    elements with adaptive integration to resolve variations at this scale.
    The function is sampled with small random elements and the average error
    is scaled to the mesh size to obtain the tolerance.

    Parameters
    ----------
    f : MeshFunction or function_like
        A function of X (a dxn array of n coordinate points in d-space) or
        an instance of a MeshFunction. It should have member functions:
        f.resolution(), f.diameter(), both of which are used to
        figure out the smallest scale.
    mesh : Mesh
        An instance of a mesh (possibly Curve, CurveHierarchy, Domain2d),
        whose ref_element is used to define the random elements.
    quad0 : Quadrature
        The less accurate quadrature. Default value is None, in which case
        a quadrature of order=2 is used.
    quad1 : Quadrature
        The more accurate quadrature. Default value is None, in which case
        a quadrature of order=quad0.order()+1 is used.

    Returns
    -------
    tol : float
        The estimated minimum tolerance for the given functions and mesh.
    """
    if quad0 is None:
        quad0 = mesh.ref_element.quadrature( order=2 )
    if quad1 is None:
        quad1 = mesh.ref_element.quadrature( order = quad0.order()+1 )

    grid_size = f.resolution()
    random_mesh = mesh.ref_element.random_elements( grid_size, f.diameter() )

    try:
        error  = sum(( w * f(random_mesh,pt) for pt,w in quad0.iterpoints() ))
    except TypeError:
        f = MeshFunction(f)
        error  = sum(( w * f(random_mesh,pt) for pt,w in quad0.iterpoints() ))
    error -= sum(( w * f(random_mesh,pt) for pt,w in quad1.iterpoints() ))
    error[:] = np.abs( error[:] )

    el_sizes = random_mesh.element_sizes()
    error *= el_sizes
    avg_error = np.mean( error )
    avg_el_size = np.mean( el_sizes )

    # At this point, we know the average error for a set of small random
    # elements. From this, we want to infer the right tolerance for
    # the given mesh and the function. The tolerance should be proportional
    # to the total error if the mesh were made up of such small elements,
    # so we multiply the average error by the ratio of the mesh area
    # and the average size of the random elements.

    if mesh.dim() == mesh.dim_of_world(): # domain in 2d or 3d
        area = mesh.volume()
    else: # if it is a curve or a surface
        area = mesh.surface_area()

    factor = 50.0 * area / avg_el_size
    tol = factor * avg_error
    return tol


def adapt_integrate(f, mesh, tol=None):
    """Adaptively integrate a function on a given mesh by refinement.

    This procedure takes a function and a mesh, and estimates the integral
    of f on mesh by numerical quadrature. This is done adaptively. Quadrature
    error in each element is estimated by comparing the values of low-order
    and high-order quadratures. If the total error is higher than tol,
    the elements with highest error are refined. The refinements continue
    until total error is below the prescribed tol.

    Parameters
    ----------
    f : MeshFunction or function_like
        An instance of MeshFunction or a function of X where X is
        a dxn array of n coordinate points in d-space).
    mesh : Mesh
        An instance of a mesh (possibly Curve, CurveHierarchy, Domain2d),
        that gives the domain of integration. Some elements of
        the mesh might be refined into smaller elements.
    tol : float
        A tolerance value for the total error in integration.
        The default value for tol is None. If the user does not
        specify a tolerance. This function tries to estimate a tol
        based on minimum scale computed from f.diameter() and
        f.resolution(). If this fails, tol is set to 100*eps,
        where eps is the machine epsilon for floating point numbers.

    Returns
    -------
    integral : float
        The computed integral of f on mesh.
    """
    # Error at each mesh element is given by the difference of
    # the low order quadrature and the high order quadrature.
    quad0 = mesh.ref_element.quadrature( order=2 )
    quad1 = mesh.ref_element.quadrature( order=3 )

    if tol is None:
        try:
            tol = adapt_integrate_tolerance( f, mesh, quad0, quad1 )
        except:
            tol = 100.0 * np.finfo(float).eps
    try:
        error = sum(( weight * f(mesh,pt) for pt,weight in quad0.iterpoints() ))
    except TypeError:
        f = MeshFunction(f)
        error = sum(( weight * f(mesh,pt) for pt,weight in quad0.iterpoints() ))
    integral = sum(( weight * f(mesh,pt) for pt,weight in quad1.iterpoints() ))
    error   -= integral
    error[:] = np.abs( error[:] )

    el_sizes  = mesh.element_sizes()
    integral *= el_sizes
    error    *= el_sizes
    total_integral = np.sum( integral )
    total_error    = np.sum( error )

    parameters = {'tolerance':tol, 'ratio':0.3}
    n_iters = 0
    while (total_error > tol) and (n_iters < MAX_ADAPT_ITERATIONS):
        mask, other = marked_elements( error, parameters )

        total_integral -= np.sum( integral[mask] )
        total_error -= np.sum( error[mask] )

        data_vecs = [ integral, error ] # Don't lose computed values.
        data_vecs, mask = mesh.refine_coarsen( mask, data_vecs )
        integral, error = data_vecs

        error[mask]    = sum(( weight * f(mesh, pt, mask)
                               for pt,weight in quad0.iterpoints() ))
        integral[mask] = sum(( weight * f(mesh, pt, mask)
                               for pt,weight in quad1.iterpoints() ))
        error[mask] -= integral[mask]
        error[mask]  = np.abs( error[mask] )

        el_sizes = mesh.element_sizes()
        integral[mask] *= el_sizes[mask]
        error[mask]    *= el_sizes[mask]

        total_integral += np.sum( integral[mask] )
        total_error += np.sum( error[mask] )

        n_iters += 1

    return (total_integral, total_error)


def adapt_order_integrate(f, mesh, tol=None):
    """Adaptively integrate a function by increasing quadrature order.

    This procedure takes a function and a mesh, and estimates the integral
    of f on mesh using numerical quadrature. This is done adaptively.
    Quadrature error in each element is estimated by comparing the values
    of low-order and high-order quadratures. If the total error is higher
    than tol, a higher order quadrature is applied to the elements with
    highest error in order to integrate those elements with more accuracy.
    This continues until total error is below the prescribed tol.

    Parameters
    ----------
    f : MeshFunction or function_like
        An instance of MeshFunction or a function of X where X is
        a dxn array of n coordinate points in d-space).
    mesh : Mesh
        An instance of a mesh (possibly Curve, CurveHierarchy, Domain2d),
        that gives the domain of integration. Some elements of
        the mesh might be refined into smaller elements.
    tol : float
        A tolerance value for the total error in integration.

    Returns
    -------
    integral : float
        The computed integral of f on mesh.
    """
    # Error at each mesh element is given by the difference of
    # the low order quadrature and the high order quadrature.

    get_quadrature = mesh.ref_element.quadrature # shortcut function

    quad_order = 3
    quad = get_quadrature( order=quad_order-1 )
    try:
        error = sum(( weight*f(mesh,pt) for pt,weight in quad.iterpoints() ))
    except TypeError:
        f = MeshFunction(f)
        error = sum(( weight*f(mesh,pt) for pt,weight in quad.iterpoints() ))

    quad = get_quadrature( order=quad_order )
    integral = sum(( weight*f(mesh,pt) for pt,weight in quad.iterpoints() ))
    error -= integral
    error[:] = np.abs(error[:])

    el_sizes  = mesh.element_sizes()
    integral *= el_sizes
    error    *= el_sizes
    total_integral = np.sum( integral )
    total_error    = np.sum( error )

    # Recompute the integrals of those elements using higher order quadrature.
    parameters = {'tolerance':tol, 'ratio':0.3}
    n_iters = 0
    while (total_error > tol) and (quad_order < quad.max_order()) \
          and (n_iters < MAX_ADAPT_ITERATIONS):
        quad_order += 1
        quad = get_quadrature( order=quad_order )

        mask, other = marked_elements( error, parameters )

        total_integral -= np.sum( integral[mask] )
        total_error -= np.sum( error[mask] )

        error[mask] = integral[mask]
        integral[mask] = sum(( weight*f(mesh, pt, mask)
                               for pt,weight in quad.iterpoints() ))
        integral[mask] *= el_sizes[mask]
        error[mask] -= integral[mask]
        error[mask] = np.abs( error[mask] )

        total_integral += np.sum( integral[mask] )
        total_error += np.sum( error[mask] )

        n_iters += 1

    return (total_integral, total_error)


def L2_norm(f, mesh, quadrature_degree=None):
    """L2 norm of f on mesh using fixed Gaussian quadrature.

    This procedure takes a function and a mesh, and estimates the
    L2 norm of f on mesh numerically using Gaussian quadrature.
    The degree of the quadrature can be specified. If it is not given,
    then the degree of the function f is used if available, otherwise
    quadrature degree is 2.

    Parameters
    ----------
    f : MeshFunction or function_like
        An instance of MeshFunction or a function of X where X is
        a dxn array of n coordinate points in d-space).
    mesh : Mesh
        An instance of a mesh (possibly Curve, CurveHierarchy, Domain2d),
        that gives the domain of integration.
    quadrature_degree : int, optional
        The degree of Gaussian quadrature to be used for integration.

    Returns
    -------
    L2_norm_est : float
        The estimated L2 norm of f on given mesh.
    """

    if quadrature_degree is not None:
        quad_degree = quadrature_degree
    else:
        try:
            quad_degree = 2*f.degree()
        except AttributeError:
            quad_degree = 4  # default: square of quadratic polynomials

    quad = mesh.ref_element.quadrature( quad_degree )

    try:
        integral = sum(( weight*f(mesh,pt)**2  for pt,weight in quad.iterpoints() ))
    except TypeError:
        f = MeshFunction(f)
        integral = sum(( weight*f(mesh,pt)**2  for pt,weight in quad.iterpoints() ))

    integral *= mesh.element_sizes()

    return np.sum(integral)**0.5


def max_norm(f, mesh, quadrature_degree=None):
    """Max norm of f on mesh using fixed Gaussian quadrature.

    This procedure takes a function and a mesh, and estimates the
    maximum norm of f on mesh numerically using Gaussian quadrature.
    The degree of the quadrature can be specified. If it is not given,
    then the degree of the function f is used if available, otherwise
    quadrature degree is 2.

    Parameters
    ----------
    f : MeshFunction or function_like
        An instance of MeshFunction or a function of X where X is
        a dxn array of n coordinate points in d-space).
    mesh : Mesh
        An instance of a mesh (possibly Curve, CurveHierarchy, Domain2d),
        that gives the domain of integration.
    quadrature_degree : int, optional
        The degree of Gaussian quadrature to be used for integration.

    Returns
    -------
    max_norm_est : float
        The estimated maximum norm of f on given mesh.
    """

    if quadrature_degree is not None:
        quad_degree = quadrature_degree
    else:
        try:
            quad_degree = 2*f.degree()
        except AttributeError:
            quad_degree = 4  # default: square of quadratic polynomials

    quad = mesh.ref_element.quadrature( quad_degree )

    try:
        max_val = max(( np.max(np.abs(f(mesh,pt))) for pt in quad.points() ))
    except TypeError:
        f = MeshFunction(f)
        max_val = max(( np.max(np.abs(f(mesh,pt))) for pt in quad.points() ))

    return max_val


def L2_error(f1, f2, mesh, quadrature_degree=None):
    """L2 error between f1, f2 on mesh using fixed Gaussian quadrature.

    This procedure takes a function and a mesh, and estimates the L2 error
    between f1 and f2 on mesh numerically using Gaussian quadrature.
    The degree of the quadrature can be specified. If it is not given,
    then the maximum of the degree of the functions f1 and f2 is used if
    their degrees are available, otherwise quadrature degree is 2.

    Parameters
    ----------
    f1 : MeshFunction or function_like
        An instance of MeshFunction or a function of X where X is
        a dxn array of n coordinate points in d-space).
    f2 : MeshFunction or function_like
        An instance of MeshFunction or a function of X where X is
        a dxn array of n coordinate points in d-space).
    mesh : Mesh
        An instance of a mesh (possibly Curve, CurveHierarchy, Domain2d),
        that gives the domain of integration.
    quadrature_degree : int, optional
        The degree of Gaussian quadrature to be used for integration.

    Returns
    -------
    L2_error_est : float
        The estimated L2 error between f1 and f2 on given mesh.
    """

    if quadrature_degree is not None:
        quad_degree = quadrature_degree
    else:
        try:
            quad_degree = 2*max( f1.degree(), f2.degree() )
        except AttributeError:
            quad_degree = 4  # assume quadratic polynomials by default

    quad = mesh.ref_element.quadrature( quad_degree )

    integral = np.zeros(mesh.size())
    for pt, weight in quad.iterpoints():
        try:
            F1 = f1( mesh, pt )
        except TypeError as e:
            f1 = MeshFunction(f1)
            F1 = f1( mesh, pt )
        try:
            F2 = f2( mesh, pt )
        except TypeError:
            f2 = MeshFunction(f2)
            F2 = f2( mesh, pt )

        diff = (F1 - F2)**2

        if diff.ndim == 1: # the case of scalar-valued mesh function
            integral += weight * diff
        elif diff.ndim == 2: # the case of vector-valued mesh function
            integral += weight * np.sum( diff, 0 )
        elif diff.ndim == 3: # the case of matrix-valued mesh function
            integral += weight * np.sum(np.sum( diff, 0 ), 0)

    integral *= mesh.element_sizes()

    return np.sum(integral)**0.5


def max_error(f1, f2, mesh, quadrature_degree=None):
    """Max error between f1, f2 on mesh using fixed Gaussian quadrature.

    This procedure takes a function and a mesh, and estimates the maximum
    error between f1 and f2 on mesh numerically using Gaussian quadrature.
    The degree of the quadrature can be specified. If it is not given,
    then the maximum of the degree of the functions f1 and f2 is used if
    their degrees are available, otherwise quadrature degree is 2.

    Parameters
    ----------
    f1 : MeshFunction or function_like
        An instance of MeshFunction or a function of X where X is
        a dxn array of n coordinate points in d-space).
    f2 : MeshFunction or function_like
        An instance of MeshFunction or a function of X where X is
        a dxn array of n coordinate points in d-space).
    mesh : Mesh
        An instance of a mesh (possibly Curve, CurveHierarchy, Domain2d),
        that gives the domain of integration.
    quadrature_degree : int, optional
        The degree of Gaussian quadrature to be used for integration.

    Returns
    -------
    max_error_est : float
        The estimated L2 error between f1 and f2 on given mesh.
    """

    if quadrature_degree is not None:
        quad_degree = quadrature_degree
    else:
        try:
            quad_degree = max( f1.degree(), f2.degree() )
        except AttributeError:
            quad_degree = 2  # assume quadratic polynomials by default

    quad = mesh.ref_element.quadrature( quad_degree )

    max_error = 0.0
    for pt in quad.points():
        try:
            F1 = f1( mesh, pt )
        except TypeError:
            f1 = MeshFunction(f1)
            F1 = f1( mesh, pt )
        try:
            F2 = f2( mesh, pt )
        except TypeError:
            f2 = MeshFunction(f2)
            F2 = f2( mesh, pt )

        diff = np.abs(F1 - F2)
        max_error = max( max_error, np.max( diff ) )

    return max_error


class Quadrature(object):

    def __init__(self):
        self._degree = 0
        self._order = 0
        self._points = np.array([])
        self._weights = np.array([])

    def __len__(self):
        return len(self._points)

    def order(self):
        return self._order

    def max_order(self):
        raise NotImplementedError

    def degree(self):
        return self._degree

    def max_degree(self):
        raise NotImplementedError

    def points(self):
        return self._points

    def weights(self):
        return self._weights


QUADRATURE1D_MAX_ORDER = 19
QUADRATURE1D_MAX_DEGREE = 2*QUADRATURE1D_MAX_ORDER - 1

class Quadrature1d(Quadrature):

    def __init__(self, degree=None, order=None):
        if degree is not None:
            if degree < 0:
                raise ValueError("Minimum degree is 0.")
            elif degree > QUADRATURE1D_MAX_DEGREE:
                raise ValueError("Degree cannot be larger than %d." % QUADRATURE1D_MAX_DEGREE)
            self._degree = degree
            self._order  = degree/2

        elif order is not None:
            if order < 0:
                raise ValueError("Minimum order is 0.")
            elif order > QUADRATURE1D_MAX_ORDER:
                raise ValueError("Order cannot be larger than %d." % QUADRATURE1D_MAX_ORDER)
            self._order  = order
            self._degree = 2*order + 1

        else: # both order and degree are None
            raise ValueError("Either degree or order should be specified.")

        self._points = _quadrature1d_points[self._order]
        self._weights = _quadrature1d_weights[self._order]

    def integrate(self,fct):
        return np.dot( self._weights, fct(self._points) )

    def max_order(self):
        return QUADRATURE1D_MAX_ORDER

    def max_degree(self):
        return QUADRATURE1D_MAX_DEGREE

    def iterpoints(self):
        return zip( self._points, self._weights )


QUADRATURE2D_MAX_ORDER = 19
QUADRATURE2D_MAX_DEGREE = QUADRATURE2D_MAX_ORDER

class Quadrature2d(Quadrature):

    def __init__(self, degree=None, order=None):
        if degree is not None:
            if degree < 0:
                raise ValueError("Minimum degree is 0.")
            elif degree > QUADRATURE2D_MAX_DEGREE:
                raise ValueError("Degree cannot be larger than %d." % QUADRATURE2D_MAX_DEGREE)
            self._degree = degree
            self._order  = max(0,degree-1)

        elif order is not None:
            if order < 0:
                raise ValueError("Minimum order is 0.")
            elif order > QUADRATURE2D_MAX_ORDER:
                raise ValueError("Order cannot be larger than %d." % QUADRATURE2D_MAX_ORDER)
            self._order  = order
            self._degree = order+1

        else: # both order and degree are None
            raise ValueError("Either degree or order should be specified.")

        self._points = _quadrature2d_points[self._order]
        self._weights = _quadrature2d_weights[self._order]

    def integrate(self,fct):
        return 0.5 * np.dot( self._weights, fct(self._points) )

    def max_order(self):
        return QUADRATURE2D_MAX_ORDER

    def max_degree(self):
        return QUADRATURE2D_MAX_DEGREE

    def iterpoints(self):
        weights = self._weights
        pts = self._points
        n = len(weights)
        return ( ( (pts[0,k],pts[1,k]), weights[k] ) for k in range(n) )


_quadrature1d_points = [
    array([                 0.5 ]),

    array([  0.2113248654051871,  0.7886751345948129 ]),

    array([  0.1127016653792583,                 0.5,  0.8872983346207417 ]),

    array([ 0.06943184420297371,  0.3300094782075719,  0.6699905217924281,
             0.9305681557970262 ]),

    array([ 0.04691007703066802,  0.2307653449471584,                 0.5,
             0.7692346550528415,  0.9530899229693319 ]),

    array([ 0.03376524289842397,  0.1693953067668678,  0.3806904069584016,
             0.6193095930415985,  0.8306046932331322,   0.966234757101576 ]),

    array([ 0.02544604382862076,  0.1292344072003028,  0.2970774243113014,
                            0.5,  0.7029225756886985,  0.8707655927996972,
             0.9745539561713792 ]),

    array([ 0.01985507175123191,  0.1016667612931866,  0.2372337950418355,
             0.4082826787521751,  0.5917173212478249,  0.7627662049581645,
             0.8983332387068134,  0.9801449282487681 ]),

    array([ 0.01591988024618696, 0.08198444633668212,  0.1933142836497048,
             0.3378732882980955,                 0.5,  0.6621267117019045,
             0.8066857163502952,  0.9180155536633179,   0.984080119753813 ]),

    array([ 0.01304673574141413, 0.06746831665550773,  0.1602952158504878,
             0.2833023029353764,  0.4255628305091844,  0.5744371694908156,
             0.7166976970646236,  0.8397047841495122,  0.9325316833444923,
             0.9869532642585859 ]),

    array([ 0.01088567092697151, 0.05646870011595234,  0.1349239972129753,
             0.2404519353965941,  0.3652284220238275,                 0.5,
             0.6347715779761725,  0.7595480646034058,  0.8650760027870247,
             0.9435312998840477,  0.9891143290730284 ]),

    array([ 0.009219682876640378, 0.04794137181476255,  0.1150486629028477,
             0.2063410228566913,  0.3160842505009099,  0.4373832957442655,
             0.5626167042557344,  0.6839157494990901,  0.7936589771433087,
             0.8849513370971523,  0.9520586281852375,  0.9907803171233596 ]),

    array([ 0.007908472640705932, 0.04120080038851104, 0.09921095463334506,
             0.1788253302798299,  0.2757536244817765,  0.3847708420224326,
                            0.5,  0.6152291579775674,  0.7242463755182235,
             0.8211746697201701,  0.9007890453666549,   0.958799199611489,
             0.9920915273592941 ]),

    array([ 0.006858095651593843, 0.03578255816821324, 0.08639934246511749,
             0.1563535475941573,   0.242375681820923,  0.3404438155360551,
             0.4459725256463282,  0.5540274743536718,  0.6595561844639448,
              0.757624318179077,  0.8436464524058427,  0.9136006575348825,
             0.9642174418317868,  0.9931419043484062 ]),

    array([ 0.006003740989757311, 0.03136330379964702,  0.0758967082947864,
              0.137791134319915,  0.2145139136957306,  0.3029243264612183,
             0.3994029530012827,                 0.5,  0.6005970469987173,
             0.6970756735387817,  0.7854860863042694,   0.862208865680085,
             0.9241032917052137,   0.968636696200353,  0.9939962590102427 ]),

    array([ 0.005299532504175031,  0.0277124884633837, 0.06718439880608412,
             0.1222977958224985,  0.1910618777986781,  0.2709916111713863,
             0.3591982246103705,  0.4524937450811813,  0.5475062549188188,
             0.6408017753896295,  0.7290083888286136,  0.8089381222013219,
             0.8777022041775016,  0.9328156011939159,  0.9722875115366163,
              0.994700467495825 ]),

    array([ 0.004712262342791318,  0.0246622391156161, 0.05988042313650704,
             0.1092429980515993,  0.1711644203916546,  0.2436547314567615,
             0.3243841182730618,  0.4107579092520761,                 0.5,
             0.5892420907479239,  0.6756158817269382,  0.7563452685432385,
             0.8288355796083453,  0.8907570019484007,  0.9401195768634929,
             0.9753377608843838,  0.9952877376572087 ]),

    array([ 0.004217415789534551, 0.02208802521430114, 0.05369876675122215,
            0.09814752051373843,  0.1541564784698234,  0.2201145844630262,
             0.2941244192685787,  0.3740568871542472,  0.4576124934791324,
             0.5423875065208676,  0.6259431128457528,  0.7058755807314213,
             0.7798854155369738,  0.8458435215301766,  0.9018524794862616,
             0.9463012332487779,  0.9779119747856988,  0.9957825842104655 ]),

    array([ 0.003796578078207824, 0.01989592393258499, 0.04842204819259105,
            0.08864267173142859,  0.1395169113323853,  0.1997273476691595,
             0.2677146293120195,  0.3417179500181851,  0.4198206771798873,
                            0.5,  0.5801793228201126,  0.6582820499818149,
             0.7322853706879805,  0.8002726523308406,  0.8604830886676147,
             0.9113573282685714,   0.951577951807409,   0.980104076067415,
             0.9962034219217921 ]),

    array([ 0.003435700407452558,  0.0180140363610431, 0.04388278587433703,
            0.08044151408889061,  0.1268340467699246,  0.1819731596367425,
             0.2445664990245864,  0.3131469556422902,  0.3861070744291775,
             0.4617367394332513,  0.5382632605667487,  0.6138929255708225,
             0.6868530443577098,  0.7554335009754136,  0.8180268403632576,
             0.8731659532300754,  0.9195584859111094,   0.956117214125663,
              0.981985963638957,  0.9965642995925474 ])
    ]

_quadrature1d_weights = [
    array([                 1.0 ]),

    array([                 0.5,                 0.5 ]),

    array([  0.2777777777777777,  0.4444444444444444,  0.2777777777777777 ]),

    array([  0.1739274225687269,  0.3260725774312731,  0.3260725774312731,
             0.1739274225687269 ]),

    array([  0.1184634425280946,  0.2393143352496832,  0.2844444444444444,
             0.2393143352496832,  0.1184634425280946 ]),

    array([ 0.08566224618958512,  0.1803807865240693,  0.2339569672863456,
             0.2339569672863456,  0.1803807865240693, 0.08566224618958512 ]),

    array([ 0.06474248308443491,  0.1398526957446384,  0.1909150252525595,
             0.2089795918367347,  0.1909150252525595,  0.1398526957446384,
            0.06474248308443491 ]),

    array([ 0.05061426814518819,  0.1111905172266872,  0.1568533229389437,
             0.1813418916891811,  0.1813418916891811,  0.1568533229389437,
             0.1111905172266872, 0.05061426814518819 ]),

    array([ 0.04063719418078722, 0.09032408034742873,  0.1303053482014678,
             0.1561735385200014,  0.1651196775006299,  0.1561735385200014,
             0.1303053482014678, 0.09032408034742873, 0.04063719418078722 ]),

    array([ 0.03333567215434404, 0.07472567457529027,   0.109543181257991,
             0.1346333596549982,  0.1477621123573764,  0.1477621123573764,
             0.1346333596549982,   0.109543181257991, 0.07472567457529027,
            0.03333567215434404 ]),

    array([ 0.02783428355808685, 0.06279018473245231, 0.09314510546386709,
             0.1165968822959953,  0.1314022722551233,  0.1364625433889503,
             0.1314022722551233,  0.1165968822959953, 0.09314510546386709,
            0.06279018473245231, 0.02783428355808685 ]),

    array([ 0.02358766819325592, 0.05346966299765921,  0.0800391642716731,
              0.101583713361533,  0.1167462682691774,  0.1245735229067014,
             0.1245735229067014,  0.1167462682691774,   0.101583713361533,
             0.0800391642716731, 0.05346966299765921, 0.02358766819325592 ]),

    array([ 0.02024200238265797, 0.04606074991886425, 0.06943675510989364,
            0.08907299038097287,  0.1039080237684443,  0.1131415901314486,
             0.1162757766154369,  0.1131415901314486,  0.1039080237684443,
            0.08907299038097287, 0.06943675510989364, 0.04606074991886425,
            0.02024200238265797 ]),

    array([ 0.01755973016587596, 0.04007904357988012, 0.06075928534395157,
            0.07860158357909677, 0.09276919873896891,  0.1025992318606478,
             0.1076319267315789,  0.1076319267315789,  0.1025992318606478,
            0.09276919873896891, 0.07860158357909677, 0.06075928534395157,
            0.04007904357988012, 0.01755973016587596 ]),

    array([  0.0153766209980587, 0.03518302374405403, 0.05357961023358597,
            0.06978533896307716, 0.08313460290849699, 0.09308050000778109,
            0.09921574266355579,  0.1012891209627806, 0.09921574266355579,
            0.09308050000778109, 0.08313460290849699, 0.06978533896307716,
            0.05357961023358597, 0.03518302374405403,  0.0153766209980587 ]),

    array([ 0.01357622970587704, 0.03112676196932394,  0.0475792558412464,
            0.06231448562776697, 0.07479799440828833, 0.08457825969750127,
            0.09130170752246178, 0.09472530522753425, 0.09472530522753425,
            0.09130170752246178, 0.08457825969750127, 0.07479799440828833,
            0.06231448562776697,  0.0475792558412464, 0.03112676196932394,
            0.01357622970587704 ]),

    array([ 0.01207415143427393, 0.02772976468699359, 0.04251807415858961,
            0.05594192359670198, 0.06756818423426277, 0.07702288053840513,
            0.08400205107822502, 0.08828135268349631, 0.08972323517810327,
            0.08828135268349631, 0.08400205107822502, 0.07702288053840513,
            0.06756818423426277, 0.05594192359670198, 0.04251807415858961,
            0.02772976468699359, 0.01207415143427393 ]),

    array([ 0.01080800676324173, 0.02485727444748493, 0.03821286512744453,
            0.05047102205314358, 0.06127760335573923, 0.07032145733532533,
            0.07734233756313265, 0.08213824187291634, 0.08457119148157177,
            0.08457119148157177, 0.08213824187291634, 0.07734233756313265,
            0.07032145733532533, 0.06127760335573923, 0.05047102205314358,
            0.03821286512744453, 0.02485727444748493, 0.01080800676324173 ]),

    array([ 0.009730894114863302,  0.0224071133828498, 0.03452227136882061,
            0.04574501081122501, 0.05578332277366701, 0.06437698126966813,
            0.07130335108680329, 0.07638302103292982, 0.07948442169697716,
            0.08052722492439185, 0.07948442169697716, 0.07638302103292982,
            0.07130335108680329, 0.06437698126966813, 0.05578332277366701,
            0.04574501081122501, 0.03452227136882061,  0.0224071133828498,
            0.009730894114863302 ]),

    array([ 0.008807003569576102, 0.02030071490019346, 0.03133602416705451,
            0.04163837078835237, 0.05096505990862021, 0.05909726598075921,
            0.06584431922458832, 0.07104805465919105, 0.07458649323630184,
            0.07637669356536293, 0.07637669356536293, 0.07458649323630184,
            0.07104805465919105, 0.06584431922458832, 0.05909726598075921,
            0.05096505990862021, 0.04163837078835237, 0.03133602416705451,
            0.02030071490019346, 0.008807003569576102 ])
    ]


_quadrature2d_points = [
    array([ [0.333333333333333, ],
            [0.333333333333333, ] ]),

    array([ [0.666666666666667, 0.166666666666667, 0.166666666666667,
             ],
            [0.166666666666667, 0.166666666666667, 0.666666666666667,
             ] ]),

    array([ [0.333333333333333, 0.6, 0.2,
             0.2, ],
            [0.333333333333333, 0.2, 0.2,
             0.6, ] ]),

    array([ [0.10810301816807, 0.445948490915965, 0.445948490915965,
             0.816847572980459, 0.09157621350977101, 0.09157621350977101,
             ],
            [0.445948490915965, 0.445948490915965, 0.10810301816807,
             0.09157621350977101, 0.09157621350977101, 0.816847572980459,
             ] ]),

    array([ [0.333333333333333, 0.05971587178977, 0.470142064105115,
             0.470142064105115, 0.797426985353087, 0.101286507323456,
             0.101286507323456, ],
            [0.333333333333333, 0.470142064105115, 0.470142064105115,
             0.05971587178977, 0.101286507323456, 0.101286507323456,
             0.797426985353087, ] ]),

    array([ [0.501426509658179, 0.24928674517091, 0.24928674517091,
             0.873821971016996, 0.063089014491502, 0.063089014491502,
             0.053145049844817, 0.310352451033784, 0.636502499121399,
             0.310352451033784, 0.636502499121399, 0.053145049844817,
             ],
            [0.24928674517091, 0.24928674517091, 0.501426509658179,
             0.063089014491502, 0.063089014491502, 0.873821971016996,
             0.310352451033784, 0.636502499121399, 0.053145049844817,
             0.053145049844817, 0.310352451033784, 0.636502499121399,
             ] ]),

    array([ [0.333333333333333, 0.47930806784192, 0.26034596607904,
             0.26034596607904, 0.869739794195568, 0.06513010290221601,
             0.06513010290221601, 0.048690315425316, 0.312865496004874,
             0.63844418856981, 0.312865496004874, 0.63844418856981,
             0.048690315425316, ],
            [0.333333333333333, 0.26034596607904, 0.26034596607904,
             0.47930806784192, 0.06513010290221601, 0.06513010290221601,
             0.869739794195568, 0.312865496004874, 0.63844418856981,
             0.048690315425316, 0.048690315425316, 0.312865496004874,
             0.63844418856981, ] ]),

    array([ [0.333333333333333, 0.081414823414554, 0.459292588292723,
             0.459292588292723, 0.65886138449648, 0.17056930775176,
             0.17056930775176, 0.898905543365938, 0.050547228317031,
             0.050547228317031, 0.008394777409958001, 0.263112829634638,
             0.728492392955404, 0.263112829634638, 0.728492392955404,
             0.008394777409958001, ],
            [0.333333333333333, 0.459292588292723, 0.459292588292723,
             0.081414823414554, 0.17056930775176, 0.17056930775176,
             0.65886138449648, 0.050547228317031, 0.050547228317031,
             0.898905543365938, 0.263112829634638, 0.728492392955404,
             0.008394777409958001, 0.008394777409958001, 0.263112829634638,
             0.728492392955404, ] ]),

    array([ [0.333333333333333, 0.020634961602525, 0.489682519198738,
             0.489682519198738, 0.125820817014127, 0.437089591492937,
             0.437089591492937, 0.623592928761935, 0.188203535619033,
             0.188203535619033, 0.910540973211095, 0.044729513394453,
             0.044729513394453, 0.036838412054736, 0.221962989160766,
             0.741198598784498, 0.221962989160766, 0.741198598784498,
             0.036838412054736, ],
            [0.333333333333333, 0.489682519198738, 0.489682519198738,
             0.020634961602525, 0.437089591492937, 0.437089591492937,
             0.125820817014127, 0.188203535619033, 0.188203535619033,
             0.623592928761935, 0.044729513394453, 0.044729513394453,
             0.910540973211095, 0.221962989160766, 0.741198598784498,
             0.036838412054736, 0.036838412054736, 0.221962989160766,
             0.741198598784498, ] ]),

    array([ [0.333333333333333, 0.028844733232685, 0.485577633383657,
             0.485577633383657, 0.781036849029926, 0.109481575485037,
             0.109481575485037, 0.14170721941488, 0.307939838764121,
             0.550352941820999, 0.307939838764121, 0.550352941820999,
             0.14170721941488, 0.025003534762686, 0.246672560639903,
             0.728323904597411, 0.246672560639903, 0.728323904597411,
             0.025003534762686, 0.009540815400299, 0.0668032510122,
             0.9236559335875, 0.0668032510122, 0.9236559335875,
             0.009540815400299, ],
            [0.333333333333333, 0.485577633383657, 0.485577633383657,
             0.028844733232685, 0.109481575485037, 0.109481575485037,
             0.781036849029926, 0.307939838764121, 0.550352941820999,
             0.14170721941488, 0.14170721941488, 0.307939838764121,
             0.550352941820999, 0.246672560639903, 0.728323904597411,
             0.025003534762686, 0.025003534762686, 0.246672560639903,
             0.728323904597411, 0.0668032510122, 0.9236559335875,
             0.009540815400299, 0.009540815400299, 0.0668032510122,
             0.9236559335875, ] ]),

    array([ [-0.06922209654151699, 0.534611048270758, 0.534611048270758,
             0.20206139406829, 0.398969302965855, 0.398969302965855,
             0.593380199137435, 0.203309900431282, 0.203309900431282,
             0.761298175434837, 0.119350912282581, 0.119350912282581,
             0.935270103777448, 0.032364948111276, 0.032364948111276,
             0.050178138310495, 0.356620648261293, 0.593201213428213,
             0.356620648261293, 0.593201213428213, 0.050178138310495,
             0.021022016536166, 0.171488980304042, 0.807489003159792,
             0.171488980304042, 0.807489003159792, 0.021022016536166,
             ],
            [0.534611048270758, 0.534611048270758, -0.06922209654151699,
             0.398969302965855, 0.398969302965855, 0.20206139406829,
             0.203309900431282, 0.203309900431282, 0.593380199137435,
             0.119350912282581, 0.119350912282581, 0.761298175434837,
             0.032364948111276, 0.032364948111276, 0.935270103777448,
             0.356620648261293, 0.593201213428213, 0.050178138310495,
             0.050178138310495, 0.356620648261293, 0.593201213428213,
             0.171488980304042, 0.807489003159792, 0.021022016536166,
             0.021022016536166, 0.171488980304042, 0.807489003159792,
             ] ]),

    array([ [0.02356522045239, 0.488217389773805, 0.488217389773805,
             0.120551215411079, 0.43972439229446, 0.43972439229446,
             0.457579229975768, 0.271210385012116, 0.271210385012116,
             0.7448477089168281, 0.127576145541586, 0.127576145541586,
             0.957365299093579, 0.02131735045321, 0.02131735045321,
             0.115343494534698, 0.275713269685514, 0.608943235779788,
             0.275713269685514, 0.608943235779788, 0.115343494534698,
             0.022838332222257, 0.28132558098994, 0.695836086787803,
             0.28132558098994, 0.695836086787803, 0.022838332222257,
             0.02573405054833, 0.116251915907597, 0.858014033544073,
             0.116251915907597, 0.858014033544073, 0.02573405054833,
             ],
            [0.488217389773805, 0.488217389773805, 0.02356522045239,
             0.43972439229446, 0.43972439229446, 0.120551215411079,
             0.271210385012116, 0.271210385012116, 0.457579229975768,
             0.127576145541586, 0.127576145541586, 0.7448477089168281,
             0.02131735045321, 0.02131735045321, 0.957365299093579,
             0.275713269685514, 0.608943235779788, 0.115343494534698,
             0.115343494534698, 0.275713269685514, 0.608943235779788,
             0.28132558098994, 0.695836086787803, 0.022838332222257,
             0.022838332222257, 0.28132558098994, 0.695836086787803,
             0.116251915907597, 0.858014033544073, 0.02573405054833,
             0.02573405054833, 0.116251915907597, 0.858014033544073,
             ] ]),

    array([ [0.333333333333333, 0.009903630120591001, 0.495048184939705,
             0.495048184939705, 0.062566729780852, 0.468716635109574,
             0.468716635109574, 0.170957326397447, 0.414521336801277,
             0.414521336801277, 0.541200855914337, 0.229399572042831,
             0.229399572042831, 0.77115100960734, 0.11442449519633,
             0.11442449519633, 0.950377217273082, 0.024811391363459,
             0.024811391363459, 0.09485382837957899, 0.268794997058761,
             0.63635117456166, 0.268794997058761, 0.63635117456166,
             0.09485382837957899, 0.018100773278807, 0.291730066734288,
             0.690169159986905, 0.291730066734288, 0.690169159986905,
             0.018100773278807, 0.02223307667409, 0.126357385491669,
             0.851409537834241, 0.126357385491669, 0.851409537834241,
             0.02223307667409, ],
            [0.333333333333333, 0.495048184939705, 0.495048184939705,
             0.009903630120591001, 0.468716635109574, 0.468716635109574,
             0.062566729780852, 0.414521336801277, 0.414521336801277,
             0.170957326397447, 0.229399572042831, 0.229399572042831,
             0.541200855914337, 0.11442449519633, 0.11442449519633,
             0.77115100960734, 0.024811391363459, 0.024811391363459,
             0.950377217273082, 0.268794997058761, 0.63635117456166,
             0.09485382837957899, 0.09485382837957899, 0.268794997058761,
             0.63635117456166, 0.291730066734288, 0.690169159986905,
             0.018100773278807, 0.018100773278807, 0.291730066734288,
             0.690169159986905, 0.126357385491669, 0.851409537834241,
             0.02223307667409, 0.02223307667409, 0.126357385491669,
             0.851409537834241, ] ]),

    array([ [0.022072179275643, 0.488963910362179, 0.488963910362179,
             0.164710561319092, 0.417644719340454, 0.417644719340454,
             0.453044943382323, 0.273477528308839, 0.273477528308839,
             0.645588935174913, 0.177205532412543, 0.177205532412543,
             0.876400233818255, 0.061799883090873, 0.061799883090873,
             0.961218077502598, 0.019390961248701, 0.019390961248701,
             0.057124757403648, 0.172266687821356, 0.770608554774996,
             0.172266687821356, 0.770608554774996, 0.057124757403648,
             0.092916249356972, 0.336861459796345, 0.570222290846683,
             0.336861459796345, 0.570222290846683, 0.092916249356972,
             0.014646950055654, 0.298372882136258, 0.686980167808088,
             0.298372882136258, 0.686980167808088, 0.014646950055654,
             0.001268330932872, 0.118974497696957, 0.879757171370171,
             0.118974497696957, 0.879757171370171, 0.001268330932872,
             ],
            [0.488963910362179, 0.488963910362179, 0.022072179275643,
             0.417644719340454, 0.417644719340454, 0.164710561319092,
             0.273477528308839, 0.273477528308839, 0.453044943382323,
             0.177205532412543, 0.177205532412543, 0.645588935174913,
             0.061799883090873, 0.061799883090873, 0.876400233818255,
             0.019390961248701, 0.019390961248701, 0.961218077502598,
             0.172266687821356, 0.770608554774996, 0.057124757403648,
             0.057124757403648, 0.172266687821356, 0.770608554774996,
             0.336861459796345, 0.570222290846683, 0.092916249356972,
             0.092916249356972, 0.336861459796345, 0.570222290846683,
             0.298372882136258, 0.686980167808088, 0.014646950055654,
             0.014646950055654, 0.298372882136258, 0.686980167808088,
             0.118974497696957, 0.879757171370171, 0.001268330932872,
             0.001268330932872, 0.118974497696957, 0.879757171370171,
             ] ]),

    array([ [-0.013945833716486, 0.506972916858243, 0.506972916858243,
             0.137187291433955, 0.431406354283023, 0.431406354283023,
             0.444612710305711, 0.277693644847144, 0.277693644847144,
             0.747070217917492, 0.126464891041254, 0.126464891041254,
             0.8583832280506281, 0.070808385974686, 0.070808385974686,
             0.9620696595178529, 0.018965170241073, 0.018965170241073,
             0.133734161966621, 0.261311371140087, 0.604954466893291,
             0.261311371140087, 0.604954466893291, 0.133734161966621,
             0.036366677396917, 0.388046767090269, 0.575586555512814,
             0.388046767090269, 0.575586555512814, 0.036366677396917,
             -0.010174883126571, 0.285712220049916, 0.724462663076655,
             0.285712220049916, 0.724462663076655, -0.010174883126571,
             0.036843869875878, 0.215599664072284, 0.747556466051838,
             0.215599664072284, 0.747556466051838, 0.036843869875878,
             0.012459809331199, 0.103575616576386, 0.883964574092416,
             0.103575616576386, 0.883964574092416, 0.012459809331199,
             ],
            [0.506972916858243, 0.506972916858243, -0.013945833716486,
             0.431406354283023, 0.431406354283023, 0.137187291433955,
             0.277693644847144, 0.277693644847144, 0.444612710305711,
             0.126464891041254, 0.126464891041254, 0.747070217917492,
             0.070808385974686, 0.070808385974686, 0.8583832280506281,
             0.018965170241073, 0.018965170241073, 0.9620696595178529,
             0.261311371140087, 0.604954466893291, 0.133734161966621,
             0.133734161966621, 0.261311371140087, 0.604954466893291,
             0.388046767090269, 0.575586555512814, 0.036366677396917,
             0.036366677396917, 0.388046767090269, 0.575586555512814,
             0.285712220049916, 0.724462663076655, -0.010174883126571,
             -0.010174883126571, 0.285712220049916, 0.724462663076655,
             0.215599664072284, 0.747556466051838, 0.036843869875878,
             0.036843869875878, 0.215599664072284, 0.747556466051838,
             0.103575616576386, 0.883964574092416, 0.012459809331199,
             0.012459809331199, 0.103575616576386, 0.883964574092416,
             ] ]),

    array([ [0.333333333333333, 0.005238916103123, 0.497380541948438,
             0.497380541948438, 0.173061122901295, 0.413469438549352,
             0.413469438549352, 0.059082801866017, 0.470458599066991,
             0.470458599066991, 0.518892500060958, 0.240553749969521,
             0.240553749969521, 0.704068411554854, 0.147965794222573,
             0.147965794222573, 0.849069624685052, 0.07546518765747399,
             0.07546518765747399, 0.96680719475395, 0.016596402623025,
             0.016596402623025, 0.103575692245252, 0.296555596579887,
             0.599868711174861, 0.296555596579887, 0.599868711174861,
             0.103575692245252, 0.020083411655416, 0.337723063403079,
             0.6421935249415049, 0.337723063403079, 0.6421935249415049,
             0.020083411655416, -0.004341002614139, 0.204748281642812,
             0.7995927209713271, 0.204748281642812, 0.7995927209713271,
             -0.004341002614139, 0.04194178646801, 0.189358492130623,
             0.768699721401368,  0.189358492130623, 0.768699721401368,
             0.04194178646801,   0.014317320230681, 0.08528361568265699,
             0.900399064086661, 0.08528361568265699, 0.900399064086661,
             0.014317320230681, ],
            [0.333333333333333, 0.497380541948438, 0.497380541948438,
             0.005238916103123, 0.413469438549352, 0.413469438549352,
             0.173061122901295, 0.470458599066991, 0.470458599066991,
             0.059082801866017, 0.240553749969521, 0.240553749969521,
             0.518892500060958, 0.147965794222573, 0.147965794222573,
             0.704068411554854, 0.07546518765747399, 0.07546518765747399,
             0.849069624685052, 0.016596402623025, 0.016596402623025,
             0.96680719475395,  0.296555596579887, 0.599868711174861,
             0.103575692245252, 0.103575692245252, 0.296555596579887,
             0.599868711174861, 0.337723063403079, 0.6421935249415049,
             0.020083411655416, 0.020083411655416, 0.337723063403079,
             0.6421935249415049, 0.204748281642812, 0.7995927209713271,
             -0.004341002614139, -0.004341002614139, 0.204748281642812,
             0.7995927209713271, 0.189358492130623, 0.768699721401368,
             0.04194178646801,  0.04194178646801, 0.189358492130623,
             0.768699721401368, 0.08528361568265699, 0.900399064086661,
             0.014317320230681, 0.014317320230681, 0.08528361568265699,
             0.900399064086661, ] ]),

    array([ [0.333333333333333, 0.005658918886452, 0.497170540556774,
             0.497170540556774, 0.035647354750751, 0.482176322624625,
             0.482176322624625, 0.099520061958437, 0.450239969020782,
             0.450239969020782, 0.199467521245206, 0.400266239377397,
             0.400266239377397, 0.495717464058095, 0.252141267970953,
             0.252141267970953, 0.675905990683077, 0.162047004658461,
             0.162047004658461, 0.848248235478508, 0.075875882260746,
             0.075875882260746, 0.968690546064356, 0.015654726967822,
             0.015654726967822, 0.010186928826919, 0.334319867363658,
             0.655493203809423, 0.334319867363658, 0.655493203809423,
             0.010186928826919, 0.135440871671036, 0.292221537796944,
             0.57233759053202, 0.292221537796944, 0.57233759053202,
             0.135440871671036, 0.054423924290583, 0.31957488542319,
             0.626001190286228, 0.31957488542319, 0.626001190286228,
             0.054423924290583, 0.012868560833637, 0.190704224192292,
             0.796427214974071, 0.190704224192292, 0.796427214974071,
             0.012868560833637, 0.067165782413524, 0.180483211648746,
             0.752351005937729, 0.180483211648746, 0.752351005937729,
             0.067165782413524, 0.014663182224828, 0.080711313679564,
             0.904625504095608, 0.080711313679564, 0.904625504095608,
             0.014663182224828, ],
            [0.333333333333333, 0.497170540556774, 0.497170540556774,
             0.005658918886452, 0.482176322624625, 0.482176322624625,
             0.035647354750751, 0.450239969020782, 0.450239969020782,
             0.099520061958437, 0.400266239377397, 0.400266239377397,
             0.199467521245206, 0.252141267970953, 0.252141267970953,
             0.495717464058095, 0.162047004658461, 0.162047004658461,
             0.675905990683077, 0.075875882260746, 0.075875882260746,
             0.848248235478508, 0.015654726967822, 0.015654726967822,
             0.968690546064356, 0.334319867363658, 0.655493203809423,
             0.010186928826919, 0.010186928826919, 0.334319867363658,
             0.655493203809423, 0.292221537796944, 0.57233759053202,
             0.135440871671036, 0.135440871671036, 0.292221537796944,
             0.57233759053202, 0.31957488542319, 0.626001190286228,
             0.054423924290583, 0.054423924290583, 0.31957488542319,
             0.626001190286228, 0.190704224192292, 0.796427214974071,
             0.012868560833637, 0.012868560833637, 0.190704224192292,
             0.796427214974071, 0.180483211648746, 0.752351005937729,
             0.067165782413524, 0.067165782413524, 0.180483211648746,
             0.752351005937729, 0.080711313679564, 0.904625504095608,
             0.014663182224828, 0.014663182224828, 0.080711313679564,
             0.904625504095608, ] ]),

    array([ [0.333333333333333, 0.013310382738157, 0.493344808630921,
             0.493344808630921, 0.061578811516086, 0.469210594241957,
             0.469210594241957, 0.127437208225989, 0.436281395887006,
             0.436281395887006, 0.210307658653168, 0.394846170673416,
             0.394846170673416, 0.500410862393686, 0.249794568803157,
             0.249794568803157, 0.677135612512315, 0.161432193743843,
             0.161432193743843, 0.846803545029257, 0.076598227485371,
             0.076598227485371, 0.9514951212931, 0.02425243935345,
             0.02425243935345, 0.913707265566071, 0.043146367216965,
             0.043146367216965, 0.00843053620242, 0.358911494940944,
             0.6326579688566361, 0.358911494940944, 0.6326579688566361,
             0.00843053620242, 0.131186551737188, 0.294402476751957,
             0.574410971510855, 0.294402476751957, 0.574410971510855,
             0.131186551737188, 0.050203151565675, 0.325017801641814,
             0.6247790467925119, 0.325017801641814, 0.6247790467925119,
             0.050203151565675, 0.066329263810916, 0.184737559666046,
             0.748933176523037, 0.184737559666046, 0.748933176523037,
             0.066329263810916, 0.011996194566236, 0.218796800013321,
             0.769207005420443, 0.218796800013321, 0.769207005420443,
             0.011996194566236, 0.014858100590125, 0.101179597136408,
             0.8839623022734669, 0.101179597136408, 0.8839623022734669,
             0.014858100590125, -0.035222015287949, 0.020874755282586,
             1.014347260005363, 0.020874755282586, 1.014347260005363,
             -0.035222015287949, ],
            [0.333333333333333, 0.493344808630921, 0.493344808630921,
             0.013310382738157, 0.469210594241957, 0.469210594241957,
             0.061578811516086, 0.436281395887006, 0.436281395887006,
             0.127437208225989, 0.394846170673416, 0.394846170673416,
             0.210307658653168, 0.249794568803157, 0.249794568803157,
             0.500410862393686, 0.161432193743843, 0.161432193743843,
             0.677135612512315, 0.076598227485371, 0.076598227485371,
             0.846803545029257, 0.02425243935345, 0.02425243935345,
             0.9514951212931, 0.043146367216965, 0.043146367216965,
             0.913707265566071, 0.358911494940944, 0.6326579688566361,
             0.00843053620242, 0.00843053620242, 0.358911494940944,
             0.6326579688566361, 0.294402476751957, 0.574410971510855,
             0.131186551737188, 0.131186551737188, 0.294402476751957,
             0.574410971510855, 0.325017801641814, 0.6247790467925119,
             0.050203151565675, 0.050203151565675, 0.325017801641814,
             0.6247790467925119, 0.184737559666046, 0.748933176523037,
             0.066329263810916, 0.066329263810916, 0.184737559666046,
             0.748933176523037, 0.218796800013321, 0.769207005420443,
             0.011996194566236, 0.011996194566236, 0.218796800013321,
             0.769207005420443, 0.101179597136408, 0.8839623022734669,
             0.014858100590125, 0.014858100590125, 0.101179597136408,
             0.8839623022734669, 0.020874755282586, 1.014347260005363,
             -0.035222015287949, -0.035222015287949, 0.020874755282586,
             1.014347260005363, ] ]),

    array([ [0.333333333333333, 0.020780025853987, 0.489609987073006,
             0.489609987073006, 0.090926214604215, 0.454536892697893,
             0.454536892697893, 0.197166638701138, 0.401416680649431,
             0.401416680649431, 0.488896691193805, 0.255551654403098,
             0.255551654403098, 0.645844115695741, 0.17707794215213,
             0.17707794215213, 0.779877893544096, 0.110061053227952,
             0.110061053227952, 0.888942751496321, 0.05552862425184,
             0.05552862425184, 0.974756272445543, 0.012621863777229,
             0.012621863777229, 0.003611417848412, 0.395754787356943,
             0.600633794794645, 0.395754787356943, 0.600633794794645,
             0.003611417848412, 0.13446675453078, 0.307929983880436,
             0.557603261588784, 0.307929983880436, 0.557603261588784,
             0.13446675453078, 0.014446025776115, 0.26456694840652,
             0.720987025817365, 0.26456694840652, 0.720987025817365,
             0.014446025776115, 0.046933578838178, 0.358539352205951,
             0.594527068955871, 0.358539352205951, 0.594527068955871,
             0.046933578838178, 0.002861120350567, 0.157807405968595,
             0.839331473680839, 0.157807405968595, 0.839331473680839,
             0.002861120350567, 0.223861424097916, 0.075050596975911,
             0.701087978926173, 0.075050596975911, 0.701087978926173,
             0.223861424097916, 0.03464707481676, 0.142421601113383,
             0.822931324069857, 0.142421601113383, 0.822931324069857,
             0.03464707481676, 0.010161119296278, 0.065494628082938,
             0.924344252620784, 0.065494628082938, 0.924344252620784,
             0.010161119296278, ],
            [0.333333333333333, 0.489609987073006, 0.489609987073006,
             0.020780025853987, 0.454536892697893, 0.454536892697893,
             0.090926214604215, 0.401416680649431, 0.401416680649431,
             0.197166638701138, 0.255551654403098, 0.255551654403098,
             0.488896691193805, 0.17707794215213, 0.17707794215213,
             0.645844115695741, 0.110061053227952, 0.110061053227952,
             0.779877893544096, 0.05552862425184, 0.05552862425184,
             0.888942751496321, 0.012621863777229, 0.012621863777229,
             0.974756272445543, 0.395754787356943, 0.600633794794645,
             0.003611417848412, 0.003611417848412, 0.395754787356943,
             0.600633794794645, 0.307929983880436, 0.557603261588784,
             0.13446675453078, 0.13446675453078, 0.307929983880436,
             0.557603261588784, 0.26456694840652, 0.720987025817365,
             0.014446025776115, 0.014446025776115, 0.26456694840652,
             0.720987025817365, 0.358539352205951, 0.594527068955871,
             0.046933578838178, 0.046933578838178, 0.358539352205951,
             0.594527068955871, 0.157807405968595, 0.839331473680839,
             0.002861120350567, 0.002861120350567, 0.157807405968595,
             0.839331473680839, 0.075050596975911, 0.701087978926173,
             0.223861424097916, 0.223861424097916, 0.075050596975911,
             0.701087978926173, 0.142421601113383, 0.822931324069857,
             0.03464707481676, 0.03464707481676, 0.142421601113383,
             0.822931324069857, 0.065494628082938, 0.924344252620784,
             0.010161119296278, 0.010161119296278, 0.065494628082938,
             0.924344252620784, ] ]),

    array([ [0.333333333333333, -0.0019009287044, 0.5009504643522,
             0.5009504643522, 0.023574084130543, 0.488212957934729,
             0.488212957934729, 0.089726636099435, 0.455136681950283,
             0.455136681950283, 0.196007481363421, 0.401996259318289,
             0.401996259318289, 0.488214180481157, 0.255892909759421,
             0.255892909759421, 0.647023488009788, 0.176488255995106,
             0.176488255995106, 0.791658289326483, 0.104170855336758,
             0.104170855336758, 0.89386207231814, 0.05306896384093,
             0.05306896384093, 0.916762569607942, 0.041618715196029,
             0.041618715196029, 0.976836157186356, 0.011581921406822,
             0.011581921406822, 0.048741583664839, 0.344855770229001,
             0.60640264610616, 0.344855770229001, 0.60640264610616,
             0.048741583664839, 0.006314115948605, 0.377843269594854,
             0.615842614456541, 0.377843269594854, 0.615842614456541,
             0.006314115948605, 0.134316520547348, 0.306635479062357,
             0.559048000390295, 0.306635479062357, 0.559048000390295,
             0.134316520547348, 0.013973893962392, 0.249419362774742,
             0.736606743262866, 0.249419362774742, 0.736606743262866,
             0.013973893962392, 0.07554913290976401, 0.212775724802802,
             0.711675142287434, 0.212775724802802, 0.711675142287434,
             0.07554913290976401, -0.008368153208227, 0.146965436053239,
             0.861402717154987, 0.146965436053239, 0.861402717154987,
             -0.008368153208227, 0.026686063258714, 0.137726978828923,
             0.835586957912363, 0.137726978828923, 0.835586957912363,
             0.026686063258714, 0.010547719294141, 0.059696109149007,
             0.929756171556853, 0.059696109149007, 0.929756171556853,
             0.010547719294141, ],
            [0.333333333333333, 0.5009504643522, 0.5009504643522,
             -0.0019009287044, 0.488212957934729, 0.488212957934729,
             0.023574084130543, 0.455136681950283, 0.455136681950283,
             0.089726636099435, 0.401996259318289, 0.401996259318289,
             0.196007481363421, 0.255892909759421, 0.255892909759421,
             0.488214180481157, 0.176488255995106, 0.176488255995106,
             0.647023488009788, 0.104170855336758, 0.104170855336758,
             0.791658289326483, 0.05306896384093, 0.05306896384093,
             0.89386207231814, 0.041618715196029, 0.041618715196029,
             0.916762569607942, 0.011581921406822, 0.011581921406822,
             0.976836157186356, 0.344855770229001, 0.60640264610616,
             0.048741583664839, 0.048741583664839, 0.344855770229001,
             0.60640264610616, 0.377843269594854, 0.615842614456541,
             0.006314115948605, 0.006314115948605, 0.377843269594854,
             0.615842614456541, 0.306635479062357, 0.559048000390295,
             0.134316520547348, 0.134316520547348, 0.306635479062357,
             0.559048000390295, 0.249419362774742, 0.736606743262866,
             0.013973893962392, 0.013973893962392, 0.249419362774742,
             0.736606743262866, 0.212775724802802, 0.711675142287434,
             0.07554913290976401, 0.07554913290976401, 0.212775724802802,
             0.711675142287434, 0.146965436053239, 0.861402717154987,
             -0.008368153208227, -0.008368153208227, 0.146965436053239,
             0.861402717154987, 0.137726978828923, 0.835586957912363,
             0.026686063258714, 0.026686063258714, 0.137726978828923,
             0.835586957912363, 0.059696109149007, 0.929756171556853,
             0.010547719294141, 0.010547719294141, 0.059696109149007,
             0.929756171556853, ] ]),
    ]

_quadrature2d_weights = [

    array([ 1.0, ]),

    array([ 0.333333333333333, 0.333333333333333, 0.333333333333333,
            ]),

    array([ -0.5625, 0.520833333333333, 0.520833333333333,
            0.520833333333333, ]),

    array([ 0.223381589678011, 0.223381589678011, 0.223381589678011,
            0.109951743655322, 0.109951743655322, 0.109951743655322,
            ]),

    array([ 0.225, 0.132394152788506, 0.132394152788506,
            0.132394152788506, 0.125939180544827, 0.125939180544827,
            0.125939180544827, ]),

    array([ 0.116786275726379, 0.116786275726379, 0.116786275726379,
            0.050844906370207, 0.050844906370207, 0.050844906370207,
            0.082851075618374, 0.082851075618374, 0.082851075618374,
            0.082851075618374, 0.082851075618374, 0.082851075618374,
            ]),

    array([ -0.149570044467682, 0.175615257433208, 0.175615257433208,
            0.175615257433208, 0.053347235608838, 0.053347235608838,
            0.053347235608838, 0.077113760890257, 0.077113760890257,
            0.077113760890257, 0.077113760890257, 0.077113760890257,
            0.077113760890257, ]),

    array([ 0.144315607677787, 0.09509163426728499, 0.09509163426728499,
            0.09509163426728499, 0.103217370534718, 0.103217370534718,
            0.103217370534718, 0.032458497623198, 0.032458497623198,
            0.032458497623198, 0.027230314174435, 0.027230314174435,
            0.027230314174435, 0.027230314174435, 0.027230314174435,
            0.027230314174435, ]),

    array([ 0.097135796282799, 0.031334700227139, 0.031334700227139,
            0.031334700227139, 0.077827541004774, 0.077827541004774,
            0.077827541004774, 0.07964773892721, 0.07964773892721,
            0.07964773892721, 0.025577675658698, 0.025577675658698,
            0.025577675658698, 0.043283539377289, 0.043283539377289,
            0.043283539377289, 0.043283539377289, 0.043283539377289,
            0.043283539377289, ]),

    array([ 0.090817990382754, 0.036725957756467, 0.036725957756467,
            0.036725957756467, 0.045321059435528, 0.045321059435528,
            0.045321059435528, 0.07275791684542, 0.07275791684542,
            0.07275791684542, 0.07275791684542, 0.07275791684542,
            0.07275791684542, 0.028327242531057, 0.028327242531057,
            0.028327242531057, 0.028327242531057, 0.028327242531057,
            0.028327242531057, 0.009421666963733, 0.009421666963733,
            0.009421666963733, 0.009421666963733, 0.009421666963733,
            0.009421666963733, ]),

    array([ 0.000927006328961, 0.000927006328961, 0.000927006328961,
            0.07714953491481299, 0.07714953491481299, 0.07714953491481299,
            0.059322977380774, 0.059322977380774, 0.059322977380774,
            0.036184540503418, 0.036184540503418, 0.036184540503418,
            0.013659731002678, 0.013659731002678, 0.013659731002678,
            0.052337111962204, 0.052337111962204, 0.052337111962204,
            0.052337111962204, 0.052337111962204, 0.052337111962204,
            0.020707659639141, 0.020707659639141, 0.020707659639141,
            0.020707659639141, 0.020707659639141, 0.020707659639141,
            ]),

    array([ 0.025731066440455, 0.025731066440455, 0.025731066440455,
            0.043692544538038, 0.043692544538038, 0.043692544538038,
            0.06285822421788501, 0.06285822421788501, 0.06285822421788501,
            0.034796112930709, 0.034796112930709, 0.034796112930709,
            0.006166261051559, 0.006166261051559, 0.006166261051559,
            0.040371557766381, 0.040371557766381, 0.040371557766381,
            0.040371557766381, 0.040371557766381, 0.040371557766381,
            0.022356773202303, 0.022356773202303, 0.022356773202303,
            0.022356773202303, 0.022356773202303, 0.022356773202303,
            0.017316231108659, 0.017316231108659, 0.017316231108659,
            0.017316231108659, 0.017316231108659, 0.017316231108659,
            ]),

    array([ 0.052520923400802, 0.01128014520933, 0.01128014520933,
            0.01128014520933, 0.031423518362454, 0.031423518362454,
            0.031423518362454, 0.047072502504194, 0.047072502504194,
            0.047072502504194, 0.047363586536355, 0.047363586536355,
            0.047363586536355, 0.031167529045794, 0.031167529045794,
            0.031167529045794, 0.007975771465074, 0.007975771465074,
            0.007975771465074, 0.036848402728732, 0.036848402728732,
            0.036848402728732, 0.036848402728732, 0.036848402728732,
            0.036848402728732, 0.017401463303822, 0.017401463303822,
            0.017401463303822, 0.017401463303822, 0.017401463303822,
            0.017401463303822, 0.015521786839045, 0.015521786839045,
            0.015521786839045, 0.015521786839045, 0.015521786839045,
            0.015521786839045, ]),

    array([ 0.021883581369429, 0.021883581369429, 0.021883581369429,
            0.032788353544125, 0.032788353544125, 0.032788353544125,
            0.051774104507292, 0.051774104507292, 0.051774104507292,
            0.042162588736993, 0.042162588736993, 0.042162588736993,
            0.014433699669777, 0.014433699669777, 0.014433699669777,
            0.0049234036024, 0.0049234036024, 0.0049234036024,
            0.024665753212564, 0.024665753212564, 0.024665753212564,
            0.024665753212564, 0.024665753212564, 0.024665753212564,
            0.038571510787061, 0.038571510787061, 0.038571510787061,
            0.038571510787061, 0.038571510787061, 0.038571510787061,
            0.014436308113534, 0.014436308113534, 0.014436308113534,
            0.014436308113534, 0.014436308113534, 0.014436308113534,
            0.005010228838501, 0.005010228838501, 0.005010228838501,
            0.005010228838501, 0.005010228838501, 0.005010228838501,
            ]),

    array([ 0.001916875642849, 0.001916875642849, 0.001916875642849,
            0.044249027271145, 0.044249027271145, 0.044249027271145,
            0.051186548718852, 0.051186548718852, 0.051186548718852,
            0.023687735870688, 0.023687735870688, 0.023687735870688,
            0.013289775690021, 0.013289775690021, 0.013289775690021,
            0.004748916608192, 0.004748916608192, 0.004748916608192,
            0.038550072599593, 0.038550072599593, 0.038550072599593,
            0.038550072599593, 0.038550072599593, 0.038550072599593,
            0.027215814320624, 0.027215814320624, 0.027215814320624,
            0.027215814320624, 0.027215814320624, 0.027215814320624,
            0.002182077366797, 0.002182077366797, 0.002182077366797,
            0.002182077366797, 0.002182077366797, 0.002182077366797,
            0.021505319847731, 0.021505319847731, 0.021505319847731,
            0.021505319847731, 0.021505319847731, 0.021505319847731,
            0.007673942631049, 0.007673942631049, 0.007673942631049,
            0.007673942631049, 0.007673942631049, 0.007673942631049,
            ]),

    array([ 0.046875697427642, 0.006405878578585, 0.006405878578585,
            0.006405878578585, 0.041710296739387, 0.041710296739387,
            0.041710296739387, 0.026891484250064, 0.026891484250064,
            0.026891484250064, 0.04213252276165, 0.04213252276165,
            0.04213252276165, 0.030000266842773, 0.030000266842773,
            0.030000266842773, 0.014200098925024, 0.014200098925024,
            0.014200098925024, 0.003582462351273, 0.003582462351273,
            0.003582462351273, 0.032773147460627, 0.032773147460627,
            0.032773147460627, 0.032773147460627, 0.032773147460627,
            0.032773147460627, 0.015298306248441, 0.015298306248441,
            0.015298306248441, 0.015298306248441, 0.015298306248441,
            0.015298306248441, 0.002386244192839, 0.002386244192839,
            0.002386244192839, 0.002386244192839, 0.002386244192839,
            0.002386244192839, 0.019084792755899, 0.019084792755899,
            0.019084792755899, 0.019084792755899, 0.019084792755899,
            0.019084792755899, 0.006850054546542, 0.006850054546542,
            0.006850054546542, 0.006850054546542, 0.006850054546542,
            0.006850054546542, ]),

    array([ 0.033437199290803, 0.005093415440507, 0.005093415440507,
            0.005093415440507, 0.014670864527638, 0.014670864527638,
            0.014670864527638, 0.024350878353672, 0.024350878353672,
            0.024350878353672, 0.031107550868969, 0.031107550868969,
            0.031107550868969, 0.03125711121862, 0.03125711121862,
            0.03125711121862, 0.024815654339665, 0.024815654339665,
            0.024815654339665, 0.014056073070557, 0.014056073070557,
            0.014056073070557, 0.003194676173779, 0.003194676173779,
            0.003194676173779, 0.008119655318993, 0.008119655318993,
            0.008119655318993, 0.008119655318993, 0.008119655318993,
            0.008119655318993, 0.026805742283163, 0.026805742283163,
            0.026805742283163, 0.026805742283163, 0.026805742283163,
            0.026805742283163, 0.018459993210822, 0.018459993210822,
            0.018459993210822, 0.018459993210822, 0.018459993210822,
            0.018459993210822, 0.008476868534328, 0.008476868534328,
            0.008476868534328, 0.008476868534328, 0.008476868534328,
            0.008476868534328, 0.018292796770025, 0.018292796770025,
            0.018292796770025, 0.018292796770025, 0.018292796770025,
            0.018292796770025, 0.006665632004165, 0.006665632004165,
            0.006665632004165, 0.006665632004165, 0.006665632004165,
            0.006665632004165, ]),

    array([ 0.030809939937647, 0.009072436679404, 0.009072436679404,
            0.009072436679404, 0.018761316939594, 0.018761316939594,
            0.018761316939594, 0.019441097985477, 0.019441097985477,
            0.019441097985477, 0.02775394861081, 0.02775394861081,
            0.02775394861081, 0.032256225351457, 0.032256225351457,
            0.032256225351457, 0.025074032616922, 0.025074032616922,
            0.025074032616922, 0.015271927971832, 0.015271927971832,
            0.015271927971832, 0.006793922022963, 0.006793922022963,
            0.006793922022963, -0.00222309872992, -0.00222309872992,
            -0.00222309872992, 0.006331914076406, 0.006331914076406,
            0.006331914076406, 0.006331914076406, 0.006331914076406,
            0.006331914076406, 0.027257538049138, 0.027257538049138,
            0.027257538049138, 0.027257538049138, 0.027257538049138,
            0.027257538049138, 0.017676785649465, 0.017676785649465,
            0.017676785649465, 0.017676785649465, 0.017676785649465,
            0.017676785649465, 0.01837948463807, 0.01837948463807,
            0.01837948463807, 0.01837948463807, 0.01837948463807,
            0.01837948463807, 0.008104732808191999, 0.008104732808191999,
            0.008104732808191999, 0.008104732808191999, 0.008104732808191999,
            0.008104732808191999, 0.007634129070725, 0.007634129070725,
            0.007634129070725, 0.007634129070725, 0.007634129070725,
            0.007634129070725, 4.6187660794e-05, 4.6187660794e-05,
            4.6187660794e-05, 4.6187660794e-05, 4.6187660794e-05,
            4.6187660794e-05, ]),

    array([ 0.032906331388919, 0.010330731891272, 0.010330731891272,
            0.010330731891272, 0.022387247263016, 0.022387247263016,
            0.022387247263016, 0.030266125869468, 0.030266125869468,
            0.030266125869468, 0.030490967802198, 0.030490967802198,
            0.030490967802198, 0.024159212741641, 0.024159212741641,
            0.024159212741641, 0.016050803586801, 0.016050803586801,
            0.016050803586801, 0.008084580261784, 0.008084580261784,
            0.008084580261784, 0.002079362027485, 0.002079362027485,
            0.002079362027485, 0.003884876904981, 0.003884876904981,
            0.003884876904981, 0.003884876904981, 0.003884876904981,
            0.003884876904981, 0.025574160612022, 0.025574160612022,
            0.025574160612022, 0.025574160612022, 0.025574160612022,
            0.025574160612022, 0.008880903573337999, 0.008880903573337999,
            0.008880903573337999, 0.008880903573337999, 0.008880903573337999,
            0.008880903573337999, 0.016124546761731, 0.016124546761731,
            0.016124546761731, 0.016124546761731, 0.016124546761731,
            0.016124546761731, 0.002491941817491, 0.002491941817491,
            0.002491941817491, 0.002491941817491, 0.002491941817491,
            0.002491941817491, 0.018242840118951, 0.018242840118951,
            0.018242840118951, 0.018242840118951, 0.018242840118951,
            0.018242840118951, 0.010258563736199, 0.010258563736199,
            0.010258563736199, 0.010258563736199, 0.010258563736199,
            0.010258563736199, 0.003799928855302, 0.003799928855302,
            0.003799928855302, 0.003799928855302, 0.003799928855302,
            0.003799928855302, ]),

    array([ 0.033057055541624, 0.000867019185663, 0.000867019185663,
            0.000867019185663, 0.011660052716448, 0.011660052716448,
            0.011660052716448, 0.022876936356421, 0.022876936356421,
            0.022876936356421, 0.030448982673938, 0.030448982673938,
            0.030448982673938, 0.030624891725355, 0.030624891725355,
            0.030624891725355, 0.0243680576768, 0.0243680576768,
            0.0243680576768, 0.015997432032024, 0.015997432032024,
            0.015997432032024, 0.007698301815602, 0.007698301815602,
            0.007698301815602, -0.0006320604974879999, -0.0006320604974879999,
            -0.0006320604974879999, 0.001751134301193, 0.001751134301193,
            0.001751134301193, 0.016465839189576, 0.016465839189576,
            0.016465839189576, 0.016465839189576, 0.016465839189576,
            0.016465839189576, 0.004839033540485, 0.004839033540485,
            0.004839033540485, 0.004839033540485, 0.004839033540485,
            0.004839033540485, 0.02580490653465, 0.02580490653465,
            0.02580490653465, 0.02580490653465, 0.02580490653465,
            0.02580490653465, 0.008471091054441, 0.008471091054441,
            0.008471091054441, 0.008471091054441, 0.008471091054441,
            0.008471091054441, 0.01835491410628, 0.01835491410628,
            0.01835491410628, 0.01835491410628, 0.01835491410628,
            0.01835491410628, 0.000704404677908, 0.000704404677908,
            0.000704404677908, 0.000704404677908, 0.000704404677908,
            0.000704404677908, 0.010112684927462, 0.010112684927462,
            0.010112684927462, 0.010112684927462, 0.010112684927462,
            0.010112684927462, 0.00357390938595, 0.00357390938595,
            0.00357390938595, 0.00357390938595, 0.00357390938595,
            0.00357390938595, ]),
    ]
