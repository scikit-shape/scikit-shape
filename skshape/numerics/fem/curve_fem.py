import numpy as np
from numpy.random import rand
from ..matrix import TridiagMatrix, BlockDiagMatrix


def _curve_load_vector(curve, f=None, f2=None, quad=None):
    n = curve.size()

    if (f is None) and (f2 is None):
        d = curve.element_sizes()
        result = 0.5*d
        result[0] += 0.5*d[n-1]
        result[1:n] += 0.5*d[0:n-1]
        return result

    # Setup quadrature, default order=2 if not specified as argument
    if quad is None: # quad_degree=3, exact for (quadratic f) x linears
        quad = curve.ref_element.quadrature( degree=3 )

    # Compute the term corresponding to the weighting function f2.

    if f2 is not None:
        F2 = sum(( w * f2( curve, pt ) for pt,w in quad.iterpoints() ))
        t = curve.tangents( smoothness='pwconst' )
        element_integrals = F2[0]*t[0] + F2[1]*t[1] # = \int_k dot(F2(x),t_k)
        result2 = -element_integrals  # integrals at elements k
        result2[0] += element_integrals[n-1]
        result2[1:n] += element_integrals[0:n-1] # integrals at elements k-1
        if f is None:
            return result2

    # Compute the term corresponding to the weighting function f.

    # Evaluate and store basis fct values at quadrature points.
    basis_fct = curve.ref_element.eval_basis_fct( quad )

    # Loop over quad points and obtain the sums for \int f\phi_k simultaneously.
    integral0 = np.zeros(n)  # integrals at element k-1
    integral1 = np.zeros(n)  # integrals at element k

    for k,(pt,w) in enumerate(quad.iterpoints()):
        # The value of f on all elements at the same local quadrature point.
        f_at_q_pt = f( curve, pt )

        # Add the value of w_k * f(x_k,j) * \phi_j(x_k) to the quadrature.
        integral1 += w * f_at_q_pt * basis_fct[1,k]

        # Add the value of w_k * f(x_k,j-1) * \phi_{j-1}(x_k) to the quadrature.
        integral0[1:n] += w * f_at_q_pt[0:n-1] * basis_fct[0,k]
        integral0[0] += w * f_at_q_pt[n-1] * basis_fct[0,k]

    # Calculate the element sizes to scale the integrals.
    d = curve.element_sizes()

    integral1[:] = d * integral1[:]
    integral0[1:n] = d[0:n-1] * integral0[1:n]
    integral0[0] = d[n-1] * integral0[0]

    result = integral0 + integral1
    if f2 is not None:  result += result2
    return result

def load_vector(curves, f=None, f2=None, quad=None):

    if not curves.has_submeshes(): # It is a single curve.
        return _curve_load_vector( curves, f, f2, quad )

    # It has multiple curves; process individually & combine.
    load_vectors = [ _curve_load_vector( curve, f, f2, quad )
                     for curve in curves.submeshes() ]

    return np.concatenate( load_vectors )


def _curve_mass_matrix(curve):
    """
    Returns the mass matrix

       M_{ij} = \int_\Gamma \phi_i \phi_j ds

    of the current curve \Gamma. {\phi_i} are piecewise linear
    test functions.
    """
    n = curve.size()
    d = curve.element_sizes()

    c = d / 6.0
    a = np.empty(n)
    a[0] = c[n-1]
    a[1:n] = c[0:n-1]
    b = 2.0 * (a + c)

    return TridiagMatrix(a,b,c)

def mass_matrix(curves):
    if not curves.has_submeshes(): # It is a single curve.
        return _curve_mass_matrix( curves )

    # It has multiple curves; process individually & combine.
    submatrices = [_curve_mass_matrix(curve) for curve in curves.submeshes()]

    return BlockDiagMatrix( submatrices )


def _curve_stiffness_matrix(curve):
    """
    Returns the stiffness matrix

       A_{ij} = \int_\Gamma D\phi_i D\phi_j ds

    of the current curve \Gamma. {\phi_i} are piecewise linear
    test functions and D\phi_i denotes the tangential derivative
    of \phi_i.
    """
    n = curve.size()
    d = curve.element_sizes()

    c = -1.0 / d
    a = np.empty(n)
    a[0] = c[n-1]
    a[1:n] = c[0:n-1]
    b = -(a + c)

    return TridiagMatrix(a,b,c)

def stiffness_matrix(curves):
    if not curves.has_submeshes(): # It is a single curve.
        return _curve_stiffness_matrix( curves )

    # It has multiple curves; process individually & combine.
    submatrices = [ _curve_stiffness_matrix( curve )
                    for curve in curves.submeshes() ]

    return BlockDiagMatrix( submatrices )


def _curve_pwconst_weighted_mass_matrix(curve, weight):
    """
    Returns the weighted mass matrix

       M_{ij} = \int_\Gamma w(x) \phi_i \phi_j d\Gamma

    of the current curve \Gamma. {\phi_i} are piecewise linear
    test functions.
    weight is a vector that represents the weight function w(x)
    that is constant on each element.
    """
    n = curve.size()
    d = curve.element_sizes()

    c = weight * d / 6.0
    a = np.empty(n)
    a[0] = c[n-1]
    a[1:n] = c[0:n-1]
    b = 2.0 * (a + c)

    return TridiagMatrix(a,b,c)

def pwconst_weighted_mass_matrix(curves, weight):
    if not curves.has_submeshes(): # It is a single curve.
        return _curve_pwconst_weighted_mass_matrix( curves, weight )

    # It has multiple curves; process individually & combine.
    start = 0
    submatrices = []
    for curve in curves.submeshes:
        end = start + curve.size()
        matrix = _curve_pwconst_weighted_mass_matrix( curve,
                                                      weight[start:end] )
        submatrices.append( matrix )
        start = end

    return BlockDiagMatrix( submatrices )


def _curve_normal_matrix(curve, normal_component):
    """
    Returns the normal matrix num (=0,1) given by

       N_{ij} = \int_\Gamma n(x) \phi_i \phi_j ds

    of the current curve \Gamma. n(x) is the component num of outward
    unit normal vector and {\phi_i} are piecewise linear test functions.
    """

    if normal_component not in [0,1]:
        raise ValueException('normal_component number for normal matrix should be one of 0 or 1!')

    normals = curve.normals( smoothness='pwconst' )

    matrix = pwconst_weighted_mass_matrix( curve, normals[normal_component,:] )

    return matrix

def normal_matrix(curves, normal_component):
    if not curves.has_submeshes(): # It is a single curve.
        return _curve_normal_matrix( curves, normal_component )

    # It has multiple curves; process individually & combine.
    submatrices = [ _curve_normal_matrix( curve, normal_component )
                    for curve in curves.submeshes() ]

    return BlockDiagMatrix( submatrices )


def _curve_weighted_mass_matrix(curve, f, quad=None):
    n = curve.size()

    # Setup quadrature, default order=3 if not specified as argument
    if quad is None: # quad_degree=4, exact for (quadratic f) x (linear)^2
        quad = curve.ref_element.quadrature( degree=4 )

    # Evaluate and store basis fct products at quadrature points.
    basis_prod_at_pt = curve.ref_element.eval_basis_fct_products( quad )

    # Calculate the element sizes to scale the integrals.
    d = curve.element_sizes()

    # Compute the diagonal elements of the mass matrix:
    #    m(i,i) = \int_\Gamma_{i} f \phi_i^2 + \int_\Gamma_{i-1} f \phi_i^2

    # Loop over quad points and obtain the sums for \int f\phi_k simultaneously.
    integral0 = np.zeros(n)
    integral1 = np.zeros(n)

    for k,(pt,w) in enumerate(quad.iterpoints()):

        # The value of f on all elements at the same local quadrature point.
        f_at_q_pt = f( curve, pt )

        # Add w_k * f(x_k,j) * sqr(\phi_j(x_k)) to the quadrature.
        integral0 += w * f_at_q_pt * basis_prod_at_pt[1,1,k]

        # Add w_k * f(x_k,j-1) * sqr(\phi_{j-1}(x_k)) to the quadrature.
        integral1[1:n] += w * f_at_q_pt[0:n-1] * basis_prod_at_pt[0,0,k]
        integral1[0] += w * f_at_q_pt[n-1] * basis_prod_at_pt[0,0,k]

    # Scale the integrals with element sizes.
    integral0 = d * integral0
    integral1[1:n] = d[0:n-1] * integral1[1:n]
    integral1[0] = d[n-1] * integral1[0]

    b = integral0 + integral1

    # Compute the elements above the diagonal of the mass matrix
    #    m(i,i+1) = \int_\Gamma_{i} f \phi_i \phi_{i+1}

    integral0[:] = 0.0
    for k,(pt,w) in enumerate(quad.iterpoints()):
        # The value of f on all elements at the same local quadrature point.
        f_at_q_pt = f( curve, pt )

        # Add the value of w_k * f(x_k,j) * \phi_j(x_k) to the quadrature.
        integral0 = integral0 + w * f_at_q_pt * basis_prod_at_pt[0,1,k]

    # Scale the integrals with element sizes.
    integral0 = d * integral0
    c = integral0 # the vector storing upper diagonal, m(i,i+1)

    # The lower diagonal 'a' can is the shifted version of the upper diagonal 'c'
    a = np.empty(n)
    a[0] = c[n-1]
    a[1:n] = c[0:n-1]

    return TridiagMatrix(a,b,c)

def weighted_mass_matrix(curves, f, quad=None):
    if not curves.has_submeshes(): # It is a single curve.
        return _curve_weighted_mass_matrix( curves, f, quad )

    # It has multiple curves; process individually & combine.
    submatrices = [ _curve_weighted_mass_matrix( curve, f, quad )
                    for curve in curves.submeshes() ]

    return BlockDiagMatrix( submatrices )


def _curve_weighted_stiffness_matrix(curve, f, quad=None):
    n = curve.size()

    # Setup quadrature, default order=2 if not specified as argument
    if quad is None: # quad_degree=2, exact for (quadratic f) x (const)^2
        quad = curve.ref_element.quadrature( degree=2 )

    # Let F = \sum_k w_k * f(x_k) at the quadrature point k (on all elements).
    F = sum(( w * f(curve, pt) for pt,w in quad.iterpoints() ))

    # Compute the diagonals for the tridiag return matrix.
    d = curve.element_sizes()
    if F.ndim == 1: # f is a scalar-valued function
        c = -F / d
    elif F.ndim == 3: # f is a matrix-valued function
        t = curve.tangents( smoothness='pwconst' )
        c = -( t[0,:]*F[0,0,:]*t[0,:] + t[0,:]*F[0,1,:]*t[1,:] +
               t[1,:]*F[1,0,:]*t[0,:] + t[1,:]*F[1,1,:]*t[1,:] ) / d
    else:
        raise ValueError("f must be a scalar or matrix-valued MeshFunction!")

    a = np.empty(n)
    a[0] = c[n-1]
    a[1:n] = c[0:n-1]

    b = -(a + c)

    return TridiagMatrix(a,b,c)

def weighted_stiffness_matrix(curves, f, quad=None):
    if not curves.has_submeshes(): # It is a single curve.
        return _curve_weighted_stiffness_matrix( curves, f, quad )

    # It has multiple curves; process individually & combine.
    submatrices = [ _curve_weighted_stiffness_matrix( curve, f, quad )
                    for curve in curves.submeshes() ]

    return BlockDiagMatrix( submatrices )


###########################################################################

class ReferenceElement:
    def __init__(self):
        self._quadrature = None

    def quadrature(self, degree=None, order=None):
        if (degree is None) and (order is None):
            if self._quadrature is not None:
                return self._quadrature
            else:
                order = 0

        if order is not None:
            if (self._quadrature is None) or (self._quadrature.order() != order):
                from numerics.integration import Quadrature1d
                self._quadrature = Quadrature1d( None, order )
        else:
            if (self._quadrature is None) or (self._quadrature.degree() != degree):
                from numerics.integration import Quadrature1d
                self._quadrature = Quadrature1d( degree, None )

        return self._quadrature

    def local_to_global(self, s, curve, mask=None, coarsened=False):
        """Returns the global coordinate for all elements at local coordinate s.

        Given a curve (which stores the information for all its elements)
        and a local coordinate s from [0,1] on the reference element,
        this function returns the corresponding global coordinates X on
        all the elements.
        If mask, an array of indices of some of the elements, is given,
        then the global coordinates on only the elements specified by
        the indices in mask is given.
        If coarsened is set to True, then the global coordinates are
        not computed for the current elements. They are computed for the
        elements that would result from coarsening the current elements.

        For example, if the curve is the square with the corner nodes
        {(-1,-1), (1,-1), (1,1), (-1,1)} ordered counterclockwise
        and s = 0.25, then the corresponding global coordinate vector
        is X = {(-0.5,-1.0), (1.0,-0.5), (0.5,1.0), (-1.0,0.5)},
        more specifically the NumPy array:
           X = [[-0.5, 1.0, 0.5, -1.0], [-1.0, -0.5, 1.0, 0.5]].

        If in addition the mask array is specified as [0,2], then
        the global coordinates are X = {(-0.5,-1.0), (0.5,1.0)},
        the coordinates of only the 0th and 2nd elements.

        If mask = [1] and coarsened = True, then the global coordinates
        are sought for the coarse element resulting from the coarsening
        of the element 1 (and 2). When element 1 is coarsened, the
        corresponding coarse element is (1,-1)-(-1,1) and the global
        coordinate on this element for s = 0.25 is X = (0.5,-0.5).
        So the return vector is the NumPy array: X = [[0.5],[-0.5]].

        If mask = None and coarsened = True, then all the elements
        are coarsened and the global coordinates are returned for
        these coarsened elements. In the square example, the first
        element with nodes (0,1) will be combined with the second
        element (1,2) to get (0,2). Similarly, the third and fourth
        are combined into (2,0). Thus the new coarse elements have
        the coordinates (-1,-1)-(1,1) and (1,1)-(-1,-1) respectively.
        Then global coordinates for s = 0.25 are (-0.5,-0.5) and
        (0.5,0.5). So the return vector is the NumPy array:
           X = [[-0.5,0.5],[-0.5,0.5]]
        In the case of odd number of elements, the last element is not
        coarsened.

        Parameters
        ----------
        s : float
            The local coordinate on the element, a scalar number
            between 0 and 1 (included).
        curve : Curve object
        mask : NumPy array, optional
            An optional one-dimensional array of integers storing
            the indices of the elements where the global coordinates
            for s will be computed, the integers should be in the
            range 0, ..., curve.size()-1.
        coarsened : bool, optional
            An optional argument denoting whether we compute the global
            coordinates for the current elements of the given curve
            (coarsened = False) or for the elements that result from
            coarsening the current elements (coarsened = True).

        Returns
        -------
        X : NumPy array
            A two-dimensional array of the global coordinates on all
            elements corresponding to the given local coordinate s,
            size of X will be 2 x curve.size().
        """
        n = curve.size()
        x = curve.coords()
        if n == 0:  return np.empty((2,0),dtype=float)

        if not coarsened: # Global coordinates for all of current elements
            if mask is not None:
                result = (1.0 - s) * x[:,mask] +  s * x[:,(mask+1)%n]
            else:
                result = (1.0 - s) * x
                result[:,0:n-1] += s * x[:,1:n]
                result[:,n-1] += s * x[:,0]

        else: # Global coordinates for elements that result from coarsening
            if mask is not None:
                result = (1.0 - s) * x[:,mask] + s * x[:,(mask+2)%n]
            else:
                result = (1.0 - s) * x[:,0:n:2]
                m = result.shape[1]
                result[:,0:m-1] += s * x[:,2:n:2]
                result[:,m-1] += s * x[:,0]

        return result


    def interpolate(self, u, s, mask=None, coarsened=False):
        """Interpolates a given data vector at s on elements of the curve.

        Interpolates a given vector u of data points linearly at the local
        coordinate s on selected elements of the curve.

        The data vector u can be scalar-valued, i.e. an array of size n
        (= curve.size()) with a single value for each node of the curve,
        or vector-valued with size d x n, or matrix-valued with size
        d1 x d2 x n.

        For example, if u is the NumPy array [1.0, 2.0, 3.0, 4.0] and
        s = 0.25, and if the interpolation is performed for all elements
        (mask = None, coarsened = False), then the interpolation is
        computed by
            [ (1.0 - 0.25) * 1.0 + 0.25 * 2.0,
              (1.0 - 0.25) * 2.0 + 0.25 * 3.0,
              (1.0 - 0.25) * 3.0 + 0.25 * 4.0,
              (1.0 - 0.25) * 4.0 + 0.25 * 1.0 ]
        and the return value is the NumPy array
            u_at_s = [ 1.25, 2.25, 3.25, 3.25 ]

        If in addition the mask array is specified as [0,2], then the
        interpolation is computed for 0th and 2nd elements only,
            u_at_s = [ 1.25, 3.25 ]

        If mask = [1] and coarsened = True, then the interpolation is
        for the 1st element when is coarsened (combined with the 2nd
        element; therefore the data comes from the 1st and the 3rd nodes
        and u_at_s = [ (1.0 - 0.25) * 2.0 + 0.25 * 4.0 ] = [ 2.5 ].

        If mask = None and coarsened = True, then all elements are
        coarsened and only the elements with nodes (0,2) and (2,0)
        remain with their data
            u_at_s = [ (1.0 - 0.25) * 1.0 + 0.25 * 3.0,
                       (1.0 - 0.25) * 3.0 + 0.25 * 1.0 ]
                   = [ 1.5, 2.5 ]
        In the case of odd number of elements, the last element is not
        coarsened.

        Parameters
        ----------
        u : NumPy array
            The data vector, a float array of size n or d x n
            or d1 x d2 x n, where n is the number of nodes on the
            curve.
        s : float
            The local coordinate on the element, a scalar number
            between 0 and 1 (included).
        mask : NumPy array, optional
            One-dimensional array of integers storing the indices
            of the elements where the global coordinates for s will
            be computed, the integers should be in the range:
            0, ..., curve.size()-1.
        coarsened : bool, optional
            An argument denoting whether we compute the global coordinates
            for the current elements of the given curve (coarsened = False)
            or for the elements that result from coarsening the current
            elements (coarsened = True).

        Returns
        -------
        u_at_s: The value of interpolation at s on all selected
            elements, a NumPy float array with same size as u.

        Raises
        ------
        ValueError
            Error if u.ndim > 3.
        """
        last_axis = u.ndim - 1
        n = u.shape[last_axis]

        if not coarsened:
            if mask is not None:
                mask1 = (mask + 1) % n
                result = (1.0 - s) * np.take( u, mask, axis=last_axis ) + \
                         s * np.take( u, mask1, axis=last_axis, mode='wrap' )
            else: # mask is None, interpolate for all elements
                result = (1.0 - s) * u
                if last_axis == 0:
                    result[0:n-1] += s * u[1:n]
                    result[n-1] += s * u[0]
                elif last_axis == 1:
                    result[:,0:n-1] += s * u[:,1:n]
                    result[:,n-1] += s * u[:,0]
                elif last_axis == 2:
                    result[:,:,0:n-1] += s * u[:,:,1:n]
                    result[:,:,n-1] += s * u[:,:,0]
                else:
                    raise ValueError('u.ndim cannot be larger than 3!')

        else: # Interpolation on the coarsened elements
            if mask is not None:
                mask2 = (mask + 2) % n
                result = (1.0 - s) * np.take( u, mask, axis=last_axis ) + \
                         s * np.take( u, mask2, axis=last_axis, mode='wrap' )
            else: # mask is None, interpolate for all coarsened elements
                m = np.ceil(n/2.0) # length of 0:n:2
                if last_axis == 0:
                    result = (1.0 - s) * u[0:n:2]
                    result[0:m-1] += s * u[2:n:2]
                    result[m-1] += s * u[0]
                elif last_axis == 1:
                    result = (1.0 - s) * u[:,0:n:2]
                    result[:,0:m-1] += s * u[:,2:n:2]
                    result[:,m-1] += s * u[:,0]
                elif last_axis == 2:
                    result = (1.0 - s) * u[:,:,0:n:2]
                    result[:,:,0:m-1] += s * u[:,:,2:n:2]
                    result[:,:,m-1] += s * u[:,:,0]
                else:
                    raise ValueError('u.ndim cannot be larger than 3!')

        return result


    def eval_basis_fct(self,quad):
        values = np.vstack( (quad.points(), 1.0-quad.points()) )
        return values

    def eval_basis_fct_products(self,quad):

        phi0 = quad.points()
        phi1 = 1.0 - phi0
        phi0_phi1 = phi0*phi1

        values = np.zeros((2,2,len(phi0)))
        values[0,0,:] = phi0**2
        values[0,1,:] = phi0_phi1
        values[1,0,:] = phi0_phi1
        values[1,1,:] = phi1**2
        return values

    def random_elements(self, grid_size, diameter):
        min_side_factor = 0.5
        max_side_factor = 1.5

        n = n_elem = min( grid_size ) - 1
        boundary_size = (diameter * grid_size[0]/n, diameter * grid_size[1]/n)

        elem_length = diameter / n

        x0 = elem_length * max_side_factor + \
             (boundary_size[0] - 2*elem_length*max_side_factor) * rand(n_elem)
        y0 = elem_length * max_side_factor + \
             (boundary_size[1] - 2*elem_length*max_side_factor) * rand(n_elem)

        min_side = min_side_factor * elem_length
        max_side = max_side_factor * elem_length
        lengths = min_side + (max_side - min_side) * rand(n_elem)

        angles = 2.0*np.pi * rand(n_elem)

        x1,y1 = x0 + lengths * np.cos(angles), y0 + lengths * np.sin(angles)

        return _RandomElementCollection( (x0,y0,x1,y1), lengths )


class _RandomElementCollection(object):
    """Encapsulation the random elements computed by ReferenceElement.

    The only purpose of this class to make random element error computations
    of integration.adapt_integrate_tolerance possible also for Curve and
    CurveHierarchy objects.
    """

    """
    Returning a Curve object from curve_fem.ReferenceElement.random_elements()
    function would not work, because the elements of a Curve are connected
    each other, so a Curve cannot store random elements scattered across
    a planar region.
    """
    def __init__(self,coords,el_sizes=None):
        self._coords = coords
        self._el_sizes = el_sizes
        self.timestamp = 0

    def coords(self):
        return self._coords

    def element_sizes(self):
        if self._el_sizes is None:
            x0,y0,x1,y1 = self._coords
            self._el_sizes = np.sqrt( (x0-x1)**2 + (y0-y1)**2 )
        return self._el_sizes

    def local_to_global(self, s, mask=None, coarsened=False):
        if coarsened or (mask is not None):
            raise ValueError("The arguments 'mask' and 'coarsened' are not used!")
        x0, y0, x1, y1 = self._coords
        result = np.empty( (2,len(x0)) )
        result[0,:] = (1.0 - s) * x0 + s * x1
        result[1,:] = (1.0 - s) * y0 + s * y1
        return result
