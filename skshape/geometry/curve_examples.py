import numpy as np
from ..numerics.fem import curve_fem as FEM
from .curve import Curve


## Ellipse: x = a cos(t), y = b sin(t)

## Limacon: r = b + a cos(t),  r = 1.5 + cos(t),  r = 0.5 + cos(t)
##     x = 0.5 a + b cos(t) + 0.5 cos(2t)
##     y = b sin(t) + 0.5 a sin(2t)

## Lemniscate of Bernoulli: r^2 = 2 a^2 cos(2t)  (2a dist btw foci)
##     x = (a sqrt(2) cos(t)) / (sin(t)^2 + 1)
##     y = (a sqrt(2) cos(t) sin(t)) / (sin(t)^2 + 1)

## Trifolium: r = a cos(t) (4 sin(t)^2 - 1)

## Figure eight curve: r^2 = a^2 cos(2t) sec(t)^4
##     x = a sin(t)
##     y = a sin(t) cos(t)


class GeneralizedCircle(Curve):

    def __init__(self, n, center=(0.0,0.0), radius=1.0, p=2):

        self._n = n
        self._center = center
        self._radius = radius
        self._p = p

        self._el_sizes = None
        self._normals = None
        self._normal_smoothness = None
        self._tangents = None
        self._curvature = None
        self._second_fund_form = None
        self.ref_element = FEM.ReferenceElement()

        self._bounding_box = None
        self._length = None
        self._area = None

        self._start = 0.0
        self._end = 2.0 * np.pi
        self._dt = (self._end - self._start) / n

        self._coords = self.exact_coords(0.0)

        super(GeneralizedCircle, self).__init__(self._coords, None)

    def exact_coords(self,s,param=None):
        if (s < 0.) or (s > 1.):
            raise ValueError("The parameter s has to be in [0,1].")

        p = self._p
        t = self._dt * (s + np.arange( 0.0, self._n ))

        coords = np.empty((2,self._n))
        x = coords[0,:]
        y = coords[1,:]

        c = np.cos(t)
        s = np.sin(t)

        nrm = (c**p + s**p)**(1.0/p)

        x[:] = self._center[0] + self._radius * c / nrm
        y[:] = self._center[1] + self._radius * s / nrm

        return coords

    def exact_normals(self,s,param=None):
        if (self._p != 2):
            raise Exception("Exact normals not available for p!=2.")
        if (s < 0.) or (s > 1.):
            raise ValueError("The parameter s has to be in [0,1].")

        p = self._p
        t = self._dt * (s + np.arange( 0.0, self._n ))
        c = np.cos(t)
        s = np.sin(t)

        nrm    = (c**p + s**p)**(1.0/p)
        d_nrm  = nrm**(1.0-p) * (c*s**(p-1.0) - s*c**(p-1.0))

        dx   =  s/nrm + c*d_nrm/nrm**2
        dy   = -c/nrm + s*d_nrm/nrm**2

        normals = np.vstack(( -dy, dx ))

        return normals

    def exact_tangents(self,s,param=None):
        if (self._p != 2):
            raise Exception("Exact tangents not available for p!=2.")
        if (s < 0.) or (s > 1.):
            raise ValueError("The parameter s has to be in [0,1].")

        p = self._p
        t = self._dt * (s + np.arange( 0.0, self._n ))
        c = np.cos(t)
        s = np.sin(t)

        nrm    = (c**p + s**p)**(1.0/p)
        d_nrm  = nrm**(1.0-p) * (c*s**(p-1.0) - s*c**(p-1.0))

        dx   =  s/nrm + c*d_nrm/nrm**2
        dy   = -c/nrm + s*d_nrm/nrm**2

        tangents = np.vstack(( -dx, -dy ))

        return tangents

    def exact_curvature(self,s,param=None):
        if (self._p != 2):
            raise Exception("Exact curvature not available for p!=2.")
        if (s < 0.) or (s > 1.):
            raise ValueError("The parameter s has to be in [0,1].")

        return np.ones(self._n) / self._radius

    def exact_second_fund_form(self,s,param=None):
        T = self.exact_tangents(s,param)
        K = self.exact_curvature(s,param)
        II = np.empty((2,2,len(K)))
        for i,j in [(0,0),(0,1),(1,0),(1,1)]:
            II[i,j] = K * T[i] * T[j]
        return II


class GeneralizedEllipse(Curve):

    def __init__(self, n, center=(0.0,0.0), axes=(1.0,1.0), p=2):

        self._n = n
        self._center = center
        self._axes = axes
        self._p = p

        self._el_sizes = None
        self._normals = None
        self._normal_smoothness = None
        self._tangents = None
        self._curvature = None
        self._second_fund_form = None
        self.ref_element = FEM.ReferenceElement()

        self._bounding_box = None
        self._length = None
        self._area = None

        self._start = 0.0
        self._end = 2.0 * np.pi
        self._dt = (self._end - self._start) / n

        self._coords = self.exact_coords(0.0)

        super(GeneralizedEllipse, self).__init__(self._coords, None)

    def exact_coords(self,s,param=None):
        if (s < 0.) or (s > 1.):
            raise ValueError("The parameter s has to be in [0,1].")

        p = self._p
        t = self._dt * (s + np.arange( 0.0, self._n ))

        coords = np.empty((2,self._n))
        x = coords[0,:]
        y = coords[1,:]

        c = np.cos(t)
        s = np.sin(t)

        nrm = (c**p + s**p)**(1.0/p)

        x[:] = self._center[0] + self._axes[0] * c / nrm
        y[:] = self._center[1] + self._axes[1] * s / nrm

        return coords


class FlowerCurve(Curve):

    def __init__(self, n, center=(0.0,0.0), radius=1.0,
                 amplitude=0.25, frequency=4.0):

        self._n = n
        self._center = center
        self._radius = radius
        self._amplitude = amplitude
        self._frequency = frequency

        self._el_sizes = None
        self._normals = None
        self._normal_smoothness = None
        self._tangents = None
        self._curvature = None
        self._second_fund_form = None
        self.ref_element = FEM.ReferenceElement()

        self._bounding_box = None
        self._length = None
        self._area = None

        self._start = 0.0
        self._end = 2.0 * np.pi
        self._dt = (self._end - self._start) / n

        self._coords = self.exact_coords(0.0)

        super(FlowerCurve, self).__init__(self._coords, None)


    def exact_coords(self, s, param=None):
        if (s < 0.) or (s > 1.):
            raise ValueError("The parameter s has to be in [0,1].")

        a = self._amplitude
        w = self._frequency

        coords = np.zeros((2,self._n))

        x = coords[0,:]
        y = coords[1,:]

        t = self._dt * (s + np.arange( 0.0, self._n ))
        r = self._radius * (1.0 + a*np.cos(w*t))

        x[:] = self._center[0] + r*np.cos(t)
        y[:] = self._center[1] + r*np.sin(t)

        return coords

    def exact_normals(self, s, param=None):
        if (s < 0.) or (s > 1.):
            raise ValueError("The parameter s has to be in [0,1].")

        a = self._amplitude
        w = self._frequency
        radius = self._radius
        t = self._dt * (s + np.arange( 0.0, self._n ))

        r  =  radius * (1.0 + a*np.cos(w*t))
        dr = -radius * a*w * np.sin(w*t)
        dx = dr * np.cos(t) - r * np.sin(t)
        dy = dr * np.sin(t) + r * np.cos(t)

        normal_size = np.sqrt(dx**2 + dy**2)
        normal = np.vstack(( dy/normal_size, -dx/normal_size ))

        return normal

    def exact_tangents(self, s, param=None):
        if (s < 0.) or (s > 1.):
            raise ValueError("The parameter s has to be in [0,1].")

        a = self._amplitude
        w = self._frequency
        radius = self._radius
        t = self._dt * (s + np.arange( 0.0, self._n ))

        r  =  radius * (1.0 + a*np.cos(w*t))
        dr = -radius * a*w * np.sin(w*t)
        dx = dr * np.cos(t) - r * np.sin(t)
        dy = dr * np.sin(t) + r * np.cos(t)

        tangent_size = np.sqrt(dx**2 + dy**2)
        tangents = np.vstack(( dx/tangent_size, dy/tangent_size ))

        return tangents

    def exact_curvature(self, s, param=None):
        if (s < 0.) or (s > 1.):
            raise ValueError("The parameter s has to be in [0,1].")

        a = self._amplitude
        w = self._frequency
        radius = self._radius
        t = self._dt * (s + np.arange( 0.0, self._n ))

        r   =  radius * (1.0 + a*np.cos(w*t))
        dr  = -radius * a*w * np.sin(w*t)
        ddr = -radius * a*w**2 * np.cos(w*t)

        dx  = dr * np.cos(t) - r * np.sin(t)
        dy  = dr * np.sin(t) + r * np.cos(t)

        ddx = (ddr-r) * np.cos(t) - 2 * dr * np.sin(t)
        ddy = (ddr-r) * np.sin(t) + 2 * dr * np.cos(t)

        normal_size = np.sqrt(dx**2 + dy**2)

        K = (dx*ddy - dy*ddx) / normal_size**3

        return K

    def exact_second_fund_form(self,s,param=None):
        T = self.exact_tangents(s,param)
        K = self.exact_curvature(s,param)
        II = np.empty((2,2,len(K)))
        for i,j in [(0,0),(0,1),(1,0),(1,1)]:
            II[i,j] = K * T[i] * T[j]
        return II


class Limacon(Curve):

    def __init__(self, n, center=(0.0,0.0), radius=1.5):
        self._n = n
        self._center = center
        self._radius = radius

        self._el_sizes = None
        self._normals = None
        self._normal_smoothness = None
        self._tangents = None
        self._curvature = None
        self._second_fund_form = None
        self.ref_element = FEM.ReferenceElement()

        self._bounding_box = None
        self._length = None
        self._area = None

        self._start = 0.0
        self._end = 2.0 * np.pi
        self._dt = (self._end - self._start) / n

        self._coords = self.exact_coords(0.0)

        super(Limacon, self).__init__(self._coords, None)

    def exact_coords(self, s, param=None):
        if (s < 0.) or (s > 1.):
            raise ValueError("The parameter s has to be in [0,1].")

        coords = np.zeros((2,self._n))

        x = coords[0,:]
        y = coords[1,:]

        t = self._dt * (s + np.arange( 0.0, self._n ))
        r = self._radius + np.cos(t)

        x[:] = self._center[0] + r*np.cos(t)
        y[:] = self._center[1] + r*np.sin(t)

        return coords

    def exact_normals(self, s, param=None):
        if (s < 0.) or (s > 1.):
            raise ValueError("The parameter s has to be in [0,1].")

        t = self._dt * (s + np.arange( 0.0, self._n ))

        r  = self._radius + np.cos(t)
        dr = -np.sin(t)
        dx = dr * np.cos(t) - r * np.sin(t)
        dy = dr * np.sin(t) + r * np.cos(t)

        normal_size = np.sqrt(dx**2 + dy**2)
        normal = np.vstack(( dy/normal_size, -dx/normal_size ))

        return normal

    def exact_tangents(self, s, param=None):
        if (s < 0.) or (s > 1.):
            raise ValueError("The parameter s has to be in [0,1].")

        t = self._dt * (s + np.arange( 0.0, self._n ))

        r  = self._radius + np.cos(t)
        dr = -np.sin(t)
        dx = dr * np.cos(t) - r * np.sin(t)
        dy = dr * np.sin(t) + r * np.cos(t)

        tangent_size = np.sqrt(dx**2 + dy**2)
        tangent = np.vstack(( dx/tangent_size, dy/tangent_size ))

        return tangent

    def exact_curvature(self, s, param=None):
        if (s < 0.) or (s > 1.):
            raise ValueError("The parameter s has to be in [0,1].")

        t = self._dt * (s + np.arange( 0.0, self._n ))

        r   = self._radius + np.cos(t)
        dr  = -np.sin(t)
        ddr = -np.cos(t)

        dx  = dr * np.cos(t) - r * np.sin(t)
        dy  = dr * np.sin(t) + r * np.cos(t)

        ddx = (ddr-r) * np.cos(t) - 2 * dr * np.sin(t)
        ddy = (ddr-r) * np.sin(t) + 2 * dr * np.cos(t)

        normal_size = np.sqrt(dx**2 + dy**2)

        K = (dx*ddy - dy*ddx) / normal_size**3

        return K

    def exact_second_fund_form(self,s,param=None):
        T = self.exact_tangents(s,param)
        K = self.exact_curvature(s,param)
        II = np.empty((2,2,len(K)))
        for i,j in [(0,0),(0,1),(1,0),(1,1)]:
            II[i,j] = K * T[i] * T[j]
        return II


def curve_example(curve_id, n=8, center=[0.0,0.0], radius=1.0, parameters={}):
    """Creates a discretized example of the specified curve.

    Creates the specified example curve discretized n nodes. The curve can be
    a square, rectangle, circle, ellipse, limacon, flower.

    Parameters
    ----------
    curve_id : str
        One of 'square', 'rectangle', 'circle', 'ellipse', 'limacon', 'flower'
    n : int, optional
        The number of points used to discretize the curve example.
    center : array_like, optional
        The center of the curve. Default value is [0.0,0.0].
    radius : float, tuple, optional
        Radius or (height, width) of the curve. Default value is 1.0.
    parameters : dict, optional
        A dictionary of parameters specific to the curve type.

    Returns
    -------
    curve: object, Curve, Limacon, FlowerCurve, GeneralizedCircle, GeneralizedEllipse
    """

    if n < 3:
        raise ValueError("Number of points on the curve cannot be less than three!")

    if curve_id == 'square':

        n = int(np.ceil(n/4.))
        h = radius / n

        x1 = center[0] + radius*np.ones(n)
        x2 = center[0] + np.arange(radius,0.,-h)
        x3 = center[0] + np.zeros(n)
        x4 = center[0] + np.arange(0.,radius,h)

        y1 = center[1] + np.arange(0.,radius,h)
        y2 = center[1] + radius*np.ones(n)
        y3 = center[1] + np.arange(radius,0.,-h)
        y4 = center[1] + np.zeros(n)

        coords = np.vstack( (np.concatenate( (x1,x2,x3,x4) ),
                             np.concatenate( (y1,y2,y3,y4) )) )
        return Curve( coords, adaptivity_parameters=None )

    elif curve_id == 'rectangle':

        rx,ry = radius
        nx,ny = int(np.ceil((n*rx)/(2*rx+2*ry))), int(np.ceil((n*ry)/(2*rx+2*ry)))
        hx,hy = rx/nx, ry/ny

        x1 = center[0] + rx*np.ones(ny)
        x2 = center[0] + np.arange(rx,0.,-hx)
        x3 = center[0] + np.zeros(ny)
        x4 = center[0] + np.arange(0.,rx,hx)

        y1 = center[1] + np.arange(0.,ry,hy)
        y2 = center[1] + ry*np.ones(nx)
        y3 = center[1] + np.arange(ry,0.,-hy)
        y4 = center[1] + np.zeros(nx)

        coords = np.vstack(( np.hstack( (x1,x2,x3,x4) ),
                             np.hstack( (y1,y2,y3,y4) ) ))
        return Curve( coords, adaptivity_parameters=None )

    elif curve_id == 'circle':
        return GeneralizedCircle( n, center, radius )

    elif curve_id == 'ellipse':
        return GeneralizedEllipse( n, center, radius )

    elif curve_id == 'limacon':
        return Limacon( n, center, radius )

    elif curve_id == 'flower':
        try:
            amplitude = parameters['amplitude']
            frequency = parameters['frequency']
        except KeyError:
            amplitude = 0.25 * radius
            frequency = 4.0
        return FlowerCurve(n, center, radius, amplitude, frequency )

    else:
        raise ValueError("curve_id should be one 'square', 'rectangle', 'circle', 'ellipse', 'limacon', 'flower'!")
