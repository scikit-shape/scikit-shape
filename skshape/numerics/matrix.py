import numpy as np
from numpy.linalg import solve as direct_solve
from scipy.sparse.construct import spdiags, bmat
from scipy.sparse.linalg import svds, LinearOperator
from scipy.linalg.lapack import dgtsv
from scipy.linalg import solve_banded


class Matrix(LinearOperator):

    def size(self):
        raise NotImplementedError("This function has not been implemented.")

    def copy(self):
        raise NotImplementedError("This function has not been implemented.")

    def __imul__(self,scalar):
        raise NotImplementedError("This function has not been implemented.")

    def __mul__(self,v):
        raise NotImplementedError("This function has not been implemented.")

    def __rmul__(self,v):
        if np.isscalar(v):
            return self.__mul__(v)
        else:
            raise ValueError("Multiply from right of Matrix works only for scalars!")

    def __add__(self,M):
        raise NotImplementedError("This function has not been implemented.")

    def __sub__(self,M):
        raise NotImplementedError("This function has not been implemented.")

    def norm(self):
        raise NotImplementedError("This function has not been implemented.")

    def _matvec(self,v):
        return self.__mul__(v)

    def matvec(self,v):
        return self.__mul__(v)

    def rmatvec(self,v):
        raise NotImplementedError("This function has not been implemented.")

    def solve(self,d):
        raise NotImplementedError("This function has not been implemented.")


class TridiagMatrix(Matrix):
    """
    The TridiagMatrix class to store tridiagonal matrices of the form

    ::

       ( b(0) c(0)             a(0) )
       ( a(1) b(1) c(1)             )
       (      a(2)  .  .            )
       (            .  .  .         )
       (               .  .  c(n-1) )
       ( c(n)            a(n) b(n)  )

    It is used for implementing efficient linear algebra for
    tridiagonal matrices.

    The following methods are available:

    Methods
    -------
    set_diag(d,band_no)
    get_diag(d,band_no=None)
    rmatvec(v)
    factorize()
    solve(d,L=None,M=None)
    to_sparse(format='lil')
    norm(norm_type='Frobenius')
    copy()
    size()

    """

    def __init__(self,a,b,c,copy=True):
        if not copy: # Don't copy, refer to the original input arrays.
            self._a = a
            self._b = b
            self._c = c
        else: # Store copies of the input arrays.
            self._b = b.copy()
            if len(a) == len(b):
                self._a = a.copy()
            else:
                self._a = np.empty(len(b))
                n = len(a)
                self._a[-n:] = a
                self._a[0:-n] = 0.0
            if len(c) == len(b):
                self._c = c.copy()
            else:
                self._c = np.empty(len(b))
                n = len(c)
                self._c[0:n] = c
                self._c[n:] = 0.0
        n = len(b)
        self.shape = (n,n)
        self._factorization = None

    def __repr__(self):
        repr_str = "lower diagonal: " + str(self._a) + "\n" \
                   "mid diagonal  : " + str(self._b) + "\n" \
                   "upper diagonal: " + str(self._c) + "\n"
        return repr_str

    def __str__(self):
        repr_str = "Upper " + str(self._a) + \
                   ", mid " + str(self._b) + ", lower " + str(self._c)
        return repr_str

    def size(self):
        'Tridiag.size() returns the size of the tridiagonal matrix'
        return len(self._b)

    def copy(self):
        return TridiagMatrix( self._a, self._b, self._c )

    def set_diag(self,d,band_no):
        """Set the diagonal band to given d array.

        Tridiag.setDiag( d, band_no ) sets the band specified with band_no to
        the values given in d. The argument band_no should be one of -1,0,1.

        """
        if not (band_no in [-1,0,1]):
            raise ValueError('Argument band_no should be one of -1,0,1')
        elif len(d) != len(self._b):
            raise ValueError('Size of input vector d should be %d' % len(self._b))
        elif band_no == -1:
            self._a[:] = d[:]
        elif band_no == 0:
            self._b[:] = d[:]
        elif band_no == 1:
            self._c[:] = d[:]

    def get_diag(self,band_no=None):
        """Returns the selected diagonal of the tridiag matrix

        Tridiag.getDiag( d, band_no ) returns the values in the band
        specified with band_no, which should be one of -1,0,1.
        The default value of band_no is None. In that case, all
        three diagonals are returned.

        """
        if band_no is None:
            return (self._a, self._b, self._c)
        if not (band_no in [-1,0,1]):
            raise ValueError('band_no should be None or one of -1,0,1')
        elif band_no == -1:
            return self._a
        elif band_no == 0:
            return self._b
        elif band_no == 1:
            return self._c

    def __imul__(self,scalar):
        self._a *= scalar
        self._b *= scalar
        self._c *= scalar
        return self

    def __mul__(self,v):
        """Matrix-vector product

        TridiagMatrix.__mul__( v ) returns the matrix-vector product with
        the vector v.

        """
        a = self._a;   b = self._b;   c = self._c

        try: # try multiplication as if v is a vector
            N = len(v) # might raise an exception if v is a scalar
            if N != len(b):
                raise ValueError('The matrix and the vector should be of the same dimension!')
            u = b * v
            u[0] += a[0] * v[N-1]
            u[1:N] += a[1:N] * v[0:N-1]
            u[N-1] += c[N-1] * v[0]
            u[0:N-1] += c[0:N-1] * v[1:N]
            return u

        except TypeError: # v might be a scalar, or we get another exception
            new_a = v * a
            new_b = v * b
            new_c = v * c
            return TridiagMatrix( new_a, new_b, new_c, copy=False )

    def rmatvec(self,v):
        a = self._a;   b = self._b;   c = self._c

        try: # try multiplication as if v is a vector
            N = len(v) # might raise an exception if v is a scalar
            if N != len(self._b):
                raise ValueError('The matrix and the vector should be of the same dimension!')
            u = b * v
            u[0] += c[N-1] * v[N-1]
            u[1:N] += c[0:N-1] * v[0:N-1]
            u[N-1] += a[0] * v[0]
            u[0:N-1] += a[1:N] * v[1:N]
            return u

        except TypeError: # v might be a scalar, or we get another exception
            new_a = v * a
            new_b = v * b
            new_c = v * c
            return TridiagMatrix( new_a, new_b, new_c, copy=False )

    def __add__(self,M):
        a = self._a + M.get_diag(-1)
        b = self._b + M.get_diag(0)
        c = self._c + M.get_diag(1)
        return TridiagMatrix(a,b,c)

    def __sub__(self,M):
        a = self._a - M.get_diag(-1)
        b = self._b - M.get_diag(0)
        c = self._c - M.get_diag(1)
        return TridiagMatrix(a,b,c)

    def norm(self,norm_type='Frobenius'):
        """Calculates the norm of the matrix.

        The norm_type should be one 0, 2, inf or 'Frobenius' (the default value).

        """
        if norm_type == 'Frobenius':
            total = np.sum(self._a**2)
            total += np.sum(self._b**2)
            total += np.sum(self._c**2)
            return np.sqrt(total)

        elif norm_type in [np.inf,'inf','Inf']:
            row_sums = np.abs(self._a) + abs(self._b) + abs(self._c)
            return np.max(row_sums)

        elif norm_type in [0,'0']:
            a = self._a
            c = self._c
            N = len(a)
            col_sums = np.abs(self._b)
            col_sums[0:N-1] += np.abs(a[1:N])
            col_sums[N-1] += np.abs(a[0])
            col_sums[1:N] += np.abs(c[0:N-1])
            col_sums[0] += np.abs(c[N-1])
            return np.max(col_sums)

        elif norm_type in [2,'2']:
            return svds(self, k=1, which='LM', return_singular_vectors=False)

        else:
            raise ValueError('Cannot compute the specified norm')

    def factorize(self):
        if self._factorization is not None:
            return self._factorization

        a = self._a
        b = self._b
        c = self._c
        n = self.size()

        periodic_tridiag_system = (a[0] != 0) or (c[n-1] != 0)

        if periodic_tridiag_system:
            b[0]   -= a[0]
            b[n-1] -= c[n-1]

        L,U = tridiag_LU( a[1:], b, c[:n-1] )
        self._factorization = (L,U)

        if periodic_tridiag_system:
            b[0]   += a[0]
            b[n-1] += c[n-1]

        return (L,U)

    def solve(self, d, L=None, M=None):
        try:
            return fast_tridiag_solve(self,d)
        except Exception as e:
            print('Could not use fast triadiagonal solver! Reverting to slow Python code. (Error = %s)' % e) #sys.exc_info()[0])

        if L is None or M is None: # Factorization not given as arguments.

            if self._factorization is not None: # Factorization cached.
                L,M = self._factorization

            else: # Factorization not available. Sove from scratch.
                return tridiag_solve(self,d)

        # At this point, LU factorization (L,M) is available and will be used.
        a = self._a
        b = self._b
        c = self._c
        n = len(b)

        if len(d) != n:
            raise Exception('Matrix A and right hand side vector d should be the same size')

        periodic_tridiag_system = (a[0] != 0) or (c[n-1] != 0)

        if periodic_tridiag_system:
            b[0]   -= a[0]
            b[n-1] -= c[n-1]

        y = tridiag_solve_from_LU( L, M, c[:n-1], d )

        if periodic_tridiag_system:
            u = np.zeros(n);  v = np.zeros(n)
            u[0] = a[0];   u[n-1] = c[n-1]
            v[0] = 1;      v[n-1] = 1

            z = tridiag_solve_from_LU( L, M, c[:n-1], u )

            x = y - z * np.dot(v,y) / (1 + np.dot(v,z))

            b[0]   += a[0]
            b[n-1] += c[n-1]
        else:
            x = y

        return x

    def to_sparse(self, format='lil'):
        a = self._a
        b = self._b
        c = self._c
        n = len(b)

        a0 = a[0]
        cn = c[n-1]
        aa = np.hstack(( a[1:n], a0 ))
        cc = np.hstack(( cn, c[0:n-1] ))

        data = np.array([aa,b,cc])
        diags = np.array([-1,0,1])

        A = spdiags( data, diags, n, n, format=format )
        A[0,n-1] = a0
        A[n-1,0] = cn

        return A


def fast_tridiag_solve(A,d):

    a = A.get_diag(-1)
    b = A.get_diag(0)
    c = A.get_diag(1)
    n = A.size()
    a0 = a[0]
    cn = c[n-1]

    a = a[1:n].copy()
    b = b.copy()
    c = c[0:n-1].copy()
    y = d.copy()

    periodic_tridiag_system = (a0 != 0) or (cn != 0)

    if periodic_tridiag_system:
        b[0]   -= a0
        b[n-1] -= cn
        a2 = a.copy()
        b2 = b.copy()
        c2 = c.copy()

    vec_size = c_int(n)
    n_rhs    = c_int(1)
    info_out = c_int(0)

    try: # First try the Lapack tridiagonal solver.
        _,_,_,y,info = dgtsv(a,b,c,d)

    except Exception as e:
        # If Lapack tridiag solver doesn't work, try SciPy's solve_banded.
        print("Call to Lapack failed: error = " + str(e))
        print("Unable to use Lapack DGTSV as tridiagonal solver, using scipy.linalg.solve_banded instead.")
        A = np.empty((3,n))
        A[1,0:n] = b
        A[0,1:n] = c;   A[0,0] = 0.0
        A[2,0:n-1] = a; A[2,n-1] = 0.0
        y = solve_banded( (1,1), A, d )

    if not periodic_tridiag_system:
        x = y

    else: # it is a periodic tridiag system
        u = np.zeros(n);  v = np.zeros(n) # also use u instead of z
        u[0] = a0;   u[n-1] = cn
        v[0] = 1;    v[n-1] = 1

        try: # First try Lapack tridiagonal solver.
            _,_,_,u,info = dgtsv(a2,b2,c2,u)
        except Exception as e:
            # If Lapack tridiag solver doesn't work, try SciPy's solve_banded.
            print("Call to Lapack failed: error = " + str(e))
            print("Unable to use Lapack DGTSV as tridiagonal solver, using scipy.linalg.solve_banded instead.")
            A = np.empty((3,n))
            A[1,0:n] = b2
            A[0,1:n] = c2;   A[0,0] = 0.0
            A[2,0:n-1] = a2; A[2,n-1] = 0.0
            u = solve_banded( (1,1), A, u )

        x = y - u * np.dot(v,y) / (1 + np.dot(v,u))
        # Used u instead of z, compare with the slow tridiag solver.

    return x


def tridiag_solve(A,d):
    """Efficient solve of a tridiagonal linear system.

    tridiag_solve( A, d ) takes a tridiagonal matrix A and right hand side
    vector d and returns the solution x of the linear system   A x = d.

    """
    a = A.get_diag(-1)
    b = A.get_diag(0)
    c = A.get_diag(1)
    n = A.size()

    if len(d) != n:
        raise Exception('Matrix A and right hand side vector d should be the same size')

    periodic_tridiag_system = (a[0] != 0) or (c[n-1] != 0)

    if periodic_tridiag_system:
        b[0]   -= a[0]
        b[n-1] -= c[n-1]

    l,m = tridiag_LU( a[1:], b, c[:n-1] )

    y = tridiag_solve_from_LU( l, m, c[:n-1], d )

    if periodic_tridiag_system:
        u = np.zeros(n);  v = np.zeros(n)
        u[0] = a[0];   u[n-1] = c[n-1]
        v[0] = 1;      v[n-1] = 1

        z = tridiag_solve_from_LU( l, m, c[:n-1], u )

        x = y - z * np.dot(v,y) / (1 + np.dot(v,z))

        b[0]   += a[0]
        b[n-1] += c[n-1]
    else:
        x = y

    return x

def tridiag_LU(a,b,c):
    """Computes the LU decomposition of the triadiagonal matrix.

    tridiag_LU( a, b, c ) computes the LU decomposition of the tridiagonal matrix

    ::

         ( b(0) c(0)             a(0) )
         ( a(1) b(1) c(1)             )
     A = (      a(2)  .  .            )
         (            .  .  .         )
         (               .  .  c(n-1) )
         ( c(n)            a(n) b(n)  )

    from the given three bands a,b,c and returns two vectors l,m that give
    the LU decomposition

       ``A = L U``

    where

       ``L = Id + diag(-1,l),   U = diag(0,m) + diag(1,r)``

    """
    N = len(b);   m = np.empty(N);   l = np.empty(N);

    m[0] = b[0]
    for i in range(N-1):
        l[i] = a[i] / m[i]
        m[i+1] = b[i+1] - l[i]*c[i]

    return (l,m)

def tridiag_solve_from_LU(l,m,c,d):
    """Computes the solution of the linear system from LU decomposition.

    tridiag_solve_from_LU( l, m, c, d ) takes the LU decomposition l,m
    (which should be calculated by tridiagLU), the third band c
    (band_no = 1), and right hand side d of the tridiagonal linear
    system and returns its solution. See documentation of tridiagLU
    for more informations on the arguments.

    """
    N = len(d)

    x = np.empty(N);  y = np.empty(N);

    y[0] = d[0]
    for i in range(1,N):
        y[i] = d[i] - l[i-1]*y[i-1]

    x[N-1] = y[N-1] / m[N-1]
    for i in range(N-2,-1,-1):
        x[i] = (y[i] - c[i]*x[i+1]) / m[i]

    return x


class BlockDiagMatrix(Matrix):

    def __init__(self,matrix_list=None):
        self._submatrices = matrix_list
        self._factorization = None

        if matrix_list is None:
            self._n = 0
        else: # Sum of sizes of all submatrices given in matrix_list
            self._n = sum(( m.size() for m in matrix_list ))
        self.shape = (self._n, self._n)

    def size(self):
        return self._n

    def copy(self):
        if self._submatrices is None:
            return BlockDiagMatrix()
        else:
            submatrices_copy = [ M.copy() for M in self._submatrices ]
            return BlockDiagMatrix( submatrices_copy )

    def submatrices(self):
        return self._submatrices

    def __imul__(self,scalar):
        for M in self._submatrices:
            M *= scalar
        return self

    def __mul__(self,v):
        try: # Try multiplication assuming v is a vector (not a scalar)
            n = len(v) # Might raise an exception if v is a scalar
            if n != self._n:
                raise ValueError('The matrix and the vector should be of the same dimension!')
            u = np.empty(n)
            start = 0
            for M in self._submatrices:
                end = start + M.size()
                u[start:end] = M * v[start:end]
                start = end
            return u

        except TypeError: # v might be a scalar or we get another exception
            new_matrix_list = [ M.copy()*v for M in self._submatrices ]
            return BlockDiagMatrix( new_matrix_list )

    def rmatvec(self,v):
        try: # Try multiplication as if v is a vector
            n = len(v) # Might raise an exception if v is a scalar
            if n != self._n:
                raise ValueError('The matrix and the vector should have the same dimension!')
            u = np.empty(n)
            start = 0
            for M in self._submatrices:
                end = start + M.size()
                u[start:end] = v[start:end] * M
                start = end
            return u

        except TypeError: # v might be a scalar, or we get another exception
            new_matrix_list = [ v*M.copy() for M in self._submatrices ]
            return BlockDiagMatrix( new_matrix_list )

    def __add__(self,M):
        new_matrix_list = [ A+B for A,B in zip(self._submatrices, M.submatrices()) ]
        return BlockDiagMatrix( new_matrix_list )

    def __sub__(self,M):
        new_matrix_list = [ A-B for A,B in zip(self._submatrices, M.submatrices()) ]
        return BlockDiagMatrix( new_matrix_list )

    def factorize(self):
        if self._factorization is not None:
            factorization = self._factorization
        else:
            factorization = [ M.factorize() for M in self._submatrices ]
            self._factorization = factorization
        return factorization

    def solve(self,v,factorization=None):
        u = np.empty( len(v) )

        if factorization is None:
            if self._factorization is not None:
                factorization = self._factorization
            else:
                start = 0
                for M in self._submatrices:
                    end = start + M.size()
                    u[start:end] = M.solve( v[start:end] )
                    start = end
                return u

        # At this point, factorization is available and will be used.
        start = 0
        for M, (L,U) in zip(self._submatrices, factorization):
            end = start + M.size()
            u[start:end] = M.solve( v[start:end], L, U )
            start = end
        return u

    def to_sparse(self, format='lil'):
        matrices = self._submatrices
        n = len( matrices )
        blocks = []
        for k in range(n):
            row = [None]*k + [ matrices[k].to_sparse(format) ] + [None]*(n-k-1)
            blocks.append( row )
        A = bmat( blocks )
        return A


class MatrixWithRankOneUpdates(Matrix):

    def __init__(self, core_matrix, update_vectors):
        self._A = core_matrix
        self._U = np.vstack(( u for u,v in update_vectors )).T
        self._V = np.vstack(( v for u,v in update_vectors ))
        self._C = None
        self._W = None
        self.shape = core_matrix.shape

    def size(self):
        return self._core_matrix.size()

    def copy(self):
        new_core_matrix = self._A.copy()
        return MatrixWithRankOneUpdates( new_core_matrix,
                                         self.update_vectors() )

    def core_matrix(self):
        return self._A

    def update_vectors(self,copy=True):
        U = self._U
        V = self._V
        if copy:
            return [ (U[:,k].copy(), V[k,:].copy()) for k in range(U.shape[1])]
        else:
            return [ (U[:,k], V[k,:]) for k in range(U.shape[1]) ]

    def __imul__(self,scalar):
        self._A *= scalar
        self._U *= scalar
        if self._W is not None: self._W *= scalar
        self._C = None
        return self

    def __mul__(self,x):
        A = self._A
        U = self._U
        V = self._V

        try: # Try multiplication assuming v is a vector (not a scalar)
            n = len(x) # Might raise an exception if v is a scalar
            if n != self._A.size():
                raise ValueError('The matrix and the vector should be of the same dimension!')
            y = (A * x) + np.dot( U, np.dot(V, x))
            return y

        except TypeError: # v might be a scalar or we get another exception
            new_A = A.copy() * x
            new_vecs = [ (U[:,k].copy(), V[k,:]*x) for k in range(U.shape[1]) ]
            return MatrixWithRankOneUpdates( new_A, new_vecs )

    def rmatvec(self,x):
        A = self._A
        U = self._U
        V = self._V

        try: # Try multiplication assuming v is a vector (not a scalar)
            n = len(x) # Might raise an exception if v is a scalar
            if n != self._A.size():
                raise ValueError('The matrix and the vector should be of the same dimension!')
            y = (x * A) + np.dot( np.dot(x, U), V)
            return y

        except TypeError: # v might be a scalar or we get another exception
            new_A = x * A.copy()
            new_vecs = [ (x*U[:,k], V[k,:].copy()) for k in range(U.shape[1]) ]
            return MatrixWithRankOneUpdates( new_A, new_vecs )

    def __add__(self,M):
        new_core_matrix = self._A + M.core_matrix()
        new_vector_list = list( self.update_vectors() )
        new_vector_list.extend( M.update_vectors() )
        return MatrixWithRankOneUpdates( new_core_matrix,
                                         new_vector_list )

    def __sub__(self,M):
        new_core_matrix = self._A - M.core_matrix()
        new_vector_list = list( self.update_vectors() )
        new_vector_list.extend(( (-u, v.copy())
                                 for u,v in M.update_vectors() ))
        return MatrixWithRankOneUpdates( new_core_matrix,
                                         new_vector_list )

    def solve(self,b):
        A = self._A
        V = self._V
        W = self._W
        C = self._C

        n_vectors, vec_size = V.shape

        if W is None:
            self._W = W = np.empty( (vec_size, n_vectors) )
            for k in range(n_vectors):
                W[:,k] = A.solve( self._U[:,k] )

        if C is None:
            self._C = C = np.eye( n_vectors ) + np.dot(V,W)

        y = A.solve(b)
        z = direct_solve( C, np.dot(V,y) ) # C is supposed to be small
        y -= np.dot(W,z)

        return y
