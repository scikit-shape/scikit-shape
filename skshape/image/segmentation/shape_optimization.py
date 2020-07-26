"""Optimization of curve/surface dependent shape energies.

This module contains shape optimization functions to compute optimal shapes
with respect to a specified shape energy. Typically, an initial curve or
curve family is specified, and the curves are iteratively moved/deformed
in a manner gradually decreasing their shape energy. This process concludes
with optimal curves with minimum shape energy.

"""

from __future__ import print_function
from copy import deepcopy
from numpy import inf, isinf, isscalar, empty, array, zeros, hstack, vstack, abs, sqrt, dot, mean, percentile, count_nonzero
from numpy.linalg import norm
from scipy.sparse.linalg import spsolve, ArpackNoConvergence
from scipy.sparse import bmat as sp_bmat


default_optimization_parameters = {
    'minimum iterations': 0,
    'maximum iterations': 1000,
    'absolute tolerance': 1e-3,
    'relative tolerance': 1e-3,
    'descent direction': ('L2',True),
    'step size parameters': None,
    'verbose': False
    }

default_step_size_parameters = {
    'initial step size': 0.01,
    'minimum step size': 1e-4,
    'maximum step size': 1.0,
    'decrease factor': 0.25,
    'increase factor': 2.0,
    'alpha': 1e-4,
    'safe step tolerance': 0.3,
#    'use error offset': False,
    'restart with maximum': False,
    'monotone line search': True,
    'nonmonotone strategy': ('smoothed maximum',5),
    'eta': 0.2              # or 'recent maximum' or 'recent average'
    }


_ENERGY_EVALUATIONS = 0
_HISTORY = {}
_VERBOSE = False


def _init_main_optim_parameters(parameters):
    global _VERBOSE

    _VERBOSE =  parameters.get('verbose',False)

    velocity_params = parameters['descent direction']

    min_iters = parameters.get('minimum iterations',0)
    max_iters = parameters.get('maximum iterations',200)

    step_params = deepcopy( parameters.get('step size parameters') )
    if step_params is None:
        step_params = deepcopy( default_step_size_parameters )

    min_dt = step_params['minimum step size']
    max_dt = step_params['maximum step size']
    old_dt = step_params['initial step size']
    dt = old_dt

    velocity_params = parameters['descent direction']

    restart_with_max = False if (velocity_params[0] in ['H1','L2']) else True
    step_params.setdefault( 'restart with maximum', restart_with_max )
    step_params.setdefault( 'maximum step size', 1.0 )

    rho_up   = step_params['increase factor']
    rho_down = step_params['decrease factor']

    if not step_params.has_key('monotone line search'):
        step_params['monotone line search'] = True

    if step_params['monotone line search']:
        Q = eta = None  # Q and eta are not needed (for monotone line search)
    else: # nonmonotone line search, so set Q and eta
        Q = 1.0;  eta = step_params.get( 'eta', 0.2 )

    return ( velocity_params, (min_iters, max_iters), step_params,
              (dt, old_dt, min_dt, max_dt, rho_down, rho_up, eta, Q)  )


def _update_optim_history(J, dt, norm_G, shape_deriv, surface, V_vec,
                          keep_history=False, reset=False):
    global _ENERGY_EVALUATIONS
    global _HISTORY

    if reset: # Resetting/restarting optimization history

        _HISTORY = {'energy': [ J ],
                    'energy evaluations': [ _ENERGY_EVALUATIONS ],
                    'shape gradient': [ norm_G ],
                    'shape derivative': [],
                    'step': [ dt ] }
        if keep_history:
            _HISTORY['velocity'] = []
            _HISTORY['surfaces'] = [ surface.copy() ]

    else: # Appending to current optimization history

        _HISTORY['energy'].append( J )
        _HISTORY['energy evaluations'].append(_ENERGY_EVALUATIONS )
        _HISTORY['shape gradient'].append( norm_G )
        _HISTORY['shape derivative'].append( shape_deriv )
        _HISTORY['step'].append( dt )
        if keep_history:
            _HISTORY['surfaces'].append( surface.copy() )
            _HISTORY['velocity'].append( V_vec.copy() )

    return _HISTORY['energy']

#############################################################################

def optimize( energy, surface, world_boundary=None,
              parameters=None, keep_history=True ):
    global _ENERGY_EVALUATIONS
    global _HISTORY
    global _VERBOSE

    # Initialize the parameters for shape optimization computations.

    if parameters is None:  parameters = deepcopy( default_optimization_parameters )

    velocity_params, (min_iters, max_iters), \
    step_params, (dt, old_dt, min_dt, max_dt, rho_down, rho_up, eta, Q) = \
                  _init_main_optim_parameters( parameters )


    # The energy should impose its accuracy criteria on the surface.

    energy.set_accuracy( surface )


    # Compute the initial shape energy and the norm of the shape gradient.

    track_J = J0 = energy( surface );   _ENERGY_EVALUATIONS = 1

    if not step_params['monotone line search']:  track_J += (1.0 - eta) * abs(J0)

    norm_G = energy.shape_gradient_norm( surface )

    energy_history = _update_optim_history( J0, dt, norm_G, None,
                                            surface, None, keep_history, reset=True )

    k = 0
    # max_step_tried = False
    take_N_steps_with_fixed_mesh = 0
    reducing_energy = True

    if _VERBOSE: print("Starting energy: %6.4f  \nStarting |G| = %6.4e" % (J0,norm_G))

    while (not energy.is_minimized(surface, parameters, _HISTORY) or k < min_iters ) \
           and reducing_energy and ( k < max_iters ):

        #-----------------------------------------------------------------
        surface.set_adaptivity(False)

        valid_step_found = False;  step_size_attempts = 0;  min_step_attempts = 0

        while not valid_step_found:

            dt_to_compute_V = min( rho_up*dt, max_dt ) # Only for semi-implicit desc. vel.

            V, V_vec, shape_deriv, shape_hess = \
               descent_velocity_and_shape_deriv( surface, energy, dt_to_compute_V, velocity_params )

            test_dt, new_J = choose_step_size( surface, V, V_vec,
                                               dt, old_dt, track_J,
                                               shape_deriv, shape_hess, energy,
                                               step_params, world_boundary )
            valid_step_found = test_dt > 0.0
            step_size_attempts += 1

            if valid_step_found:
                dt = test_dt
            else: # The line search failure might be due to energy-gradient
                # inconsistency. To fix this, refine the surface mesh based
                # on the shape gradient.
                dt = min_dt
                break

        #-----------------------------------------------------------------

        # TODO: CHECK and change the following line
#        if not valid_step_found:  break

        if take_N_steps_with_fixed_mesh > 0:
            take_N_steps_with_fixed_mesh -= 1
        else: # We are done taking steps with fixed mesh, so turn on adaptivity.
            surface.set_adaptivity(True)

        surface.move( dt*V_vec, world_boundary )

        norm_G = energy.shape_gradient_norm( surface )

        energy_history = _update_optim_history( new_J, dt, norm_G, shape_deriv,
                                                surface, V_vec, keep_history )

        track_J, Q, eta = update_energy_aggregate( track_J, new_J, Q, eta,
                                                   energy_history, step_params )
        old_dt = dt
        k = k+1

        if _VERBOSE: print("\nk=%d:  J=%f, |G|=%f, dJ=%f, old_dt=%f" %
                          (k, new_J, norm_G, shape_deriv, dt))

    print("\nk=%d:  J=%f, |G|=%f, dJ=%f, old_dt=%f" % (k, new_J, norm_G, shape_deriv, dt))
    print("\nTotal energy evaluations: %d" % _ENERGY_EVALUATIONS)

    # TODO: Need to undo "set_accuracy( surface )"

    if keep_history:
        return surface, _HISTORY
    else:
        return surface


def _update_energy_aggregate( track_J, new_J, Q, eta, energy_history, step_params, gradient_ratio=None ):

    if step_params['monotone line search']:  return (new_J, Q, eta)

    strategy, len_history = step_params.get('nonmonotone strategy',
                                            ('smoothed maximum',5))

    if strategy == 'recent maximum': # Grippo et al.
        track_J = max( energy_history[ -len_history: ] )

    elif strategy == 'smoothed maximum': # Amini et al.
        track_J = max( energy_history[ -len_history: ] )
        track_J = eta * track_J + (1-eta) * new_J

    elif strategy == 'recent average': # Zhang & Hager
        track_J = eta*Q*track_J + new_J
        Q = eta*Q + 1.0
        track_J /= Q
    else:
        raise ValueError('Unknown strategy for nonmonotone line search')

    # Adaptive adjustment of eta: decrease close to min
    ## if abs(norm_G) < 0.01:
    ##     eta = (4./5.)*eta + 0.01
    ## else:
    ##     eta = max( 0.97*eta, 0.5 )

    return (track_J, Q, eta)


## def _compute_semiimplicit_velocity(surface, energy, dt):
##     FEM = surface.FEM
##     X = surface.coords()
##     dim, n = X.shape

##     # TODO: This is true for ChanVeseEnergy only, needs to be fixed for others.
##     mu = energy._mu

##     M = FEM.mass_matrix( surface )
##     A = FEM.stiffness_matrix( surface )
##     N = [ FEM.normal_matrix( surface, k ) for k in range(dim) ]
##     D, f = energy.shape_gradient( surface, split=True, FEM='vector' )

##     # Assemble the right hand side vector.
##     rhs = [ -mu*(A*X[k]) - f[k] for k in range(dim) ]

##     # Assemble the coefficient matrix.
##     coef_mtx = M + (mu*dt)*A

##     # Solve for velocity.
##     V_vec = empty_like(X)
##     for k in range(dim):
##         V_vec[k] = coef_mtx.solve( rhs[k] )

##     V = M.solve( sum( N[k]*V_vec[k] for k in range(dim) ) )

##     return (V, V_vec)


def _compute_semiimplicit_velocity(surface, energy, dt):
    FEM = surface.FEM
    X = surface.coords()
    dim, n = X.shape

    M = FEM.mass_matrix( surface )
    A = FEM.stiffness_matrix( surface )
    N = [ FEM.normal_matrix( surface, k ) for k in range(dim) ]
    D, f = energy.shape_gradient( surface, split=True )

    # Assemble the right hand side vector.
    rhs = [ A*X[k,:] for k in range(dim) ]
    rhs.append( zeros( (dim+1)*n ) )
    rhs.append( -f )
    rhs = hstack( rhs )

    # Assemble the coefficient matrix.
    M = M.to_sparse()
    A = A.to_sparse()
    D = D.to_sparse()
    N = [ mtx.to_sparse() for mtx in N ]
    A_blocks = [ [None]*k +[ -dt*A ]+ [None]*(dim-k-1) for k in range(dim)]
    M_blocks = [ [None]*k +  [ M ]  + [None]*(dim-k-1) for k in range(dim)]
    A2 = sp_bmat( A_blocks )
    M2 = sp_bmat( M_blocks )
    N2 = sp_bmat( [ [-mtx] for mtx in N ] )
    N2t = -sp_bmat( [ N ] )

    coef_mtx = sp_bmat([[  A2,   M2,  None, None ],
                        [  M2,  None,  N2,  None ],
                        [ None,  N2t, None,   M  ],
                        [ None, None,   M,    D  ]], 'csr')

    # Solve the system for Y = (V_vec,K_vec,V,K).
    Y = spsolve( coef_mtx, rhs )
    V_vec = vstack([ Y[n*k:n*(k+1)] for k in range(dim) ])
    V = Y[ (2*dim*n):((2*dim+1)*n) ]

    return (V, V_vec)


def _gradient_based_velocity(surface, energy, dt, parameters):
    FEM = surface.FEM
    dim = surface.dim_of_world()
    gradient_type, param = parameters[0:2]

    if gradient_type == 'L2' and param != 'explicit': # => use semi-implicit
        return _compute_semiimplicit_velocity( surface, energy, dt )

    # Compute the FEM matrices needed for the velocity computations.
    M = FEM.mass_matrix( surface )

    # Compute the velocity coefficient matrix.
    if gradient_type == 'Newton':
        threshold = param if type(param) is float else None
        B = energy.shape_hessian( surface, threshold=threshold )

    elif gradient_type == 'L2':
        B = M

    elif gradient_type == 'H1':
        if isscalar(param): # compute A_ij = c \int \phi_i \phi_j
            A = param * FEM.stiffness_matrix( surface )
        else: # param is a function: compute A_ij = \int f(x) \phi_i \phi_j
            A = FEM.weighted_stiffness_matrix( surface, param )
        B = A + M

    # N: pair/triple of mass matrices weighted by the components of the normal
    N = [ FEM.normal_matrix( surface, k ) for k in range(dim) ]

    # Solve for velocity:  B V = -G = -shape_gradient
    G = energy.shape_gradient( surface )
    V = B.solve( -G )

    # Compute the velocity vector:  V_vec = V n,  n:normal
    V_vec = empty((dim,len(V)))
    for k in range(dim):
        V_vec[k] = M.solve( N[k]*V )

    return (V, V_vec)


def _hessian_based_velocity(surface, energy):
    # tau = 2.0  # Constant used in velocity selection criterion below

    if surface.size() == 0:  return ValueError("Surface is empty set!")
    FEM = surface.FEM
    M = FEM.mass_matrix( surface )

    # Compute the shape gradient.
    dJ = d2J = None # initialization
    G = energy.shape_gradient( surface )

    # Compute the shape Hessian of the energy.
    H = energy.shape_hessian( surface, full_hessian=True, threshold=-inf )

    # Compute the negative curvature direction.
    try:
        ## from scipy.sparse.linalg import LinearOperator
        ## n = surface.size()
        ## M2 = LinearOperator((n,n),matvec=M,dtype=float)
        ## M2inv = LinearOperator((n,n),matvec=lambda v: M.solve(v),dtype=float)
        ## H2 = LinearOperator((n,n),matvec=H,dtype=float)
        ## eigval, eigvec = eigs( H2, 1, M2inv, which='SR', #v0=-G,
        ##                        tol=1e-3, maxiter=1000, Minv=M2 )
        #eigval, eigvec = eigs( H, k=1, which='SR', v0=-G, tol=1e-6, maxiter=1000 )
        #eigval, eigvec = eigval.real, eigvec[:,0].real
        from numpy import eye, argmin
        from numpy.linalg import eig
        H2 = array([H*e for e in eye(surface.size())])
        eigval, eigvec = eig(H2)
        k = argmin(eigval.real)
        eigval, eigvec = eigval[k].real, eigvec[:,k].real
    except ArpackNoConvergence:
        eigval = None

    if eigval is None: # No eigval info, use modified Hessian to compute V.
        H_plus = energy.shape_hessian( surface, full_hessian=False, threshold=1.0 )
        V = H_plus.solve( -G )
        dJ = energy.shape_derivative( surface, V )
        if _VERBOSE: print("(x)", end=' ')

    elif eigval > 0.0: # H is SPD, so we use a full Newton direction.
        V = H.solve( -G )
        dJ = energy.shape_derivative( surface, V )
        if _VERBOSE: print("(+,ev:%4.3f) %f,%f" % (eigval,norm(V),dJ), end=' ')

    else: # min eigenvalue < 0, so check both Newton & neg.curv. directions.

        # If eigvec direction makes dJ > 0.0 (increases J), then flip sign.
        dJ_neg_curv = energy.shape_derivative( surface, eigvec )
        if dJ_neg_curv > 0.0:
            ## A = 0.001 * surface.FEM.stiffness_matrix(surface)
            ## eigvec = (A+M).solve(M*eigvec)
            ## dJ_neg_curv = energy.shape_derivative( surface, eigvec )
            eigvec *= -1
            dJ_neg_curv *= -1
        # Compute Q(V) = 0.5 * d2J(surface;V,V) + dJ(surface;V)
        #              = 0.5 * eigval + dJ(surface;eigvec)   (note <V,V>=1)
        quadratic_model_value = 0.5 * eigval + dJ_neg_curv
        ## norm_V = norm(eigvec)
        ## quadratic_model_value = 0.5*dot(eigvec,H*eigvec)/ norm_V**2 + dJ_neg_curv/norm_V

        # Compute the Newton direction using a pos.def. modification of the Hessian
        H_plus = energy.shape_hessian( surface, full_hessian=False, threshold=1.0 )
        V = H_plus.solve( -G )
        #V = H.solve( -G )
        dJ = energy.shape_derivative( surface, V )
        norm_V = norm(V)
        QN = 0.5*dot(V,H*V)/ norm_V**2 + dJ/norm_V
        # print("%f,%f" % (QN, quadratic_model_value), end=' ')

        # If neg. curvature direction is more promising than Newton direction.
        # if (dJ / norm(V)) > tau*quadratic_model_value:
        if QN > quadratic_model_value:
            if _VERBOSE: print("(-)", end=' ')
            V = eigvec
            dJ = dJ_neg_curv
            d2J = eigval
        else:
            if _VERBOSE: print("(-+)", end=' ')


    # Compute the vector velocity V_vec from the scalar velocity V

    dim = surface.dim_of_world()
    # N: pair/triple of mass matrices weighted by the components of the normal
    N = [ FEM.normal_matrix( surface, k ) for k in range(dim) ]
    # Compute the velocity vector:  V_vec = V n,  n:normal
    V_vec = empty((dim,len(V)))
    for k in range(dim):
        V_vec[k] = M.solve( N[k]*V )

    return (V, V_vec, dJ, d2J)


def _descent_velocity_and_shape_deriv(surface, energy, dt, parameters):
    if surface.size() == 0:  return ValueError("Surface is empty set!")

    descent_dir = parameters[0]
    normalize_V = True if (parameters[-1] == True) else False

    if descent_dir in ['Hessian','hessian']:
        V, V_vec, dJ, d2J = _hessian_based_velocity( surface, energy )

    elif descent_dir in ['L2','H1','Newton','newton']:
        V, V_vec = _gradient_based_velocity( surface, energy, dt, parameters )
        if normalize_V: # => normalize velocity magnitude.
            norm_V = norm(V);   V /= norm_V;   V_vec /= norm_V
            #M = surface.FEM.mass_matrix( surface )
            #norm_V = dot(V,M*V)**0.5;   V /= norm_V;   V_vec /= norm_V
        dJ = energy.shape_derivative( surface, V )
        d2J = None

    else: # Unknown descent direction!
        raise ValueError("Descent direction should be one of "+
                         "'L2','H1','Newton','Hessian'.")

    return (V, V_vec, dJ, d2J)


def _choose_step_size(surface, V, V_vec, dt, old_dt, J0, dJ, d2J, energy,
                     parameters=default_step_size_parameters,
                     world_boundary=None, test_step=None):
    if d2J is None:
        return _backtrack_step_size(surface, V, V_vec, dt, old_dt, J0, dJ, d2J,
                                   energy, parameters, world_boundary, test_step)

    else: # d2J is given, we are using Hessian-based velocity => maximize step.
        return _maximize_step_size( surface, V, V_vec, dt, old_dt, J0, dJ, d2J,
                                   energy, parameters, world_boundary )


def _init_line_search_parameters(surface, energy, V, dt, old_dt, parameters):
    defaults = default_step_size_parameters

    min_dt = parameters.get('minimum step size', defaults['minimum step size'])
    max_dt = parameters.get('maximum step size', defaults['maximum step size'])
    rho1   = parameters.get('decrease factor', defaults['decrease factor'])
    rho2   = parameters.get('increase factor', defaults['increase factor'])
    alpha  = parameters.get('alpha', defaults['alpha'])
    safe_step_tol = parameters.get('safe step tolerance', defaults['safe step tolerance'])
    restart_with_max_dt = parameters.get('restart with maximum', defaults['restart with maximum'])

    if not (0.0 < rho1 < 1.0):
        raise ValueError("The step decrease factor rho1 has to be between 0.0 and 1.0!")
    if not (rho2 > 1.0):
        raise ValueError("The step increase factor rho2 has to be greater than 1.0!")

    # Compute the max step size that doesn't distort the surface mesh.
    if surface.dim() > 1 and safe_step_tol is not None:
        max_dt = min( max_dt, surface.safe_step_size( V, safe_step_tol ) )

    # Check the energy basin/valley width, so that max step size won't overshoot it.
    try:
        basin_width = energy.basin_width()
        if basin_width is not None:
            max_dt = min( max_dt, basin_width/abs(V).max() )
    except NotImplementedError:
        pass

    min_dt = min( min_dt, rho1*max_dt )

    return (min_dt, max_dt, rho1, rho2, alpha, restart_with_max_dt)


def _backtrack_step_size(surface, V, V_vec, dt, old_dt, J0, dJ, d2J, energy,
                        parameters, world_boundary=None, test_step=None):
    global _ENERGY_EVALUATIONS
    global _HISTORY
    global _VERBOSE
    if surface.size() == 0:  raise ValueError("Surface is empty set!")

    # Get the parameters for the line search.
    min_dt, max_dt, rho1, rho2, alpha, restart_with_max_dt = \
            _init_line_search_parameters( surface, energy, V, dt, old_dt, parameters )

    # Assign the new step size to be tried by line search.
    if test_step is not None: # A test step was already provided.
        dt,J_test = test_step
        new_dt = max( rho1*dt, min_dt )  # Reduce step.
    else: # A test step wasn't provided, so we need to choose one.
        J_test = inf
        if restart_with_max_dt:
            new_dt = max_dt
        elif dt >= old_dt: # Try a larger new step.
            new_dt = min( rho2*dt, max_dt )
        else: # dt < old_dt  # Try current step again (being conservative).
            new_dt = max( dt, min_dt )

    # While loop: reduce step size dt as needed to satisfy Armijo condition.
    d2J = 0.0 if (d2J is None) else min( 0.0, d2J )
    # TODO: Finalize the energy offset idea.
    offset = 0.0
#    if not parameters['use error offset']:
#        offset = 0.0
#    else:
#        n = min(energy._pixels.shape)
#        offset = 0.0 #0.01 * integrate(energy._image, surface) / (n-1)

    i=0; polymod_failed = False
    energy_is_reduced = (J_test < J0 + alpha*(dt*dJ + 0.5*dt*dt*d2J) + offset)

    while not energy_is_reduced and (new_dt > min_dt):
        if _VERBOSE: print(str(i), end=' ')
        prev_dt, dt, J_prev = dt, new_dt, J_test

        # Keep moving surface with reduced dt until we take a successful step.
        while not surface.move( dt*V_vec, world_boundary ):
            dt = rho1*dt
            if _VERBOSE: print('x', end=' ')

        J_test = energy( surface );   _ENERGY_EVALUATIONS += 1
        if _VERBOSE: print("(%4.2e, %10.8f)" % (dt,J_test), end=' ')
        energy_is_reduced = (J_test < J0 + alpha*(dt*dJ + 0.5*dt*dt*d2J) + offset)

        surface.move_back() # move back, b/c this is just a test step.
        i = i+1

        # Now predict a new step candidate using the two previous steps.
        if not polymod_failed:
            new_dt = _polymod_step(J0,dJ, dt,J_test, 0.1,0.5, prev_dt,J_prev)
            polymod_failed = (dt == new_dt)
        # If polymod failed at prev iter, then stop using it, reduce dt by rho1
        if polymod_failed:
            new_dt = rho1*dt

        new_dt = max( min(new_dt,max_dt), min_dt )

    if not energy_is_reduced:  dt = 0.0  # Set dt=0 when line search fails.
    if _VERBOSE: print("  dt = %f   (new dt = %f, min dt = %f, max dt = %f)" % (dt,new_dt,min_dt,max_dt))
    return (dt, J_test)


def _maximize_step_size(surface, V, V_vec, dt, old_dt, J0, dJ, d2J, energy,
                       parameters, world_boundary=None, test_step=None):
    global _ENERGY_EVALUATIONS
    global _HISTORY
    global _VERBOSE
    if surface.size() == 0:  raise ValueError("Surface is empty set!")

    # Get the parameters for the line search.
    min_dt, max_dt, rho1, rho2, alpha, restart_with_max_dt = \
            _init_line_search_parameters( surface, energy, V, dt, old_dt, parameters )

    # Assign the new step size to be tried by line search.
    if restart_with_max_dt:
        new_dt = max_dt
    elif dt >= old_dt: # Try a larger step.
        new_dt = min( rho2*dt, max_dt )
    else: # dt < old_dt  # Try a smaller step.
        new_dt = max( dt, min_dt )

    # Take 1st step new_dt & decide whether to decrease or increase step size.
    test_step = None
    move_success = surface.move( new_dt*V_vec, world_boundary )

    if not move_success: # move failed, need to reduce step size.
        reduce_step_size = True
    else: # surface.move() succeeded, so check energy J_test
        J_test = energy( surface );   _ENERGY_EVALUATIONS += 1
        if _VERBOSE: print("(%6.4f)" % J_test, end=' ')
        surface.move_back()
        test_step = new_dt, J_test
        reduce_step_size = J_test > J0 + alpha*new_dt*(dJ + 0.5*new_dt*d2J)

    # If step new_dt didn't work, use default method to reduce step size.
    if reduce_step_size:
        print('Need to reduce')
        return backtrack_step_size( surface, V, V_vec, dt, old_dt, J0, dJ, d2J,
                                    energy, parameters, test_step )

    # If moving with test_dt was successful, then try increasing test_dt.
    i = 0;  dt = new_dt;  new_dt = rho2*new_dt;  J_test_old = J0
    while (J_test <= J0 + alpha*(dt*dJ + 0.5*dt*dt*d2J)) and \
              (J_test <= J_test_old) and (new_dt < rho2*max_dt):
        if _VERBOSE: print(str(i), end=' ')
        prev_dt, dt, new_dt = dt, new_dt, rho2*new_dt
        move_success = surface.move( dt*V_vec, world_boundary )

        if not move_success: # move failed, stop iterations
            return (prev_dt, J_test)
        elif surface.size() == 0: # move success, but surface vanished, stop
            surface.move_back()
            return (prev_dt, J_test)
        else: # move success
            J_test_old = J_test
            J_test = energy( surface );   _ENERGY_EVALUATIONS += 1
            if _VERBOSE: print("(%8.6f)" % J_test, end=' ')
            surface.move_back()
            i = i-1

    dt = prev_dt if new_dt < rho2*max_dt else min(dt,max_dt)
    J_test = J_test_old if new_dt < rho2*max_dt else J_test
    if _VERBOSE: print("  dt = %f" % dt)
    return (dt, J_test)


def _polymod_step(J0, dJ0, dt1, J1, beta_low, beta_high, dt2=None, J2=None):
    min_dt, max_dt = beta_low*dt1, beta_high*dt1

    if (dt1 == dt2) and (J2 is not None) and not isinf(J2):
        raise ValueError("The steps dt1 & dt2 should have different values!")

    if (dt2 is None) or (J2 is None) or isinf(J2): # then use quadratic model
        new_dt = -dJ0 / (2.0 * dt1*(J1 - J0 - dJ0))

    else: # (dt2,J2) undefined, then use the cubic model
        A00, A01, A10, A11 = dt1**2, dt1**3, dt2**2, dt2**3
        Ainv = array([[A11, -A01], [-A10, A00]]) / (A00*A11 - A01*A10)
        b = array([ J1 - (J0 + dJ0*dt1), J2 - (J0 + dJ0*dt2) ])
        c = dot(Ainv,b)
        delta = max( 0.0, (c[0]**2 - 3*c[1]*dJ0) )
        new_dt = (-c[0] + sqrt(delta)) / (3*c[1])

    new_dt = max( min(new_dt,max_dt), min_dt )
    return new_dt
