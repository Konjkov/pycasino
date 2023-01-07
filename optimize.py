import warnings
from numpy_config import np
from scipy.optimize._optimize import (
    OptimizeResult,
    _prepare_scalar_function,
    _LineSearchError,
    _epsilon,
    _status_message,
    _minimize_newtoncg,
)
from scipy.optimize._linesearch import line_search_wolfe1, line_search_wolfe2, LineSearchWarning


def _line_search_wolfe12(f, fprime, xk, pk, gfk, old_fval, old_old_fval,
                         **kwargs):
    """
    Same as line_search_wolfe1, but fall back to line_search_wolfe2 if
    suitable step length is not found, and raise an exception if a
    suitable step length is not found.

    Raises
    ------
    _LineSearchError
        If no suitable step size is found

    """

    extra_condition = kwargs.pop('extra_condition', None)

    ret = line_search_wolfe1(f, fprime, xk, pk, gfk,
                             old_fval, old_old_fval,
                             **kwargs)

    if ret[0] is not None and extra_condition is not None:
        xp1 = xk + ret[0] * pk
        if not extra_condition(ret[0], xp1, ret[3], ret[5]):
            # Reject step if extra_condition fails
            ret = (None,)

    if ret[0] is None:
        # line search failed: try different one.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', LineSearchWarning)
            kwargs2 = {}
            for key in ('c1', 'c2', 'amax'):
                if key in kwargs:
                    kwargs2[key] = kwargs[key]
            ret = line_search_wolfe2(f, fprime, xk, pk, gfk,
                                     old_fval, old_old_fval,
                                     extra_condition=extra_condition,
                                     **kwargs2)

    if ret[0] is None:
        raise _LineSearchError()

    return ret


def minimize_newtoncg(fun, x0, args=(), jac=None, hess=None, hessp=None,
                      callback=None, xtol=1e-5, eps=_epsilon, maxiter=None,
                      disp=False, return_all=False, bounds=None, constraints=None):
    """
    Minimization of scalar function of one or more variables using the
    Newton-CG algorithm.

    Note that the `jac` parameter (Jacobian) is required.

    Options
    -------
    disp : bool
        Set to True to print convergence messages.
    xtol : float
        Average relative error in solution `xopt` acceptable for
        convergence.
    maxiter : int
        Maximum number of iterations to perform.
    eps : float or ndarray
        If `hessp` is approximated, use this value for the step size.
    return_all : bool, optional
        Set to True to return a list of the best solution at each of the
        iterations.
    """
    if jac is None:
        raise ValueError('Jacobian is required for Newton-CG method')
    if hess is None:
        raise ValueError('Hessian is required for Newton-CG method')

    x0 = np.asarray(x0).flatten()
    sf = _prepare_scalar_function(
        fun, x0, jac, args=args, epsilon=eps, hess=hess
    )

    def terminate(warnflag, msg):
        if disp:
            print(msg)
            print("         Current function value: %f" % old_fval)
            print("         Iterations: %d" % k)
            print("         Function evaluations: %d" % sf.nfev)
            print("         Gradient evaluations: %d" % sf.ngev)
            print("         Hessian evaluations: %d" % hcalls)
        fval = old_fval
        result = OptimizeResult(fun=fval, jac=gfk, nfev=sf.nfev,
                                njev=sf.ngev, nhev=hcalls, status=warnflag,
                                success=(warnflag == 0), message=msg, x=xk,
                                nit=k)
        if return_all:
            result['allvecs'] = allvecs
        return result

    hcalls = 0
    maxiter = maxiter or len(x0) * 200
    cg_maxiter = 20 * len(x0)

    xtol = len(x0) * xtol
    update = [2 * xtol]
    xk = x0
    if return_all:
        allvecs = [xk]
    k = 0
    gfk = None
    old_fval = sf.fun(x0)
    old_old_fval = None
    float64eps = np.finfo(np.float64).eps
    while np.add.reduce(np.abs(update)) > xtol:
        if k >= maxiter:
            msg = "Warning: " + _status_message['maxiter']
            return terminate(1, msg)
        # Compute a search direction pk by applying the CG method to
        #  del2 f(xk) p = - grad f(xk) starting from 0.
        b = -sf.grad(xk)
        maggrad = np.add.reduce(np.abs(b))
        eta = np.min([0.5, np.sqrt(maggrad)])
        termcond = eta * maggrad
        xsupi = np.zeros(len(x0), dtype=x0.dtype)
        ri = -b
        psupi = -ri
        i = 0
        dri0 = np.dot(ri, ri)
        A = sf.hess(xk)
        hcalls = hcalls + 1

        for k2 in range(cg_maxiter):
            if np.add.reduce(np.abs(ri)) <= termcond:
                break
            Ap = A @ psupi
            # check curvature
            Ap = np.asarray(Ap).squeeze()  # get rid of matrices...
            curv = psupi @ Ap
            if 0 <= curv <= 3 * float64eps:
                break
            elif curv < 0:
                if i > 0:
                    break
                else:
                    # fall back to steepest descent direction
                    xsupi = dri0 / (-curv) * b
                    break
            alphai = dri0 / curv
            xsupi = xsupi + alphai * psupi
            ri = ri + alphai * Ap
            dri1 = np.dot(ri, ri)
            betai = dri1 / dri0
            psupi = -ri + betai * psupi
            i = i + 1
            dri0 = dri1  # update np.dot(ri,ri) for next time.
        else:
            # curvature keeps increasing, bail out
            msg = ("Warning: CG iterations didn't converge. The Hessian is not "
                   "positive definite.")
            return terminate(3, msg)

        pk = xsupi  # search direction is solution to system.
        gfk = -b  # gradient at xk

        try:
            alphak, fc, gc, old_fval, old_old_fval, gfkp1 = \
                _line_search_wolfe12(sf.fun, sf.grad, xk, pk, gfk,
                                     old_fval, old_old_fval)
        except _LineSearchError:
            # Line search failed to find a better solution.
            msg = "Warning: " + _status_message['pr_loss']
            return terminate(0, msg)

        update = alphak * pk
        xk = xk + update  # upcast if necessary
        if callback is not None:
            callback(xk)
        if return_all:
            allvecs.append(xk)
        k += 1
    else:
        if np.isnan(old_fval) or np.isnan(update).any():
            return terminate(3, _status_message['nan'])

        msg = _status_message['success']
        return terminate(0, msg)
