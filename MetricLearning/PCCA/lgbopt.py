""" 
    ==================================
    Light Gradient Based Optimization
    ==================================

    This module is dedicated to gradient based optimization schemes when
    the gradient is expensive to compute. This means that the gradient
    computation is avoided as much as possible.

    Specifically, the line search procedure do not recomputed the
    gradient during the process and looks for a point verifying the
    following sufficient decrease condition (aka Armijo condition):
    
        f(x + alpha * p) <= f(x) + c * alpha * < df(x), p >

    where f is the objective to minimize, x is the current point, 
    p is the descent direction, c is a constant, df(x) is the gradient
    at point x, < ., .> represents the inner product and alpha is the
    descent step we want to determine.

    Two minimization routines are provided :
        
    fmin_gd :
        is the steepest gradient descent algorithm.

    fmin_lgfbs :
        uses l-gfbs quasi-Newton method (sparse approximation
        of the hessian matrix).

    fmin_cg :
        uses Fletcher-Reeves conjugate gradient method.

    :Note:

        The implementation is based on the description of the
        algorithms found in:
        
            \J. Nocedal and S. Wright. Numerical Optimization.
    
    :Author: Alexis Mignon (c) Oct. 2012
    :E-mail: alexis.mignon@gmail.com

"""
try:
    from numpy import sqrt, abs, inner as inner_, sum
except ImportError:
    def inner_(x, y):
        """ Computes inner product between two vectors
        """
        return sum(( xi * yi for xi, yi in zip(x, y)))
    from math import sqrt, abs

class MultiOptVar(object):
    """ class MultiOptVar

    class to hold an optimization variable made of several independent
    variables.

    MultiOptVar(var1, [var2, ...], inner=inner_)
    
    Parameters:
    -----------
    var1, var2, etc: objects
        the independant variables. These variables must support
        basic algebraic operations in vector spaces.

    inner: function or list of functions, optional
        The inner product to be used on independant variables.
        If a single function is given then the inner product is assumed
        to be the same for all variables.
        If a list of function is given then they are used as inner
        products for each independant variable respectively.
        By default, the standard inner product on vectors is used.

    """
    def __init__(self, *variables, **kwargs):
        """ constructor
        """
        self.variables = variables
        inner = kwargs.get("inner", inner_)
        try:
            self.inner = list(inner)
        except TypeError:
            self.inner = [inner for i in range(len(variables))]

    def __iter__(self):
        return iter(self.variables)

    def __mul__(self, scalar):
        return MultiOptVar(*(scalar * var for var in self.variables), inner = self.inner)

    def __rmul__(self, scalar):
        return MultiOptVar(*(scalar * var for var in self.variables), inner = self.inner)

    def __neg__(self):
        return MultiOptVar(*(-var for var in self.variables), inner = self.inner)

    def __add__(self, other):
        return MultiOptVar(*(var1 + var2 for var1, var2 in zip(self.variables, other.variables)), inner = self.inner)

    def __sub__(self, other):
        return MultiOptVar(*(var1 - var2 for var1, var2 in zip(self.variables, other.variables)), inner = self.inner)

    def __div__(self, scalar):
        return MultiOptVar(*(var/scalar for var in self.variables), inner = self.inner)

    def dot(self, other):
        return sum(
            [ inner(var1,var2) for var1, var2, inner
                    in zip(self.variables, other.variables, self.inner)
            ]
        )

    @staticmethod
    def inner(var1, var2):
        return var1.dot(var2)

def line_search(f, x0, df0, p=None, f0=None, args=(), alpha_0=1.0, c=1e-4, inner=inner_, 
                maxiter=100, rho_lo=1e-3, rho_hi=0.9, augs = None):
    """ Interpolation Line search for the steapest gradient descent.

    Finds the a step length in the descending direction -df0 verifying
    the Amijo's sufficient decrease conditions.

    Parameters:
    ----------
    f : callable
        the function to minimize.
    x0 : array_like
        the starting point.
    df0 : array_like
        the gradient value at x0
    p : array_like, optional
        the descent direction. If None (default) -df0 is taken.
    f0 : float, optional
        The function value at x0. If None (default) it is computed.
    alpha_0 : float, optional
        the initial descent step (default = 1.0).    
    c : float, optional
        the constant used for the sufficient decrease (Armijo), condition:
        f(x + alpha * p) <= f(x) + c * alpha * < df(x), p>
        (default = 1e-4)
    inner : callable, optional
        the function used to compute the inner product. The default
        is the ordinary dot product.
    maxiter : int, optional
        maximum number of iterations allowed. (default=100)
    rho_lo : float, optional
        Lowest ratio valued allowed between steps coefficient  in concecutive
        iterations. (default=1e-3)
    rho_hi : float, optional
        lowest ratio valued allowed between steps coefficient in concecutive
        iterations. (default=0.9)
        If not rho_lo <= alpha_[t+1]/alpha_[t] <= rho_hi, then
        alpha_[t+1] = 0.5 * alpha_[t] is taken.

    Returns:
    --------
        (xopt, fval)
    xopt: array_like
        the optimal point
    fval: float
        the optimal value found

    Notes:
    ------
        
    Adapted for Gradient Descent from the interpolation procedure
    described in:
        
        \J. Nocedal and S. Wright. Numerical Optimization. Chap.3 p56

    In this implementation x0 (and fd0) can be any object supporting
    addition, multiplication by a scalar and for which the inner
    product is defined (through the 'inner' function).
    """
    if f0 is None:
        f0 = f(x0, *args)

    if p is None:
        p = -df0

    dphi0 = inner(df0, p)

    x1 = x0 + alpha_0 * p
    f1 = f(x1, augs, *args)

    if f1 <= f0 + c * alpha_0 * dphi0:
        return x1, f1

    # perfoms quadratic interpolation
    alpha_0_2 = alpha_0 * alpha_0
    alpha_1 = - dphi0* alpha_0_2 / ( 2 * (f1 - f0 - dphi0 * alpha_0) )

    x2 = x0 + alpha_1 * p

    f2 = f(x2, augs, *args)

    if f2 <= f0 + c * alpha_1 * dphi0:
        return x2, f2

    alpha_0_3 = alpha_0_2 * alpha_0

    iter_ = 0
    while True:
        # performs cubic interpolation
        alpha_1_2 = alpha_1 * alpha_1
        ff1 = f2 - f0 - dphi0 * alpha_1
        ff0 = f1 - f0 - dphi0 * alpha_0

        den = 1/(alpha_0_2 * alpha_1_2 * (alpha_1 - alpha_0))
        _3a = 3 * (alpha_0_2 * ff1 - alpha_1_2 * ff0) / den
        b = (alpha_1_2 * alpha_1 * ff0 - alpha_0_3 * ff1) / den

        alpha_2 = (-b + sqrt(max(0,b*b - _3a * dphi0)))/ _3a

        if not  rho_lo <= alpha_2/alpha_1 <= rho_hi:
            alpha_2 = alpha_1 / 2.

        x3 = x0 + alpha_2 * p
        f3 = f(x3, augs, *args)

        if f3 <= f0 + c * alpha_2 * dphi0:
            return x3, f3

        iter_ += 1
        if iter_ >= maxiter:
            print "Maximum number of iteration reached before a good step "+ \
                    "size was found!"
            return x3, f3

        x1 = x2
        x2 = x3
        f1 = f2
        f2 = f3
        alpha_0 = alpha_1
        alpha_1 = alpha_2

def _print_info(iter_, fval, grad_norm2):
    """ prints information about convergence"""
    print "iter:", iter_, "fval:", fval, "|grad|:", sqrt(grad_norm2)

def fmin_gd(f, df, x0, args=(), alpha_0=1.0, gtol=1e-6, maxiter=100, 
            maxiter_line_search=100, c=1e-4, inner=inner_,
            rho_lo=1e-3, rho_hi=0.9, 
            verbose=False, callback=None, augs=None):
    """ Steepest gradient descent optimization.

    Parameters
    ----------
    f : callable
        the function to be minimized.
    df : callable
        the function that computed the gradient.
    x0 : array_like
        the starting point.
    alpha_0 : float. optional (default 1.0)
        Starting value for the descent step.
    gtol : float
        the value of the gradient norm under which we consider the optimization
        as converged.
    maxiter : int
        Maximum number of iterations allowed.
    maxiter_line_search : int
        Maximum number of iteration allowed for the inner line_search process.
    c : float
        The constant used for the sufficient decrease (Armijo) condition:
            f(x + alpha * p) <= f(x) + c * alpha * < df(x), p >
        (default = 1e-4)
    inner : callable
        the function used to compute the inner product. The default is the
        ordinary dot product.
    verbose : boolean
        If True, displays information about the convergence of the algorithm.
    rho_lo : float
        Lowest ratio valued allowed between steps coefficient in concecutive
        iterations. (default=1e-3)
    rho_hi : float
        Lowest ratio valued allowed between steps coefficient in concecutive
        iterations. (default=0.9).
        If not rho_lo <= alpha_[t+1]/alpha_[t] <= rho_hi, then
        alpha_[t+1] = 0.5 * alpha_[t] is taken.
    callback : callable
        A function called after each iteration. The function
        is called as callback(x).

    Returns:
    -------
        (xopt, fval)

    xopt : ndarray
        The optimal point
    fval : float
        the optimal value found
    """

    f0 = f(x0, augs, *args)
    dfx = df(x0, *args)

    alpha_start = alpha_0
    norm_dfx = inner(dfx, dfx)

    if verbose:
        _print_info(0, f0, norm_dfx)

    if norm_dfx <= gtol * gtol:
        return x0, f0

    iter_ = 0
    while True:
        x1, f1 = line_search(f, x0, dfx, f0=f0, args=args,
                                alpha_0=alpha_start, c=c, inner=inner, 
                                maxiter=maxiter_line_search, augs=augs,
                                rho_lo=rho_lo, rho_hi=rho_hi)
        if f1 >= f0:
            alpha_start = alpha_0
            x1, f1 = line_search(f, x0, dfx, f0=f0, args=args,
                                alpha_0=alpha_start, c=c, inner=inner, 
                                maxiter=maxiter_line_search, augs = augs,
                                rho_lo=rho_lo, rho_hi=rho_hi)
            if f1 >= f0:
                print "Could not minimize in the descent direction"
                return x0, f0

        if callback is not None:
            callback(x1)
        iter_ += 1
        if iter_ >= maxiter:
            print "Maximum number of iteration reached."
            return x1, f1

        dfx = df(x1, *args)
        norm_dfx = inner(dfx, dfx)
        if verbose:
            if iter_%100 == 0:
                _print_info(iter_, f1, norm_dfx)
        
        if norm_dfx <= gtol * gtol:
            return x1, f1

        alpha_start = 2*(f0 - f1)/norm_dfx
        x0 = x1
        f0 = f1

def fmin_lbfgs(f, df, x0, args=(), alpha_0=1.0, m=5, gtol=1e-6, maxiter=100, 
                maxiter_line_search=10, c=1e-4, inner=inner_, 
                verbose=False, rho_lo=1e-3, rho_hi=0.9, callback=None):
    """ Optimization with the Low-memory Broyden, Fletcher, Goldfarb, 
    and Shanno (l-BFGS) quasi-Newton method.

    Parameters
    ----------

    f : callable
        The function to minimize.
        
    df : callable
        the function that computed the gradient.
        
    x0 : array_like
        The starting point.
        
    alpha_0 : float, optional
        Starting value for the descent step.
        
    m : int, optional
        Number of points used to approximate the inverse of the Hessian matrix.
        
    gtol : float, optional
        the value of the gradient norm under which we consider
        the optimization as converged.
        
    maxiter : int, optional
        Maximum number of iterations allowed.
        
    maxiter_line_search : int, optional
        maximum number of iteration allowed for the inner line_search process.
        
    c : float, optional
        the constant used for the sufficient decrease (Armijo)
        condition:
            f(x + alpha * p) <= f(x) + c * alpha * < df(x), p >
        (default = 1e-4)
        
    inner: callable
        the function used to compute the inner product. The default
        is the ordinary dot product.

    verbose : boolean
        If True, displays information about the convergence of the algorithm.
        
    callback: float
        A function called after each iteration. The function is called as
        callback(x).

    Returns:
    --------
            (xopt, fval)
            
    xopt : array_like
        the optimal point

    fval : float
        The optimal value found

    Notes
    -----
    
        In this implementation x0 (and fd0) can be any object supporting
        addition, multiplication by a scalar and for which the inner
        product is defined (through the 'inner' function).
        
        Implemented from \:

            \J. Nocedal and S. Wright. Numerical Optimization.
    """
    sy = []

    f0 = f(x0, *args)
    dfx = df(x0, *args)

    norm_dfx = inner(dfx, dfx)

    if verbose:
        _print_info(0, f0, norm_dfx)

    if norm_dfx <= gtol * gtol:
        return x0, f0

    p = -dfx

    iter_ = 0
    gamma = 1.0
    
    while True:
        x1, f1 = line_search(f, x0, dfx, p=p, f0=f0, args=args,
                            alpha_0=alpha_0, 
                            c=c, inner=inner, maxiter=maxiter_line_search, 
                            rho_lo=rho_lo, rho_hi=rho_hi)
        if f1 >= f0:
            print "Could not minimize in the descent direction, try "+\
                    "steepest direction"
            sy = []
            p = -dfx
            x1, f1 = line_search(f, x0, dfx, p=p, f0=f0, args=args,
                            alpha_0=alpha_0, 
                            c=c, inner=inner, maxiter=maxiter_line_search, 
                            rho_lo=rho_lo, rho_hi=rho_hi)
            if f1 >= f0:
                print "Could not minimize in the steepest direction: abort"
                return x0, f0

        if callback is not None:
            callback(x1)
        iter_ += 1
        if iter_ >= maxiter:
            print "Maximum number of iteration reached."
            return x1, f1

        dfx1 = df(x1, *args)
        norm_dfx1 = inner(dfx1, dfx1)

        if verbose:
            _print_info(iter_, f1, norm_dfx1)
        
        if norm_dfx1 <= gtol * gtol:
            return x1, f1

        s = (x1-x0)
        y = (dfx1 - dfx)
        rho = 1.0/inner(y, s)
        gamma1 = inner(y, s)/inner(y, y)
        
        sy.append((y, s, rho))
        if len(sy) > m:
            sy.pop(0)

        q = dfx1.copy()
        a = []
        for s, y, rho in sy[-2::-1]:
            ai = rho * inner(s, q)
            q -= ai * y
            a.insert(0, ai)

        r = gamma * q
        for (s, y, rho), ai in zip(sy[:-1], a):
            b = rho * inner(y, r)
            r += s * (ai - b)
        p = -r

        x0 = x1
        f0 = f1
        dfx = dfx1
        gamma = gamma1

def fmin_cg(f, df, x0, args=(), alpha_0=1.0, gtol=1e-6, maxiter=100, 
            maxiter_line_search=100, c=1e-4, inner=inner_, 
            restart_coef = 0.1, verbose=False, callback=None):
    """ Steepest gradient descent optimization.

    Parameters
    ----------
    f : callable
        the function to be minimized.
    df : callable
        the function that computed the gradient.
    x0 : array_like
        the starting point.
    alpha_0 : float. optional (default 1.0)
        Starting value for the descent step.
    gtol : float
        the value of the gradient norm under which we consider the optimization
        as converged.
    maxiter : int
        Maximum number of iterations allowed.
    maxiter_line_search : int
        Maximum number of iteration allowed for the inner line_search process.
    c : float
        The constant used for the sufficient decrease (Armijo) condition:
            f(x + alpha * p) <= f(x) + c * alpha * < df(x), p >
        (default = 1e-4)
    inner : callable
        the function used to compute the inner product. The default is the
        ordinary dot product.
    restart_coef : float
        Replace conjugate gradient step with steapest descent step when:
        < df(x[k]), df(x[k-1]) >/< df(x[k]), df(x[k]) > >= restart_coef
        i.e. when the angle between two consecutive gradient directions
        are not orthogonal enough.
    verbose : boolean
        If True, displays information about the convergence of the algorithm.

    callback : callable
        A function called after each iteration. The function
        is called as callback(x).

    Returns:
    -------
        (xopt, fval)

    xopt : ndarray
        The optimal point
    fval : float
        the optimal value found
    """

    f0 = f(x0, *args)
    dfx = df(x0, *args)

    norm_dfx = inner(dfx, dfx)

    if verbose:
        _print_info(0, f0, norm_dfx)

    if norm_dfx <= gtol * gtol:
        return x0, f0

    p = -dfx

    iter_ = 0
    gamma = 1.0
    
    while True:
        x1, f1 = line_search(f, x0, dfx, p=p, f0=f0, args=args,
                            alpha_0=alpha_0, c=c,
                            inner=inner, maxiter=maxiter_line_search)
        if f1 >= f0:
            print "Could not minimize in the descent direction, try "+\
                    "steepest direction"
            beta = 0.0
            p = -dfx1
            x1, f1 = line_search(f, x0, dfx, p=p, f0=f0, args=args,
                            alpha_0=alpha_0, c=c,
                            inner=inner, maxiter=maxiter_line_search)
            if f1 >= f0:
                print "Could not minimize in the steepest direction:"+\
                        " abort."
            return x0, f0

        if callback is not None:
            callback(x1)
        iter_ += 1
        if iter_ >= maxiter:
            print "Maximum number of iteration reached."
            return x1, f1

        dfx1 = df(x1, *args)
        norm_dfx1 = inner(dfx1, dfx1)

        if verbose:
            _print_info(iter_, f1, norm_dfx1)
        
        if norm_dfx1 <= gtol * gtol:
            return x1, f1
            
        angle = abs(inner(dfx, dfx1))/norm_dfx1
        if angle >= restart_coef:
            beta = 0.0
            p = -dfx1
        else:
            beta = norm_dfx1/norm_dfx
            p = - dfx1 + beta * p

        x0 = x1
        f0 = f1
        dfx = dfx1
