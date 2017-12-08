# -*- coding: utf-8 -*-
# Author: Alexis Mignon <alexis.mignon@gmail.com>
# Date: 19/03/2015
# pylint: disable=C0103
"""
========================
Distance Metric Learning
========================

"""
import numpy as np
# pylint: disable=E0611
from scipy.sparse import linalg as la, csr_matrix
import lgbopt

def compute_differences(X, pairs):
    """Compute differences between pairs of elements

Parameters
----------
X: 2d array, shape (n_samples, n_features)
    The data array

pairs: 2d array, shape (n_pairs, 2)
    Pair of element indices

Returns
-------
diff: 2d array, shape (n_pairs, n_features)
    ``diff[i] = X[pairs[i, 0]] - X[pairs[i, 1]]``
"""
    return X[pairs[:, 0]] - X[pairs[:, 1]]


def compute_distances(X, pairs, proj_mat=None, squared=True):
    """Compute distances between pairs of data points

Parameters
----------
X: 2d array, shape (n_samples, n_features)
    The data array

pairs: 2d array, shape (n_pairs, 2)
    Pair of element indices

proj_mat: 2d array, shape (n_components, n_features), optional (default None)
    Projection matrix applied to the data point before computing the
    Euclidean distance.

squared: bool, optional (default True)
    If True, returns the squared Euclidean distance, otherwise, returns
    Euclidean distance

Returns
-------
distances: 2d array, shape (n_pairs,)
    Euclidean distance (squared if requested) between pairs of data points.
"""
    diff = compute_differences(X, pairs)
    if proj_mat is not None:
        # pylint: disable=E1101
        diff = np.dot(diff, proj_mat.T)
    dists = (diff ** 2).sum(-1)
    # pylint: disable=E1101
    return dists if squared else np.sqrt(dists)


def compute_score_from_distances(distances, labels, threshold):
    """Compute score using distance values"""
    # pylint: disable=E1101
    return (np.sign(threshold - distances) == labels).mean()


def compute_score(X, pairs, labels, proj_mat=None, threshold=None):
    """Compute pairs classification score based on distance measure

Parameters
----------
X: 2d array, shape (n_samples, n_features)
    Data samples.
pairs: 2d array, shape (n_pairs, 2)
    Pairs of data point indices
labels: 1d array, shape (n_pairs)
    Pair label {-1, 1}
proj_mat: 2d array, shape (n_components, n_features), optional (default None)
    Projection matrix to apply before computing Euclidean distance
threshold: float, optional (default None)
    Threshold value on distances. If None given, the score is computed using
    the best threshold.

Returns
-------
score: float
    Classification score. A positive pair of elements is well classified if
    its corresponding distance is smaller than the threshold.
"""
    dists = compute_distances(X, pairs, proj_mat)
    if threshold is not None:
        return compute_score_from_distances(dists, labels, threshold)
    else:
        _, score = compute_best_threshold_from_distances(dists, labels)
        return score


def compute_best_threshold_from_distances(distances, labels):
    """Compute the best threshold value

Parameters
----------
distances: 1d array, shape (n_pairs,)
    Distance values
labels: 1d array, shape (n_pairs,)
    Label associated to the corresponding distance value

Returns
-------
best_threshold: float
    The threshold giving the highest score
best_score: float
    The highest found score
"""
    scores = [(compute_score_from_distances(distances, labels, thresh),
               thresh)
              for thresh in distances]
    best = max(scores)
    return best[0], best[1]


def compute_best_threshold(X, pairs, labels, proj_mat=None):
    """Compute the best threshold value

Parameters
----------
distances: 1d array, shape (n_pairs,)
    Distance values
pairs: 2d array, shape (n_pairs, 2)
    Pairs of data point indices
labels: 1d array, shape (n_pairs)
    Pair label {-1, 1}
proj_mat: 2d array, shape (n_components, n_features), optional (default None)
    Projection matrix to apply before computing Euclidean distance

Returns
-------
best_threshold: float
    The threshold giving the highest score
best_score: float
    The highest found score

"""
    dists = compute_distances(X, pairs, proj_mat)
    return compute_best_threshold_from_distances(dists, labels)


def log_loss(X, beta=1.0):
    """ Generalized logistic loss function

The generaliezed logistic loss function:

    .. math:: \\frac{1}{\\beta} \\log(1 + \\exp(\\beta x))

Parameters
----------
X: array
    Data array
beta: float, optional (default 1.0)
    Smoothness factor. Lower values mean smoother function.
"""
    betaX = X * beta
    # pylint: disable=E1101
    betaX = X * beta
    # pylint: disable=E1101
    result = np.empty_like(X)

    big = betaX >= 40.
    small = betaX < -40.
    mid = ~big & ~small
    result[big] = X[big]
    result[small] = 0
    result[mid] = np.log(1. + np.exp(betaX[mid])) / beta

    return result


def sigmoid(X, beta=1.0):
    """ Generalized sigmoid function

The generaliezed sigmoid function:

    .. math:: (1 + \\exp(-\\beta x))^{-1}

Parameters
----------
X: array
    Data array
beta: float, optional (default 1.0)
    Smoothness factor. Lower values mean smoother function.
"""
    betaX = X * beta
    # pylint: disable=E1101
    result = np.empty_like(X)

    big = betaX >= 40.
    small = betaX < -40.
    mid = ~big & ~small
    result[big] = 1.0
    result[small] = 0.0
    result[mid] = 1.0 / (1 + np.exp(-betaX[mid]))

    return result


def pca(X, n_components):
    """Compute whiten PCA coeficients

Parameters
----------
X: array, shape (n_samples, n_features)
    Data array
n_components: int
    Dimension of the projection space

Returns
-------
coefs: 2d array, shape (n_components, n_features)
    Whiten PCA coefficients
"""
    # pylint: disable=E1101
    cov = np.cov(X.T)
    w, u = la.eigsh(cov, k=n_components, which="LA")
    return (u / np.sqrt(w)).T
    #return u.T
    


class MatThreshold(lgbopt.MultiOptVar):
    """Matrix and threshold optimization variable"""
    def __init__(self, matrix, threshold):
        super(MatThreshold, self).__init__(
            matrix, threshold,
            inner=[matrix_dot, threshold_dot]
        )


def matrix_dot(mat1, mat2):
    """Inner product for matrices

Compute the inner product between matrices as:

    .. math:: \\mathrm{Trace}(M_1^T M_2)

Parameters
----------
mat1: 2d array
   First matrix
mat2: 2d matrix
    Second matrix

Returns
-------
dot: float
    The matrix inner product.
"""
    return (mat1 * mat2).sum()


def threshold_dot(thresh1, thresh2):
    """Simple float product"""
    return thresh1 * thresh2


# pylint: disable=R0902
class DML(object):
    """Distance Metric Learning

Base object for Distance Metric Learning.

Parameters
----------
n_components: int
    Dimension of the output projection space

loss_func: object
    Object representing the loss function. It should be called as:
    ``loss_func(x, labels)`` where x is typically the squared distances
    minus the threshold.
    It should also implement a ``derivative method``:
    ``loss_func.derivative(x, labels)``

alpha: float, optional (default 0.0)
    Regularization factor. If greater than 0.0, the following factor is added
    to the loss function:
        ``alpha * Trace(LL^T)``

fit_threshold: bool, optional (default True)
    Does the distance threshold need to be fitted.

kernel: bool, optional (default False)
    Is the data to fit a kernel ?

verbose: bool, optional (default False)
    If True, display some information about the convergence.

callback: callable, optional (default None)
    Function called at each iteration. Signature:
    ``callback(x)``

**opt_args: key word parameters
    Parameters passed to the ``lgbopt.fmin_gd`` function.
"""
    # pylint: disable=R0913
    def __init__(self, n_components, loss_func, alpha=0.0,
                 fit_threshold=True,
                 kernel=False,
                 verbose=False, callback=None, **opt_args):
        self.n_components = n_components
        self.loss_func = loss_func
        self.fit_threshold = fit_threshold
        self.alpha = alpha
        self.kernel = kernel
        self.verbose = verbose
        self.callback = callback
        self.opt_args = opt_args

    # pylint: disable=W0201
    def fit(self, X, y):
        """Fit the model

Parameters
----------
X: 2d array, shape (n_samples, n_features)
    Data array
y: 2d array, shape (n_pairs, 3)
    Array containing pairs elements indices and associated label.
    The indices of pairs elements are contained in ``y[:, :2]``, while the
    labels are given by ``y[:, 2]``. Labels are in {-1, 1}. The elements
    of the i-th pair are ``X[y[i, 0]]`` and ``X[y[i, 1]]``.
"""
        # pylint: disable=E1101
        X = np.asarray(X)
        y = np.asarray(y)
        pairs = y[:, :2].astype("int")
        labels = y[:, 2]
        max_itr = 100

        # initialize with whiten PCA
        L0 = pca(X, self.n_components)

        # Normalize to have unit average distance
        L0 /= np.sqrt(compute_distances(X, pairs, proj_mat=L0).mean())
        b0 = 1.0

        self._diff = compute_differences(X, pairs)
        self._labels = labels

        n_pairs = len(pairs)
        n_samples, self.n_features = X.shape

        if self.kernel:
            self._U = np.zeros((n_pairs, n_samples))
            self._U[range(n_pairs), pairs[:, 0]] = 1.0
            self._U[range(n_pairs), pairs[:, 1]] = -1.0
            self._U = csr_matrix(self._U)
            # max_itr = 1000

        x0 = MatThreshold(L0, b0)

        if self.callback is not None:
            self.callback(x0)

        if True: # self.alpha > 0:
            eta = 0.1
            _iter = 0
            eps = 1e-6
            L0, threshold = x0
            while True:
                f0 = self._compute_obj(x0,X)
                df0 = self._compute_grad(x0)
                x1 = x0 - eta*df0
                f1 = self._compute_obj(x1,X)
                L1, threshold = x1
                if f1 > f0:
                    eta = eta * 0.9
                    if eta < 1e-50:
                        print "iter:", _iter, "fval", f1
                        break
                    else:
                        continue
                else:
                    df, threshold = self._compute_grad(x1)
                    norm_dfx = np.sqrt(np.inner(df,df).sum())
                    eta = 2*(f0 - f1)/norm_dfx
                    # eta = eta * 1.1
                if f0 - f1 < eps and ( ((L0-L1) ** 2).sum() / (L0 ** 2).sum() ) < eps:
                    print "iter:", _iter, "fval", f1
                    break
                if _iter%100 == 0:
                    print "iter:", _iter, "fval", f1
                x0 = x1
                L0 = L1
                _iter += 1
                if _iter > max_itr:
                    break
            L, threshold = x0
        # else:
        # L, self.threshold_ = lgbopt.fmin_gd(
        #     self._compute_obj, self._compute_grad, x0,
        #     maxiter=max_itr,verbose=True,
        #     callback=self.callback, inner=MatThreshold.dot,augs=X,
        #     **self.opt_args)[0]

        self.coefs_ = L.T

        del self._diff, self._labels
        if self.kernel:
            del self._U
        return self

    def _compute_obj(self, x, K):
        """Compute the objective function"""
        L, threshold = x
        # pylint: disable=E1101
        dists = np.dot(self._diff, L.T)
        dists = (dists ** 2).sum(-1)

        loss = self.loss_func(dists - threshold, self._labels).sum()
        if self.alpha > 0.0:
            # loss += self.alpha * (L ** 2).sum()
            regterm = np.trace(np.dot(np.dot(L, K), L.T))
            loss += self.alpha * regterm
        # loss = loss.mean()
        if self.verbose:
            print "fval:", loss
        return loss

    def _compute_grad(self, x):
        """Compute the gradient of the objective function"""

        L, threshold = x
        # pylint: disable=E1101
        proj = np.dot(self._diff, L.T)
        dists = (proj ** 2).sum(-1)
        dloss = self.loss_func.derivative(dists - threshold, self._labels)

        if self.kernel:
            dL = (proj.T * dloss) * self._U
        else:
            dL = np.dot(proj.T * dloss, self._diff)

        if self.alpha > 0.0:
            dL += self.alpha * L

        # dL /= len(self._diff)
        dL *= 2

        if self.fit_threshold:
            dthres = -dloss.mean()
        else:
            dthres = 0.0

        dx = MatThreshold(dL, dthres)

        if self.verbose:
            print "|grad|", np.sqrt(MatThreshold.dot(dx, dx))

        return MatThreshold(dL, dthres)

    def score(self, X, y):
        """Compute the pairs classification score

Parameters
----------
X: 2d array, shape (n_samples, n_features)
    Data array
y: 2d array, shape (n_pairs, 3)
    Array containing pairs elements indices and associated label.
    The indices of pairs elements are contained in ``y[:, :2]``, while the
    labels are given by ``y[:, 2]``. Labels are in {-1, 1}. The elements
    of the i-th pair are ``X[y[i, 0]]`` and ``X[y[i, 1]]``.
"""
        # pylint: disable=E1101
        X = np.asarray(X)
        y = np.asarray(y)
        pairs = y[:, :2].astype("int")
        labels = y[:, 2]

        return compute_score(X, pairs, labels, self.coefs_.T,
                             self.threshold_)

    def transform(self, X):
        return np.dot(X, self.coefs_)

# pylint: disable=R0903
class PCCALoss(object):
    """The loss function of the PCCA model

Parameters
----------
beta: float, optional (default 3.0)
    Smoothness factor
"""
    def __init__(self, beta=3.0):
        self.beta = beta

    def __call__(self, x, y):
        """Compute the loss function value"""
        return log_loss(y * x, self.beta)

    def derivative(self, x, y):
        """Compute the derivative of the loss function"""
        return y * sigmoid(y * x, self.beta)


class PCCA(DML):
    """Pairwise Constrained Component Analysis

Parameters
----------
n_components: int
    Dimension of the output projection space

beta: float, optional (default 3.0)
    The smoothness factor of the generalized logistic loss function

alpha: float, optional (default 0.0)
    Regularization factor. If greater than 0.0, the following factor is added
    to the loss function:
        ``alpha * Trace(LL^T)``

kernel: bool, optional (default False)
    Is the data to fit a kernel ?

verbose: bool, optional (default False)
    If True, display some information about the convergence.

callback: callable, optional (default None)
    Function called at each iteration. Signature:
    ``callback(x)``

**opt_args: key word parameters
    Parameters passed to the ``lgbopt.fmin_gd`` function.

"""
    # pylint: disable=R0913
    def __init__(self, n_components, beta=3.0, alpha=0.0,
                 kernel=False,
                 verbose=False, callback=None, **opt_args):
        super(self.__class__, self).__init__(
            n_components, alpha=alpha,
            loss_func=PCCALoss(beta),
            fit_threshold=False, kernel=kernel,
            verbose=verbose, callback=callback, **opt_args
        )


class CrossValCallback(object):
    """Callback function to print cross validation score"""
    def __init__(self, X, y, step=1):
        # pylint: disable=E1101
        self.X = np.asarray(X)
        self.y = np.asarray(y)
        self.pairs = self.y[:, :2].astype("int")
        self.labels = self.y[:, 2]

        self.step = step

        self._niter = 0

    def __call__(self, x):
        if self._niter % self.step == 0:
            L, b = x
            print "score: ", compute_score(
                self.X, self.pairs, self.labels, L, b
            )
        self._niter += 1


class Mat2Threshold(lgbopt.MultiOptVar):
    """Matrix and threshold optimization variable"""
    def __init__(self, matrix1, matrix2, threshold):
        super(Mat2Threshold, self).__init__(
            matrix1, matrix2, threshold,
            inner=[matrix_dot, matrix_dot, threshold_dot]
        )


# pylint: disable=R0902
class CrossModalDML(object):
    """Distance Metric Learning

Base object for Distance Metric Learning.

Parameters
----------
n_components: int
    Dimension of the output projection space

loss_func: object
    Object representing the loss function. It should be called as:
    ``loss_func(x, labels)`` where x is typically the squared distances
    minus the threshold.
    It should also implement a ``derivative method``:
    ``loss_func.derivative(x, labels)``

alpha: float, optional (default 0.0)
    Regularization factor. If greater than 0.0, the following factor is added
    to the loss function:
        ``alpha * Trace(LL^T)``

fit_threshold: bool, optional (default True)
    Does the distance threshold need to be fitted.

kernel: bool, optional (default False)
    Is the data to fit a kernel ?

verbose: bool, optional (default False)
    If True, display some information about the convergence.

callback: callable, optional (default None)
    Function called at each iteration. Signature:
    ``callback(x)``

**opt_args: key word parameters
    Parameters passed to the ``lgbopt.fmin_gd`` function.
"""
    # pylint: disable=R0913
    def __init__(self, n_components, loss_func, alpha=0.0,
                 fit_threshold=True,
                 kernel=False,
                 verbose=False, callback=None, **opt_args):
        self.n_components = n_components
        self.loss_func = loss_func
        self.fit_threshold = fit_threshold
        self.alpha = alpha
        self.kernel = kernel
        self.verbose = verbose
        self.callback = callback
        self.opt_args = opt_args

    # pylint: disable=W0201
    def fit(self, X, y):
        """Fit the model

Parameters
----------
X: 2d array, shape (n_samples, n_features)
    Data array
y: 2d array, shape (n_pairs, 3)
    Array containing pairs elements indices and associated label.
    The indices of pairs elements are contained in ``y[:, :2]``, while the
    labels are given by ``y[:, 2]``. Labels are in {-1, 1}. The elements
    of the i-th pair are ``X[y[i, 0]]`` and ``X[y[i, 1]]``.
"""
        # pylint: disable=E1101
        X1 = np.asarray(X[0])
        X2 = np.asarray(X[1])
        y = np.asarray(y)
        pairs = y[:, :2].astype("int")
        labels = y[:, 2]

        # initialize with whiten PCA
        L0 = pca(np.hstack(X), self.n_components)
        A0 = L0[:, :X1.shape[1]]
        B0 = L0[:, X1.shape[1]:]

        # Normalize to have unit average distance
        dist = np.sqrt(compute_cm_distances(X1, X2, pairs,
                                            proj_mat1=A0,
                                            proj_mat2=B0).mean())
        b0 = 1.0
        A0 /= dist
        B0 /= dist

        self._labels = labels

        n_pairs = len(pairs)
        n_samples, self.n_features1 = X1.shape

        if self.kernel:
            self._U = np.zeros((n_pairs, n_samples))
            self._U[range(n_pairs), pairs[:, 0]] = 1.0
            self._U = csr_matrix(self._U)

            self._V = np.zeros((n_pairs, n_samples))
            self._V[range(n_pairs), pairs[:, 1]] = 1.0
            self._V = csr_matrix(self._V)

        self._X1 = X1
        self._X2 = X2
        self._pairs = pairs

        x0 = Mat2Threshold(A0, B0, b0)

        if self.callback is not None:
            self.callback(x0)

        A, B, self.threshold_ = lgbopt.fmin_gd(
            self._compute_obj, self._compute_grad, x0,
            callback=self.callback, inner=Mat2Threshold.dot,
            **self.opt_args)[0]

        self.coefs_ = [A.T, B.T]

        del self._labels, self._X1, self._X2, self._pairs
        if self.kernel:
            del self._U, self._V
        return self

    def _compute_obj(self, x):
        """Compute the objective function"""
        A, B, threshold = x
        # pylint: disable=E1101
        dists = np.dot(self._X1, A.T) - np.dot(self._X2, B.T)
        dists = (dists ** 2).sum(-1)

        loss = self.loss_func(dists - threshold, self._labels).mean()
        if self.alpha > 0.0:
            loss += self.alpha * (A ** 2).sum()
            loss += self.alpha * (B ** 2).sum()
        loss = loss.mean()
        if self.verbose:
            print "fval:", loss
        return loss

    def _compute_grad(self, x):
        """Compute the gradient of the objective function"""

        A, B, threshold = x

        # pylint: disable=E1101
        proj = (np.dot(self._X1[self._pairs[:, 0]], A.T)
                - np.dot(self._X2[self._pairs[:, 1]], B.T))
        dists = (proj ** 2).sum(-1)
        dloss = self.loss_func.derivative(dists - threshold, self._labels)

        if self.kernel:
            dA = (proj.T * dloss) * self._U
            dB = -(proj.T * dloss) * self._V
        else:
            dA = np.dot(proj.T * dloss, self._X1)
            dB = -np.dot(proj.T * dloss, self._X2)

        dA /= len(self._X1)
        dB /= len(self._X1)

        if self.alpha > 0.0:
            dA += self.alpha * A
            dB += self.alpha * B

        dA *= 2
        dB *= 2

        if self.fit_threshold:
            dthres = -dloss.mean()
        else:
            dthres = 0.0

        dx = Mat2Threshold(dA, dB, dthres)

        if self.verbose:
            print "|grad|", np.sqrt(Mat2Threshold.dot(dx, dx))

        return dx

    def score(self, X, y):
        """Compute the pairs classification score

Parameters
----------
X: 2d array, shape (n_samples, n_features)
    Data array
y: 2d array, shape (n_pairs, 3)
    Array containing pairs elements indices and associated label.
    The indices of pairs elements are contained in ``y[:, :2]``, while the
    labels are given by ``y[:, 2]``. Labels are in {-1, 1}. The elements
    of the i-th pair are ``X[y[i, 0]]`` and ``X[y[i, 1]]``.
"""
        # pylint: disable=E1101
        X1 = np.asarray(X[0])
        X2 = np.asarray(X[1])
        y = np.asarray(y)
        pairs = y[:, :2].astype("int")
        labels = y[:, 2]

        return compute_cm_score(X1, X2, pairs, labels,
                                self.coefs_[0].T,
                                self.coefs_[1].T,
                                self.threshold_)


def compute_cm_distances(X1, X2, pairs, proj_mat1, proj_mat2, squared=True):
    """Compute distances between pairs of data points

Parameters
----------
X1: 2d array, shape (n_samples, n_features)
    First data array

X1: 2d array, shape (n_samples, n_features)
    Second data array

pairs: 2d array, shape (n_pairs, 2)
    Pair of element indices

proj_mat1: 2d array, shape (n_components, n_features)
    Projection matrix applied to the first data points before computing the
    Euclidean distance.

proj_mat2: 2d array, shape (n_components, n_features)
    Projection matrix applied to the second data points before computing the
    Euclidean distance.

squared: bool, optional (default True)
    If True, returns the squared Euclidean distance, otherwise, returns
    Euclidean distance

Returns
-------
distances: 2d array, shape (n_pairs,)
    Euclidean distance (squared if requested) between pairs of data points.
"""
    # pylint: disable=E1101
    diff = (np.dot(X1[pairs[:, 0]], proj_mat1.T)
            - np.dot(X2[pairs[:, 1]], proj_mat2.T))
    dists = (diff ** 2).sum(-1)
    return dists if squared else np.sqrt(dists)


def compute_cm_score(X1, X2, pairs, labels, proj_mat1, proj_mat2,
                     threshold=None):
    """Compute pairs classification score based on distance measure

Parameters
----------
X1: 2d array, shape (n_samples, n_features)
    First data array

X1: 2d array, shape (n_samples, n_features)
    Second data array

pairs: 2d array, shape (n_pairs, 2)
    Pairs of data point indices

labels: 1d array, shape (n_pairs)
    Pair label {-1, 1}

proj_mat1: 2d array, shape (n_components, n_features)
    Projection matrix applied to the first data points before computing the
    Euclidean distance.

proj_mat2: 2d array, shape (n_components, n_features)
    Projection matrix applied to the second data points before computing the
    Euclidean distance.

threshold: float, optional (default None)
    Threshold value on distances. If None given, the score is computed using
    the best threshold.

Returns
-------
score: float
    Classification score. A positive pair of elements is well classified if
    its corresponding distance is smaller than the threshold.
"""
    dists = compute_cm_distances(X1, X2, pairs, proj_mat1, proj_mat2)
    if threshold is not None:
        return compute_score_from_distances(dists, labels, threshold)
    else:
        _, score = compute_best_threshold_from_distances(dists, labels)
        return score
