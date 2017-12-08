import dml
import unittest
import numpy as np
import csv


X = np.loadtxt('x.out')
C = np.loadtxt('c.out')
y = np.loadtxt('y.out')
y = np.hstack([C, y[:, np.newaxis]])

callback = dml.CrossValCallback(X, y)

ml = dml.PCCA(40, callback=None,
              alpha=0,
              kernel=True, verbose=False)
ml.fit(X, y)
print ml.coefs_.shape
np.savetxt('projectionMatrix.out', np.asarray(ml.coefs_))