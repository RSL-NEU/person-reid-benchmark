# -*- coding: utf-8 -*-
"""
Created on Fri Feb 13 21:35:44 2015

@author: alexis
"""

import dml
import unittest
import numpy as np
import csv


class TestDML(unittest.TestCase):
    def setUp(self):
        self.callback = False
        self.verbose = False

    def test_pcca(self):
        rng = np.random.RandomState(1)
        #X = rng.randn(10, 20)        
        #pairs = [(i, i + 1) for i in range(5)] + [(i, i + 5) for i in range(5)]
        #labels = np.asarray([1] * 5 + [-1] * 5)

        #y = np.hstack([pairs, labels[:, np.newaxis]])
        X=np.loadtxt('x.out');
        C=np.loadtxt('c.out');
        y=np.loadtxt('y.out');
        y=np.hstack([C, y[:, np.newaxis]])
        #print X,pairs,labels,y
        callback = dml.CrossValCallback(X, y) if self.callback else None
        ml = dml.PCCA(40, callback=callback,
                      verbose=self.verbose)
        ml.fit(X, y)

        print ml.coefs_.shape
        np.savetxt('projectionMatrix.out',np.asarray(ml.coefs_))
        #assert(ml.score(X, y) == 1.0)

    def test_kernel_pcca(self):
        rng = np.random.RandomState(1)
        X = rng.randn(10, 5)
        X = np.dot(X, X.T)
        pairs = [(i, i + 1) for i in range(5)] + [(i, i + 5) for i in range(5)]
        labels = np.asarray([1] * 5 + [-1] * 5)

        y = np.hstack([pairs, labels[:, np.newaxis]])
        callback = dml.CrossValCallback(X, y) if self.callback else None
        ml = dml.PCCA(3, callback=callback, kernel=True,
                      verbose=self.verbose)
        ml.fit(X, y)

        assert(ml.score(X, y) == 1.0)

if __name__ == "__main__":
    unittest.main()
