# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 16:39:15 2022

@author: sebja
"""

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.interpolate import interp1d

M = 10_000

alpha= 0.05

Z = np.random.rand(M)
Y = np.random.rand(M)

X = (Z>alpha) * (1+Y)

plt.hist(X, bins=np.linspace(0,2,101), density=True)
plt.show()

F = ECDF(X)

U = F(X)

plt.title('Using ECDF')
plt.scatter(U, X, s= 0.1)
plt.xlabel(r'$U$')
plt.ylabel(r'$X$')
plt.show()

#%%
u = np.linspace(0.04,0.046,10001)
qtl = np.quantile(X, u)

plt.title('With Quantile')
plt.scatter(u, qtl, s=0.1)
plt.xlabel(r'$U$')
plt.ylabel(r'$X$')
plt.show()