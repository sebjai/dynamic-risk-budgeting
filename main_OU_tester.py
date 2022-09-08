# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 13:47:32 2022

@author: sebja
"""

from Simulator_OU import Simulator_OU

import matplotlib.pyplot as plt
import numpy as np

#%%

OU = Simulator_OU(n_assets=2, T=2)
X = OU.Simulate()

fig, ax = plt.subplots(X.shape[-1])

t = np.linspace(0,OU.T,100)*OU.dt

for i in range(X.shape[-1]):
    ax[i].plot(np.linspace(0,OU.T,OU.T+1)*OU.dt, X[:,:,i].T, alpha=0.25, linewidth=0.5)
    ax[i].plot(t, np.exp(OU.mu[0,i]*t),color='k')
    ax[i].set_xticks([0, 1*OU.dt, 2*OU.dt])

plt.tight_layout()
plt.show()