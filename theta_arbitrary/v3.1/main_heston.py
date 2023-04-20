# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 17:21:10 2023

@author: jaimunga
"""

from heston import heston
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

#%%
env=heston(n_assets=2, T=3)
S, v= env.simulate(10_000)

def kde(x, z):
    
    h = 1.06*np.std(x)*len(x)**(-1/5)
    
    f = lambda a : np.sum(norm.pdf((a-x)/h)/h/len(x))
    
    pdf = np.zeros(len(z))
    for i in range(len(z)):
        pdf[i] = f(z[i])
        
    return pdf
        

def plot(x, ylim):
    
    qtl = np.quantile(x,[0.1,0.5,0.9], axis=0)
    for i in range(env.n):
        
        plt.subplot(1,env.n, i+1)
        plt.plot(x[:100,:,i].T, alpha=0.5)
        
        plt.plot(qtl[:,:,i].T, color='k')
        plt.ylim(ylim)
    plt.tight_layout()
    plt.show()
    
    z = np.linspace(ylim[0],ylim[1],501)
    for i in range(env.n):
        
        f = kde(x[:,-1,i], z)
        plt.hist(x[:,-1,i].T, alpha=0.5, density=True, bins=np.linspace(ylim[0],ylim[1],51))
        plt.plot(z, f, label= r'$Asset-{0:1d}$'.format(i))
    
    plt.xlabel(r'$\log(S_T^i/S_0^i)$')
    plt.legend()
    plt.show()    
    

plot(np.log(S),(-0.8,0.6))
plot(v,(-0.2,0.5))