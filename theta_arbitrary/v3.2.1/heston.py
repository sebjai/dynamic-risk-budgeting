# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 17:01:38 2023

@author: sebja
"""

import numpy as np
from scipy.stats import chi2
from scipy.stats import t as t_rv
from scipy.stats import norm
import pdb

class  heston():
    
    def __init__(self, n_assets = 3, T = 12, dt=1/12):
        
        self.n = n_assets
        self.T = T
        self.kappa = np.linspace(4, 6, self.n).reshape(1,-1)
        self.theta = ((np.linspace(0.1, 0.3,self.n))**2).reshape(1,-1)
        self.eta = np.linspace(0.5, 2, self.n).reshape(1,-1)
        self.mu = np.linspace(0.05, 0.08, self.n).reshape(1,-1)
        self.rho_xx = 0.3
        self.rho_xv = -0.5
        self.S0 = np.ones(self.n).reshape(1,-1)
        self.df = 4
        self.dt = dt
        
        self.cov = np.eye((2*self.n))
        for i in range(self.n):
            for j in range(self.n):
                
                if i < j:
                    self.cov[i,j] = self.rho_xx
                    self.cov[j,i] = self.rho_xx
                
            self.cov[i, i+self.n] = self.rho_xv
            self.cov[i+self.n, i] = self.rho_xv
                
        
        
    def simulate(self, n_sims=256):
        
        X = np.zeros((n_sims, self.T+1, self.n))
        v = np.zeros((n_sims, self.T+1, self.n))
        v[:,0,:] = self.theta.reshape(1,-1)
        
        zeros = np.zeros((2*self.n))
        
        for t in range(self.T):
            
            Z = np.random.multivariate_normal(zeros, self.cov, size=n_sims)
            s = chi2.rvs(self.df, size=n_sims)[:, np.newaxis]
            
            dW_X = np.sqrt(self.dt) * norm.ppf(t_rv.cdf(np.sqrt(self.df/s)*Z[:,:self.n], self.df))
            dW_v = np.sqrt(self.dt) * Z[:,self.n:]
            
            vp = np.maximum(v[:,t,:],0)
            X[:,t+1,:] = X[:,t,:] + (self.mu - 0.5*vp) * self.dt + np.sqrt(vp) * dW_X
            v[:,t+1,:] = self.theta \
                + (vp - self.theta)*np.exp(-self.kappa*self.dt) \
                    + self.eta * np.sqrt(vp) * dW_v \
                        + 0.25*self.eta**2*(dW_v**2-self.dt)
            v[:,t+1,:] = np.maximum(v[:,t+1,:],0)
            
        S = self.S0.reshape(1,1,-1) * np.exp(X)
        
        return S, v