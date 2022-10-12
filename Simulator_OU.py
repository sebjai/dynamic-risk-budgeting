# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 14:49:15 2022

@author: sebja
"""

import numpy as np

class  Simulator_OU():
    
    def __init__(self, n_assets = 3, T = 12, dt=1/12):
        
        self.n = n_assets
        self.T = T
        self.kappa = np.linspace(1, 2, self.n).reshape(1,-1)
        self.sigma = np.linspace(0.2, 0.3, self.n).reshape(1,-1)
        self.mu = np.linspace(0.05, 0.08, self.n).reshape(1,-1)
        self.rho = -0.3
        self.S0 = np.ones(self.n).reshape(1,-1)
        
        self.dt = dt
        
    def Simulate(self, Nsims=256):
        
        X = np.zeros((Nsims, self.T+1, self.n))

        mean = np.zeros((self.n,))
        cov = self.rho*np.ones((self.n,self.n)) + (1-self.rho) * np.eye(self.n)
        
        for i in range(self.T):
            
            dW = np.sqrt(self.dt) * np.random.multivariate_normal(mean, cov, size=Nsims)
            
            X[:, i+1, :] = X[:,i,:] * np.exp(-self.kappa*self.dt) + self.sigma * dW
            
        t = self.dt*np.linspace(1e-10, self.T, self.T+1).reshape(1,-1, 1)
        g = self.mu.reshape(1,1,-1) * t - self.sigma.reshape(1,1,-1)**2*(1-np.exp(-2*self.kappa.reshape(1,1,-1)*t))/(4.0*self.kappa.reshape(1,1,-1))        
        
        S = self.S0.reshape(1,1,-1) * np.exp( g + X)
        
        return S