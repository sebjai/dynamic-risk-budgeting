# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 14:04:28 2022

@author: sebja
"""

from Simulator_OU import Simulator_OU
import numpy as np
import torch

class Environment():
    
    def __init__(self, Simulator : Simulator_OU, trans_cost = 0):
        
        self.Simulator = Simulator
        self.T = self.Simulator.T
        self.n = self.Simulator.n
        self.X0 = 1
        
    def Generate_Epoch(self, theta):
        
        nsims = theta.shape[0]
        
        # grab a simulation of the market prices
        S = self.Simulator.Simulate(nsims)
        
        # stores wealth process
        X = torch.zeros((nsims, self.T+1))
        X[:,0] = self.X0
        
        # number of each assets (as theta represents percentage in wealth)
        alpha = torch.zeros((nsims, self.T, self.n))
        alpha[:,0,:] = theta[:,0,:] * X[:, 0].reshape(-1,1) / S[:,0,:]
        
        for i in range(1, self.T+1):
            
            # new wealth at period end
            X[:,i] = torch.sum( alpha[:,i-1,:] * S[:,i,:], axis=1)
            
            # rebalance assets
            alpha[:,i,:] = theta[:,i,:] * X[:,i].reshape(-1,1) / S[:,i,:]    
            
        costs = torch.diff(X, axis=1)            
        
        return S, alpha, X, costs