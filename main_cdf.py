# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 16:07:49 2022

@author: sebja
"""

import numpy as np
import torch

from cdf import cdf

class Sim():
    
    def Simulate(self, nsims = 500):
        
        X = torch.zeros((nsims,2))
        Y = torch.zeros((nsims,2))
        
        Y[:,0] = torch.randn(nsims)
        X[:,0] = 2*Y[:,0] + torch.randn(nsims) 
        
        Y[:,1] = 2*Y[:,0] + torch.randn(nsims)
        X[:,1] = -4*Y[:,1] + X[:,0] + torch.randn(nsims) 

        return X, Y 

sim = Sim()
X, Y = sim.Simulate(100)

model = cdf(sim)
model.Learn(n_iter=10_000)