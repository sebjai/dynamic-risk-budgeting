# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 15:50:26 2022

@author: sebja
"""

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np

import pdb

from scipy.stats import norm

class Net(nn.Module):
    
    def __init__(self, n_in, n_out, nNodes, nLayers):
        super(Net, self).__init__()
        
        # single hidden layer
        self.prop_in_to_h = nn.Linear( n_in, nNodes)
        
        self.prop_h_to_h = nn.ModuleList([nn.Linear(nNodes, nNodes) for i in range(nLayers-1)])
            
        self.prop_h_to_out = nn.Linear(nNodes, n_out)
        
        self.g = nn.SiLU()
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, Z):
        
        h = self.g(self.prop_in_to_h(Z))
            
        for prop in self.prop_h_to_h:
            h = self.g(prop(h))
        
        output = self.sigmoid(self.prop_h_to_out(h))
        
        return output
    
class cdf():
    
    def __init__(self, Simulator):
        
        self.F = Net(3, 1, 32, 5)
        self.F_optimizer = optim.AdamW(self.F.parameters(), lr=0.001)
        
        self.z = torch.linspace(-20, 20, 101).reshape(1, 1, -1)
        self.dz = self.z[0,0,1]-self.z[0,0,0]
        
        self.t = torch.linspace(0,1,2).reshape(1,-1,1)
        
        self.Simulator = Simulator
        
        self.loss = []
        
    def __Score__(self, X, Y):
        
        y = Y.reshape(-1,2,1).repeat(1,1,self.z.shape[2])
        x = X.reshape(-1,2,1).repeat(1,1,self.z.shape[2])
        
        z = self.z.repeat(Y.shape[0], Y.shape[1], 1)
        t = self.t.repeat(Y.shape[0], 1, z.shape[2])
        
        Z = torch.concat((y.unsqueeze(axis=3), 
                          t.unsqueeze(axis=3),
                          z.unsqueeze(axis=3)), axis=3)
        
        F = self.F(Z)
        
        score = torch.mean( torch.sum((F[...,0]-1.0*(z>=x))**2*self.dz, axis=2) )
        
        eps = 0.001
        z_plus_eps = z + eps
        Z_eps = torch.concat((y.unsqueeze(axis=3), 
                          t.unsqueeze(axis=3),
                          z_plus_eps.unsqueeze(axis=3)), axis=3)        
        
        F_eps = self.F(Z_eps)
        dF = (F_eps-F)/eps
        
        convexity_error = torch.mean(torch.sum((dF*(dF<0)), axis=2))
        
        return score, convexity_error
        
    def Learn(self, n_iter=1_000, batch_size = 1024, n_print=50):
        
        a = 1.01
        mu = 1
        lam = 1
        
        self.lam = []
        self.mu = []
        self.error = []
        
        for i in tqdm(range(n_iter)):
            
            self.F_optimizer.zero_grad()
            
            X, Y = self.Simulator.Simulate(batch_size)
            
            score, convexity_error = self.__Score__(X, Y)
            
            loss = score + lam*convexity_error + 0.5*mu*convexity_error**2

            self.lam.append(lam)
            self.mu.append(mu)
            self.error.append(convexity_error.item())
            
            if np.mod(i,10) == 0:
                lam += mu * self.error[-1]
                if np.abs(self.error[-1])>0:
                    mu *= a
            
            loss.backward()
            
            self.F_optimizer.step()
            
            self.loss.append(loss.item())
            
            if np.mod(i, n_print) == 0:
                self.Plot()
                
    def Plot(self):
        
        fig, ax = plt.subplots(1,3, figsize=(10,3))
        
        ax[0].plot(self.loss)
        ax[0].set_yscale('log')
        
        y = torch.linspace(-5,5, 5).reshape(-1,1,1)
        y = y.repeat(1,1,self.z.shape[2])
        
        z = self.z.repeat(y.shape[0], y.shape[1], 1)
        
        ones = torch.ones(y.shape[0], 1, z.shape[2])
        
        for j in range(2):
            
            Z = torch.concat((y.unsqueeze(axis=3), 
                              j*ones.unsqueeze(axis=3),
                              z.unsqueeze(axis=3)), axis=3)
            
            F = self.F(Z)            
            
            for k in range(y.shape[0]):
                ax[j+1].plot(z[k,:,:].detach().numpy().squeeze(),
                         F[k,:,:].detach().numpy().squeeze(),
                         linewidth=1,
                         label= r'$'+ str(y[k,0,0].detach().numpy()) +'$')
                
            ax[j+1].set_title(r'$t=' + str(j) + '$')
            ax[j+1].legend(fontsize=8)
            
        z_np = z[0,0,:].numpy()
        F0 = lambda z, y: norm.cdf(z-2*y)
        for k in range(y.shape[0]):
            ax[1].plot(z_np, F0(z_np, y[k,0,0].numpy()), linestyle='--', color='k')
        
        F1 = lambda z, y: norm.cdf((z+3*y)/np.sqrt(3))
        for k in range(y.shape[0]):
            ax[2].plot(z_np, F1(z_np, y[k,0,0].numpy()), linestyle='--', color='k')
                
        plt.tight_layout()
        plt.show()
        