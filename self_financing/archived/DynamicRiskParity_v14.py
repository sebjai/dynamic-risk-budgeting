# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 8:07:00 2022

@author: jaimunga
"""

import torch
import torch.nn as nn
import torch.optim as optim

from Simulator_OU import Simulator_OU
from tqdm import tqdm

import pdb
import numpy as np
import matplotlib.pyplot as plt

import copy


class Net(nn.Module):
    
    def __init__(self, 
                 n_in, 
                 n_out, 
                 nNodes, 
                 nLayers, 
                 out_activation=None, 
                 device='cpu'):
        super(Net, self).__init__()
        
        self.device = device
        
        # single hidden layer
        self.prop_in_to_h = nn.Linear( n_in, nNodes).to(self.device)
        
        self.prop_h_to_h = nn.ModuleList([nn.Linear(nNodes, nNodes).to(self.device) for i in range(nLayers-1)])
            
        self.prop_h_to_out = nn.Linear(nNodes, n_out).to(self.device)
        
        self.g = nn.SiLU()
        
        self.out_activation = out_activation
        
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, h, Y):
        
        Y_flat = Y.flatten(start_dim=-2)
        
        h = torch.cat((h.flatten(start_dim=-2), 
                       Y_flat), 
                      axis=-1)
        
        h = self.g(self.prop_in_to_h(h))
            
        for prop in self.prop_h_to_h:
            h = self.g(prop(h))
        
        # hidden layer to output layer
        output = self.prop_h_to_out(h)
        
        if self.out_activation == 'softplus':
            output = self.softplus(output)
        elif self.out_activation == 'sigmoid':
            output = self.sigmoid(output)
        
        return output
    
class betaNet(nn.Module):
    
    def __init__(self, 
                 nIn, 
                 nOut, 
                 gru_hidden=5, 
                 gru_layers=5, 
                 linear_hidden = 36, 
                 linear_layers=5,
                 device='cpu'):
        super(betaNet, self).__init__()

        self.device = device
        self.gru = torch.nn.GRU(input_size=nIn, 
                                hidden_size=gru_hidden, 
                                num_layers=gru_layers, 
                                batch_first=True).to(self.device)
        
        self.gru_to_hidden = nn.Linear(gru_hidden*gru_layers+2+nOut, linear_hidden).to(self.device)
        self.linear_hidden_to_hidden = nn.ModuleList([nn.Linear(linear_hidden, linear_hidden).to(self.device) for i in range(linear_layers-1)])
        self.hidden_to_out = nn.Linear(linear_hidden, nOut).to(self.device)
        
        self.g = nn.SiLU()
        self.softmax = nn.Softmax(dim=1)
        self.softplus = nn.Softplus()
        
    def forward(self, h, Y):
        
        _, h_out = self.gru(Y.clone(), h.clone())
        
        x = torch.cat((h.transpose(0,1).flatten(start_dim=-2), 
                       Y.flatten(start_dim=-2)),
                      axis=-1)
        x = self.gru_to_hidden(x)
        x = self.g(x)
        
        for linear in self.linear_hidden_to_hidden:
            x = self.g(linear(x))
            
        x = self.hidden_to_out(x)
        
        beta = self.softmax(x)
        
        # beta = torch.zeros(x.shape).to(self.device)
        # beta[:,:-1] = torch.tanh(x)[:,:-1]
        # beta[:,-1] = 1.0-torch.sum(beta[:,:-1], axis=1)
        
        return h_out, beta
        
        
class InitialWealthNet(nn.Module):
    
    def __init__(self, device='cpu'):
        super(InitialWealthNet, self).__init__()

        self.device = device
        self.linear = nn.Linear(1, 1).to(self.device)
        
        nn.init.constant_(self.linear.bias.data, 0)
        nn.init.uniform_(self.linear.weight, a=1.0, b=1.0)
        self.g = nn.Softplus()
        self.scale = 1
        
    def forward(self, batch_size):
        
        x = torch.ones((batch_size,1)).to(self.device)
        h = self.scale*self.g(self.linear(x))
        
        return h
    
class DynamicRiskParity():
    
    def __init__(self, Simulator : Simulator_OU, X0=1, B = 0, alpha=0.8, p=0.5):
        
        self.ver = 14
        
        # set the device to use
        if torch.cuda.device_count() > 0:
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')
        
        self.Simulator = Simulator
        self.X0 = X0        
        self.T = Simulator.T
        self.d = Simulator.n
        self.dt = Simulator.dt
        self.alpha = alpha
        self.p = p
        # self.gamma = lambda u : self.p*(u>self.alpha)/(1-alpha) + (1-self.p)
        self.sigmoid = nn.Sigmoid()
        self.gamma = lambda u : self.p*self.sigmoid((u-self.alpha)*5)/(1-alpha) + (1-self.p)
        
        if B == 0:
            self.B = torch.ones((self.T, 1, self.d))/(self.T*self.d)
        else:
            self.B = torch.tensor(B).float()
            
        self.B = self.B.to(self.device)
        
        #
        # the percentage of wealth in each asset: 
        # states = past asset prices (encoded in a GRU), current time and wealth
        #
        self.beta = betaNet(nIn=self.d+2,
                            nOut=self.Simulator.n,
                            gru_hidden=self.Simulator.n,
                            gru_layers=5,
                            linear_hidden=32,
                            linear_layers=5,
                            device=self.device)
        self.beta_optimizer = optim.AdamW(self.beta.parameters(), 
                                          lr = 0.001)
        
        self.wealth_0= InitialWealthNet(device=self.device)
        self.wealth_0_optimizer = optim.AdamW(self.wealth_0.parameters(), 
                                              lr = 0.002)
        
        self.__initialize_CVaR_VaR_Mean_Net__(self.device)
        
        # for storing losses
        self.VaR_CVaR_mean_loss = []
        self.F_loss = []
        self.beta_loss = []
        self.mean_beta = []

        self.eta = 0.1 # the Lagrange multiplier factor        
        
        self.W_0 = []
        self.RC = []
        self.V_0 = []
        
    def __initialize_CVaR_VaR_Mean_Net__(self, device='cpu'):
    
        # Need to elicit both conditional VaR (using network psi) and conditional 
        # CVaR (using network chi)
        # VaR = psi( states )
        # CVaR = psi( states ) + chi( states )
        # the states consists of the output of the gru layers
        
        gru_total_hidden = self.beta.gru.num_layers*self.beta.gru.hidden_size
        n_in = gru_total_hidden+2+self.Simulator.n
        
        # for conditional cdf of c_t + V_{t+1}
        self.F = Net(n_in=n_in+1, n_out=1, nNodes=16, nLayers=3, out_activation='sigmoid', device=self.device)
        self.F_optimizer = optim.AdamW(self.F.parameters(), lr=0.001)
        
        # for conditional mean
        self.mu = Net(n_in=n_in, n_out=1, nNodes=32, nLayers=5, device=self.device)
        self.mu_target = copy.deepcopy(self.mu)
        self.mu_optimizer = optim.AdamW(self.mu.parameters(), lr=0.001)
        
        # for conditional VaR
        self.psi = Net(n_in=n_in, n_out=1, nNodes=32, nLayers=5, device=self.device)
        self.psi_target = copy.deepcopy(self.psi)
        self.psi_optimizer = optim.AdamW(self.psi.parameters(), lr=0.001)
        
        # for increment from conditional VaR to conditional CVaR
        # i.e., CVaR = chi + psi
        self.chi = Net(n_in=n_in, n_out=1, nNodes=32, nLayers=5, out_activation='softplus', device=self.device)
        self.chi_optimizer = optim.AdamW(self.chi.parameters(), lr=0.001)
        self.chi_target = copy.deepcopy(self.chi)
        
        self.VaR = lambda h, Y : self.psi(h, Y)
        self.CVaR = lambda h, Y : self.psi(h, Y) + self.chi(h, Y)
        
        self.risk_measure = lambda h, Y :  (self.p * self.CVaR(h, Y) 
                                            + (1.0-self.p)* self.mu(h, Y))
        
        self.VaR_target = lambda h, Y : self.psi_target(h, Y)
        self.CVaR_target = lambda h, Y : self.psi_target(h, Y) + self.chi_target(h, Y)
        self.risk_measure_target = lambda h, Y :  (self.p * self.CVaR_target(h, Y) 
                                                   + (1.0-self.p)* self.mu_target(h, Y))
        
        
    def __Score__(self, VaR, CVaR, mu, X):
        
        C = 5.0
        A = ((X<=VaR)*1.0-self.alpha)*VaR + (X>VaR)*X
        B = (CVaR+C)*(1.0-self.alpha)
        
        # for conditional VaR & CVaR
        score = torch.mean(torch.log( (CVaR+C)/(X+C)) - (CVaR/(CVaR+C)) + A/B)
        
        # for conditional mean
        score += torch.mean((mu-X)**2)
        
        return score
    
    def __F_Score__(self, h, Y, X):
        
        N = 101
        z = torch.linspace(-2,2,N).reshape(N,1,1,1).repeat(1,Y.shape[0], Y.shape[1], 1).to(self.device)
        dz = z[1,0,0,0]-z[0,0,0,0]              
        
        Z = torch.concat((Y.unsqueeze(axis=0).repeat(N,1,1,1), 
                          z), 
                         axis=3)
        
        F = self.F(h.unsqueeze(axis=0).repeat(N,1,1,1), Z)
        
        score = torch.mean( torch.sum((F-1.0*(z[...,0]>=X.unsqueeze(axis=0).repeat(N,1,1)))**2*dz, 
                                      axis=0) )
        
        # add increasing penalty
        d_dz_F = torch.diff(F, axis=0)/dz
        score += torch.mean( torch.sum(d_dz_F**2*(d_dz_F<0)*dz), axis=0 )
        
        return score
    
    def __RunEpoch__(self, batch_size = 256):
        
        # grab a simulation of the market prices
        S = torch.tensor(self.Simulator.Simulate(batch_size)).float().transpose(0,1).to(self.device)
        
        # number of each asset invested
        beta = torch.zeros((self.T, batch_size, self.d)).to(self.device)
        
        #
        # store the hidden states from all layers and each time
        # h[i, ...] is the hidden for time step t_{i-1}
        #
        h = torch.zeros((self.T+1, 
                         self.beta.gru.num_layers, 
                         batch_size, 
                         self.beta.gru.hidden_size)).to(self.device)
        
        # stores wealth process
        wealth = torch.zeros((self.T+1, batch_size)).to(self.device)
        wealth[0,:] = self.wealth_0(batch_size).reshape(-1)              
        
        # concatenate t_i, wealth, and asset prices
        Y = torch.zeros((self.T, batch_size, 1, self.Simulator.n+2)).to(self.device)
        ones = torch.ones((batch_size,1)).to(self.device)
        zeros = 0*ones
        Y[0,...] = torch.cat( (zeros, 
                               wealth[0,:].reshape(-1,1), 
                               S[0,:,:]), axis=1).unsqueeze(axis=1)
        
        # push through the neural-net to get weights
        h[1,...] , beta[0,:,:] = self.beta(h[0,...], Y[0,...])
    
        # convert to weights to number of shares
        theta = torch.zeros((self.T, batch_size,  self.d)).to(self.device)
        theta[0,:,:] = beta[0,:,:].clone() * wealth[0,:].reshape(-1,1).clone() / S[0,:,:]
            
        for i in range(1, self.T):
            
            # new wealth at period end
            wealth[i,:] = torch.sum( theta[i-1,:,:] * S[i,:,:], axis=1).clone()
            
            # concatenate t_i, and asset prices    
            Y[i,...] = torch.cat(((i*self.dt)*ones, 
                                  wealth[i,:].reshape(-1,1).clone(),
                                  S[i,:,:]), axis=1).unsqueeze(axis=1)
            
            # push through the neural-net to get new number of assets
            h[i+1,...], beta[i,:,:] = self.beta(h[i,...].clone(), Y[i,...].clone())
                
            # rebalance assets holdings
            theta[i,:,:] = beta[i,:,:].clone() * wealth[i,:].reshape(-1,1).clone() / S[i,:,:]  
            
        wealth[-1,:] = torch.sum( theta[-1,:,:] * S[-1,:,:], axis=1).clone()
            
        costs = -torch.diff(wealth, axis=0)         
        
        return costs, h, Y, beta, theta, S, wealth    
        
    
    def Update_ValueFunction(self, N_iter  = 100, batch_size = 256, n_print=100):
        
        count = 0
        for j in range(N_iter):
        
            costs, h, Y, beta, theta, S, wealth = self.__RunEpoch__(batch_size)
            
            # costs = -theta^T \Delta S
            costs = costs.unsqueeze(axis=2)
            
            loss = 0.0
            
            for i in range(0, self.T):

                cost_plus_V_one_ahead = costs[i,:]
                
                if i < self.T-1:
                    cost_plus_V_one_ahead += self.risk_measure_target(h[i+2,...].transpose(0,1), 
                                                                      Y[i+1,...])
                
                loss += self.__Score__(self.VaR(h[i+1,...].transpose(0,1), Y[i,...]),
                                       self.CVaR(h[i+1,...].transpose(0,1), Y[i,...]),
                                       self.mu(h[i+1,...].transpose(0,1), Y[i,...]),
                                       cost_plus_V_one_ahead)

            self.psi_optimizer.zero_grad()
            self.chi_optimizer.zero_grad()
            self.mu_optimizer.zero_grad()

            loss.backward()
            
            self.psi_optimizer.step()
            self.chi_optimizer.step()
            self.mu_optimizer.step()
            
            self.VaR_CVaR_mean_loss.append(loss.item())
            
            self.W_0.append(self.wealth_0(1).item())
            
            count += 1
            if np.mod(count, n_print) == 0:
                
                plt.plot(self.VaR_CVaR_mean_loss)
                plt.yscale('log')
                plt.show()
                
        self.chi_target = copy.deepcopy(self.chi)
        self.psi_target = copy.deepcopy(self.psi)
        self.mu_target = copy.deepcopy(self.mu)
        
        
    def Update_F(self, N_iter  = 100, batch_size = 256, n_print=100):
        
        count = 0
        for j in range(N_iter):
        
            costs, h, Y, beta, theta, S, wealth = self.__RunEpoch__(batch_size)
            
            # costs = -theta^T \Delta S
            costs = costs.unsqueeze(axis=2).detach()
            h = h.detach()
            Y = Y.detach()
            
            loss = 0.0
            
            for i in range(0, self.T):

                cost_plus_V_one_ahead = costs[i,:]
                
                if i < self.T-1:
                    cost_plus_V_one_ahead += self.risk_measure_target(h[i+2,...].transpose(0,1), 
                                                                      Y[i+1,...]).detach()
                
                loss += self.__F_Score__(h[i+1,...].transpose(0,1), 
                                         Y[i,...], 
                                         cost_plus_V_one_ahead)

            self.F_optimizer.zero_grad()
                
            # loss.backward(retain_graph=True)
            loss.backward()
            
            self.F_optimizer.step()
            
            self.F_loss.append(loss.item())
            
            self.W_0.append(self.wealth_0(1).item())
            
            count += 1
            if np.mod(count, n_print) == 0:
                
                plt.plot(self.F_loss)
                plt.yscale('log')
                plt.show()
        
    def __Compute_V_Gamma__(self, costs, h, Y):
        
            batch_size = costs.shape[1]
            
            # compute the value function at points in time            
            V = torch.zeros((self.T+1, batch_size)).to(self.device)
            for j in range(self.T):
                V[j,:] = self.risk_measure_target(h = h[j+1,...].transpose(0,1), 
                                                  Y=Y[j,...]).squeeze()
            
            costs_plus_V_onestep_ahead  = (costs + V[1:,:])

            # evalute the Gamma processes
            U = torch.zeros((self.T, batch_size)).to(self.device)
            
            Z = torch.concat((Y,
                              costs_plus_V_onestep_ahead.unsqueeze(axis=2).unsqueeze(axis=3)), 
                             axis=3)
            
            for j in range(self.T):
                U[j,:] = self.F(h[j+1,...].transpose(0,1), Z[j,...])[:,0]
                
            Gamma = self.gamma(U)
                
            cumprod_Gamma = torch.cumprod(Gamma, axis=0)
            
            return V, Gamma, cumprod_Gamma
        
    def RiskContributions(self, batch_size=2048):
        
        costs, h, Y, beta, theta, S, X = self.__RunEpoch__(batch_size)
        
        V,  Gamma, cumprod_Gamma = self.__Compute_V_Gamma__(costs.detach(),
                                                            h.detach(), 
                                                            Y.detach())
        
        Delta_S = torch.diff(S, axis=0)

        # compute loss
        RC = torch.zeros((self.T, self.d)).to(self.device)
        RC_err = torch.zeros((self.T, self.d)).to(self.device)
        for t in range(self.T):
            
            for n in range(self.d):
            
                Z = theta[t,:,n]  * (-Delta_S[t,:,n])  * cumprod_Gamma[t,:]
                # Z = theta[t,:,n]  * (-Delta_S[t,:,n])  * Gamma[t,:]
                
                RC[t,n] = torch.mean( Z )
                RC_err[t,n] = torch.std( Z )/np.sqrt(batch_size)

        return RC, RC_err  
                
    def Update_Policy(self, N_iter=100, batch_size = 256):
        
        # torch.autograd.set_detect_anomaly(True)
        count = 0
        for i in range(N_iter):
            
            RC, _  = self.RiskContributions(batch_size = 512)
            
            loss = torch.mean((RC-self.B)**2)
                    
            self.beta_optimizer.zero_grad()
            self.wealth_0_optimizer.zero_grad()    
            
            loss.backward()
            
            self.beta_optimizer.step()
            self.wealth_0_optimizer.step()
                    
            beta_loss  = torch.tensor(0)
            
            self.beta_loss.append( beta_loss.item())
            
            costs, h, Y, beta, theta, S, X = self.__RunEpoch__(batch_size)

            V,  Gamma, cumprod_Gamma = self.__Compute_V_Gamma__(costs, h, Y)  
            
            self.V_0.append(V[0,0].item())
            
            self.mean_beta.append( torch.mean(beta, axis=1).reshape(-1).cpu().detach().numpy() )
            
            count+=1
            if np.mod(count, 100) == 0:
                
                plt.plot(self.beta_loss)
                plt.yscale('log')
                plt.show()
             
            self.RC.append(RC.cpu().detach().numpy())  
                
    def Train(self, n_iter=10_000, n_print = 10, M_value_iter = 10, M_policy_iter=1, batch_size=256):
        
        # torch.autograd.set_detect_anomaly(False)
        
        self.__Initialize_W_0__()
        
        print("training value function on initialization...")
        self.Update_ValueFunction(N_iter= 10, n_print=500, batch_size=batch_size)
        print("training F on initialization...")
        self.Update_F(N_iter= 10, n_print= 500, batch_size=batch_size)
        
        print("main training...")
        self.PlotPaths(500)

        count = 0
        for i in tqdm(range(n_iter)):
            
            # this updates mu, psi, chi
            self.Update_ValueFunction(N_iter=M_value_iter, n_print=500, batch_size=batch_size)
            
            # this udpates F
            self.Update_F(N_iter=5, n_print=500, batch_size=batch_size)
            
            # this updates beta and w_0
            self.Update_Policy(N_iter=M_policy_iter, batch_size=batch_size)
            
            count += 1
            
            if np.mod(count, n_print)==0:
                
                self.PlotSummary()
                self.PlotPaths(500)
                self.PlotHist()
                
    def __Initialize_W_0__(self):
        
        
        self.wealth_0.scale = 2
        
        # for i in range(10):
            
        #     self.Update_ValueFunction(N_iter=5, n_print=500, batch_size=512)
        #     self.Update_Policy(N_iter=5, batch_size=512)
        
        # scale = np.linspace(0.1, 500, 10)
        
        # V_0 = []
        # for s in scale:
            
        #     self.wealth_0.scale = s
        #     self.Update_ValueFunction(N_iter=10, batch_size=512)
            
        #     W_0 = self.wealth_0(1)
            
        #     # compute the value function at time zero
        #     h = torch.zeros((self.beta.gru.num_layers, 1, self.beta.gru.hidden_size))
        #     Y = torch.zeros((1, 1, self.Simulator.n+2))
        #     S_0 = torch.tensor(self.Simulator.S0).float() * torch.ones(( 1, self.d))
            
        #     Y = torch.cat( (torch.zeros((1,1)), W_0.reshape(-1,1), S_0), axis=1).unsqueeze(axis=1)            
            
        #     V_0.append(self.CVaR(0, h = h.transpose(0,1), Y=Y).squeeze().item())

        # idx = np.argmin(np.abs(np.array(V_0)-1))
        
        # self.wealth_0.scale = scale[idx]
        
        # self.VaR_CVaR_loss = []
        
        # self.W_0 = []
        
        # print(V_0)
        # print(scale)
        # print(scale[idx])
                
    def PlotHist(self, batch_size = 10_000):
        
        costs, h, Y, beta, theta, S, X = self.__RunEpoch__(batch_size)

        V,  Gamma,  cumprod_Gamma = self.__Compute_V_Gamma__(costs, h, Y)
        
        costs = costs.cpu().detach().numpy()
        V = V.cpu().detach().numpy()
        
        def kde(x, bins):
            h = np.std(x)*1.06*(len(x))**(-1/5)
            
            h += 1e-6
            dx = x.reshape(1,-1)-bins.reshape(-1,1)
            
            f = np.sum(np.exp(-0.5*dx**2/h**2)/np.sqrt(2*np.pi*h**2), axis=1)/len(x)
            
            return f
        
        bins=np.linspace(-0.2, 0.2, 51)
        kde_bins = np.linspace(-0.2, 0.2, 501)
        for i in range(self.T):
            
            plt.subplot(3,self.T, i+1)
            plt.hist(costs[i,:], bins=bins, density=True, alpha=0.5)
            f = kde(costs[i,:], kde_bins)
            plt.plot(kde_bins,f,linewidth=1,color='r')
            
            plt.xlabel(r'$c_'+str(i) + '$', fontsize=16)
        
            plt.subplot(3,self.T, i+1 + self.T)
            c_p_V = costs[i,:]
            if i < self.T-1:
                c_p_V += V[i+1,:]
                
            plt.hist(c_p_V, bins=bins, density=True, alpha=0.5)
            qtl = np.quantile(c_p_V, self.alpha)
            plt.axvline(qtl,color='k',linestyle='--',linewidth=1)
            f = kde(c_p_V, kde_bins)
            plt.plot(kde_bins,f,linewidth=1,color='r')
            plt.fill_between(kde_bins[kde_bins>=qtl], f[kde_bins>=qtl], alpha=0.5, color='r')
            
                
            plt.xlabel(r'$c_' + str(i) + '\;+\;V_'+str(i+1) + '$', fontsize=16)
        
            plt.subplot(3,self.T, i+1 + 2*self.T)
            plt.hist(V[i,:], bins=bins, density=True, alpha=0.5)
            f = kde(V[i,:], kde_bins)
            plt.plot(kde_bins,f,linewidth=1,color='r')
            plt.xlabel(r'$V_'+str(i) + '$', fontsize=16)
            
        plt.tight_layout()
        plt.show()
            
    def MovingAverage(self,x, n):
        
        y = np.zeros(len(x))
        
        for i in range(len(x)):
            if i < n:
                y[i] = x[i]
            else:
                y[i] = np.mean(x[i-n:i])
                
        return y
        
    def PlotSummary(self):
        
        plt.subplot(2,3,1)
        plt.plot(self.MovingAverage(self.beta_loss,10))
        plt.yscale('symlog')
        plt.ylim(1e-6, 0.01)
        plt.title(r'$V_0-\sum \mathbb{E}\log \theta$')
        
        plt.subplot(2,3,2)
        plt.plot(self.MovingAverage(self.VaR_CVaR_mean_loss,10))
        plt.yscale('symlog')
        plt.ylim(0.01, 0.1)
        plt.title(r'$S(VaR,CVaR)$')
        
        plt.subplot(2,3,3)
        plt.plot(self.W_0)
        plt.title(r'$W_0$')
        
        plt.subplot(2,3,4)
        RC = np.array(self.RC)
        RC_flat = RC.reshape(len(self.RC),-1)
        for i in range(RC_flat.shape[1]):
            plt.plot(self.MovingAverage(RC_flat[:,i],10))
        plt.ylabel('RC')
        
        
        plt.subplot(2,3,5) 
        V_0_est = np.sum(np.sum(RC, axis=1),axis=1)
        plt.plot(V_0_est, label=r'$\sum RC$', linewidth=1) 
        plt.plot(self.V_0, label=r'$V_0$', linewidth=1)
        plt.axhline(0.1, linestyle='--', color='k')
        plt.ylim(0,0.5)
        plt.legend(fontsize=8)
        
        plt.subplot(2,3,6)
        plt.plot(np.array(self.mean_beta))
        plt.title(r'$\mathbb{E}[\beta_{t,i}]$')
        
        plt.tight_layout()
        plt.show()      
        
        plt.figure(figsize=(10,4))
        idx = 1
        for k in range(self.d):
            for j in range(self.T):
               
                plt.subplot(self.d, self.T, idx)
                plt.plot(RC[:,j,k],alpha=0.5)
                plt.plot(self.MovingAverage(RC[:,j,k],50))
                plt.ylim(0,self.B[0,0,0].cpu().numpy()*self.eta*2)
                plt.ylabel(r'$RC_{' + str(j) + ','+str(k)+ '}$')
                plt.axhline(self.B[j,0,k].cpu()*self.eta, linestyle='--', color='k')
                idx += 1
        plt.suptitle(r'$RC$')
        plt.tight_layout()
        plt.show()      
                
    # def PlotValueFunc(self):
        
    #     print("need to check and also add VaR and mean outputs")
        
    #     # pdb.set_trace()
        
    #     S1_1, S2_1 = torch.meshgrid(torch.linspace(0.8, 1.1, 100),torch.linspace(0.8, 2.0, 100))
        
    #     S1_1 = S1_1.unsqueeze(axis=2)
    #     S2_1 = S2_1.unsqueeze(axis=2)
        
    #     ones = torch.ones(S1_1.shape)
        
    #     S1_0 = self.Simulator.S0[0][0] * ones
    #     S2_0 = self.Simulator.S0[0][1] * ones
        
    #     # concatenate t_i, wealth, and asset prices
    #     Y_0 = torch.cat((0*ones, S1_0, S2_0), axis=2)

    #     h_0 = torch.zeros((self.beta.gru.num_layers, 
    #                        S1_0.shape[0], 
    #                        S1_0.shape[1], 
    #                        self.beta.gru.hidden_size))
        
    #     h_1 = torch.zeros((self.beta.gru.num_layers, 
    #                        S1_0.shape[0], 
    #                        S1_0.shape[1], 
    #                        self.beta.gru.hidden_size))        

    #     X_1 = np.linspace(0.95, 1.05, 3)
    #     CVaR = torch.zeros((X_1.shape[0], S1_1.shape[0],S1_1.shape[1]))
        
    #     fig, ax = plt.subplots(X_1.shape[0], sharex=True, sharey=True)
    #     for j, x_1 in enumerate(X_1):
            
    #         Y_1 = torch.cat(((1*self.dt)*ones,  S1_1, S2_1), axis=2)

    #         for i in range(S1_0.shape[1]):
    #             # push through the neural-net to get weights
            
    #             h_1[:,:,i,:], beta_0 = self.beta(h_0[:,:,i,:], Y_0[:,i,:].unsqueeze(axis=1))
            
    #             CVaR[j,:,i] = self.CVaR(h_1[:,:,i,:].transpose(0,1), Y_1[:,i,:].unsqueeze(axis=1)).squeeze()
        
    #         im = ax[j].contourf(S1_1[:,:,0].numpy(), S2_1[:,:,0].numpy(), CVaR[j,:,:].detach().numpy())
    #         ax[j].set_title('$X_1={0:.2f}$'.format(x_1))
            
    #     plt.tight_layout()
        
    #     fig.subplots_adjust(right=0.8)
    #     cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    #     fig.colorbar(im, cax=cbar_ax)
        
    #     ax[-1].set_xlabel('$S^{(1)}_1$')
    #     ax[-2].set_ylabel('$S^{(2)}_1$')

    #     plt.show()
        
    # def Plotbeta(self, print_beta=False):
        
    #     # pdb.set_trace()
        
    #     S1_1, S2_1 = torch.meshgrid(torch.linspace(0.8, 1.1, 100),torch.linspace(0.8, 2.0, 100))
        
    #     S1_1 = S1_1.unsqueeze(axis=2)
    #     S2_1 = S2_1.unsqueeze(axis=2)
        
    #     ones = torch.ones(S1_1.shape)
        
    #     S1_0 = self.Simulator.S0[0][0] * ones
    #     S2_0 = self.Simulator.S0[0][1] * ones
        
    #     # concatenate t_i, X, and asset prices
    #     Y_0 = torch.cat((0*ones, S1_0, S2_0), axis=2)

    #     h_0 = torch.zeros((self.beta.gru.num_layers, 
    #                        S1_0.shape[0], 
    #                        S1_0.shape[1], 
    #                        self.beta.gru.hidden_size))
        
    #     h_1 = h_0.clone()
    #     h_2 = h_1.clone()

    #     X_1 = np.linspace(0.95, 1.05, 3)
        
    #     fig0, ax0 = plt.subplots(nrows=self.d, ncols=X_1.shape[0], sharex='all', sharey='all')
    #     fig1, ax1 = plt.subplots(nrows=self.d, ncols=X_1.shape[0], sharex='all', sharey='all')        
    #     for j, x_1 in enumerate(X_1):
            
    #         Y_1 = torch.cat(((1*self.dt)*ones, S1_1, S2_1), axis=2)
            
    #         beta_0 = torch.zeros((S1_0.shape[0], S1_0.shape[0], self.d))
    #         beta_1 = beta_0.clone()
            
    #         for i in range(S1_0.shape[1]):
    #             # push through the neural-net to get weights
            
    #             h_1[:,:,i,:], beta_0[:,i,:] = self.beta(h_0[:,:,i,:], Y_0[:,i,:].unsqueeze(axis=1))
    #             h_2[:,:,i,:], beta_1[:,i,:] = self.beta(h_1[:,:,i,:], Y_1[:,i,:].unsqueeze(axis=1))
            
            
    #         rescale_factor = (self.X0/ torch.sum(beta_0 * S1_0, axis=2))
    #         beta_0 *= rescale_factor.unsqueeze(axis=2)
    #         beta_1 *= rescale_factor.unsqueeze(axis=2)
            
    #         # varbeta_0 = (beta_0[:,:,0] * S1_0[:,:,0] / self.X0).detach().numpy()
    #         varbeta_0 = beta_0[:,:,0].detach()
    #         im0 = ax0[0,j].contourf(S1_1[:,:,0].numpy(), S2_1[:,:,0].numpy(), varbeta_0, vmin=0, vmax=1)
    #         ax0[0, j].set_title(r'$\varbeta_0^{0:d}$, $X_0={1:.2f}$'.format(0,x_1), fontsize=20)

    #         # varbeta_0 = (beta_0[:,:,1] * S2_0[:,:,0] / self.X0).detach().numpy()
    #         varbeta_0 = beta_0[:,:,1].detach()
    #         im0 = ax0[1,j].contourf(S1_1[:,:,0].numpy(), S2_1[:,:,0].numpy(), varbeta_0, vmin=0, vmax=1)
    #         ax0[1, j].set_title(r'$\varbeta_0^{0:d}$, $X_0={1:.2f}$'.format(1,x_1), fontsize=20)            
            
            
    #         # varbeta_1 = (beta_1[:,:,0] * S1_1[:,:,0] / self.X0).detach().numpy()
    #         varbeta_1 = beta_1[:,:,0].detach()
    #         im1 = ax1[0,j].contourf(S1_1[:,:,0].numpy(), S2_1[:,:,0].numpy(), varbeta_1, vmin=0, vmax=1)
    #         ax1[0, j].set_title(r'$\varbeta_1^{0:d}$, $X_1={1:.2f}$'.format(0,x_1), fontsize=20)
            
    #         # varbeta_1 = (beta_1[:,:,1] * S2_1[:,:,0] / self.X0).detach().numpy()
    #         varbeta_1 = beta_1[:,:,1].detach()
    #         im1 = ax1[1,j].contourf(S1_1[:,:,0].numpy(), S2_1[:,:,0].numpy(), varbeta_1, vmin=0, vmax=1)
    #         ax1[1, j].set_title(r'$\varbeta_1^{0:d}$, $X_1={1:.2f}$'.format(1,x_1), fontsize=20)     
            
    #         if print_beta:
    #             print(j,end="\n\n")
    #             print(r'$\beta_0^0$')
    #             print(varbeta_0)
    #             print(r'$\beta_0^1$')
    #             print(varbeta_0)
    #             print(r'$\beta_1^0$\n')
    #             print(varbeta_1)                
    #             print(r'$\beta_1^1$\n')
    #             print(varbeta_1)
                
    #     fig0.tight_layout()
        
    #     fig0.subplots_adjust(right=0.8)
    #     cbar_ax = fig0.add_axes([0.85, 0.15, 0.05, 0.7])
    #     fig0.colorbar(im0, cax=cbar_ax)
        
    #     ax0[-1,-1].set_xlabel('$S^{(1)}_1$')
    #     ax0[0,0].set_ylabel('$S^{(2)}_1$')
        
        
    #     fig1.tight_layout()
        
    #     fig1.subplots_adjust(right=0.8)
    #     cbar_ax = fig1.add_axes([0.85, 0.15, 0.05, 0.7])
    #     fig1.colorbar(im1, cax=cbar_ax)
        
    #     ax1[-1,-1].set_xlabel('$S^{(1)}_1$')
    #     ax1[0,0].set_ylabel('$S^{(2)}_1$')        
        
    #     plt.show()
        
    def PlotPaths(self, batch_size=1, title=None):
        
        costs, h, Y, beta, theta, S, wealth = self.__RunEpoch__(batch_size)
        
        # rescale_factor = ( self.X0/ torch.sum(beta[0,:,:]*S[0,:,:],axis=1) )
        # beta *= rescale_factor.reshape(1,-1,1)
        # wealth *= rescale_factor.reshape(1,-1)
        
        costs = costs.cpu().detach().numpy()
        h = h.cpu().detach().numpy()
        Y = Y.cpu().detach().numpy()
        beta = beta.cpu().detach().numpy()
        S = S.cpu().detach().numpy()
        wealth = wealth.cpu().detach().numpy()
        
        
        # for j in range(batch_size):
        #     plt.plot(S[:,j,0], linestyle= '-')
        #     plt.plot(S[:,j,1], linestyle= '--')
        # plt.title(r'$S_t$')
        # plt.show()
        
        fig, ax = plt.subplots(nrows=self.d,ncols=2)
        
        # pdb.set_trace()
        
        qtl_S = np.quantile(S, [0.1,0.9], axis=1)
        qtl_beta = np.quantile(beta, [0.1,0.9], axis=1)
        
        for j in range(self.d):
                ax[j,0].set_ylabel(r'$S^{0:2d}$'.format(j+1))
                ax[j,0].plot(S[:,:,j], alpha=0.1)
                ax[j,0].plot(S[:,0,j], color='r', linewidth=1)
                ax[j,0].plot(qtl_S[:,:,j].T, color='k', linewidth=1)
                ax[j,0].set_ylim(0.7, 1.4)
                ax[j,1].set_xticks([0,1,2])
                
                ax[j,1].set_ylabel(r'$\beta^{0:2d}$'.format(j+1))
                ax[j,1].plot(beta[:,:,j], alpha=0.1)
                ax[j,1].plot(beta[:,0,j], color='r', linewidth=1)
                ax[j,1].plot(qtl_beta[:,:,j].T, color='k', linewidth=1)
                ax[j,1].set_ylim(0, 1)
                ax[j,1].set_xticks([0,1,2])
                
                
            # ax[1].set_ylim(0,1)
        if title is not None:
            plt.suptitle(title)
            plt.tight_layout()
            
            fig.savefig(title)
        
        plt.tight_layout()
        plt.show()
        
        for j in range(1, self.T):
            
            fig = plt.figure(figsize=(12,6))
            
            for k in range(self.d):
                
                ax = plt.subplot(1,self.d,k+1)
                
                plt.title(r'$\beta_{0:1d}^{1:1d}$'.format(j,k+1), fontsize=20)
                im1=plt.scatter(S[j,:,0], S[j,:,1], 
                                s=10, alpha=0.8, c=beta[j,:,k], 
                                cmap='brg', vmin=0.3, vmax=0.7)
                plt.scatter(S[j,0,0],S[j,0,1],
                            s=50, 
                            c = beta[j,0,k], 
                            facecolors=None,
                            edgecolors='k')
                plt.xlabel(r'$S_{0:1d}^1$'.format(j),fontsize=20)
                plt.ylabel(r'$S_{0:1d}^2$'.format(j),fontsize=20)
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                plt.xlim(0.7, 1.4)
                plt.ylim(0.8, 1.3)
                
            plt.tight_layout(pad=2)
            
            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
            fig.colorbar(im1, cax=cbar_ax)
        
            plt.show()
        
        return S, beta