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

import dill
from datetime import datetime

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
        self.softmax = nn.Softmax()
        
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
        elif self.out_activation == 'softmax':
            output = self.softmax(output)
            
        return output
    
class ThetaNet(nn.Module):
    
    def __init__(self, 
                 nIn, 
                 nOut, 
                 gru_hidden=5, 
                 gru_layers=5, 
                 linear_hidden = 36, 
                 linear_layers=5,
                 device='cpu'):
        super(ThetaNet, self).__init__()

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
        self.sigmoid = nn.Sigmoid()
        
        
    def forward(self, h, Y):
        
        _, h_out = self.gru(Y.clone(), h.clone())
        
        x = torch.cat((h.transpose(0,1).flatten(start_dim=-2), 
                       Y.flatten(start_dim=-2)),
                      axis=-1)
        
        x = self.gru_to_hidden(x)
        x = self.g(x)
        
        for linear in self.linear_hidden_to_hidden:
            x = self.g(linear(x))
            
        theta = 10*self.sigmoid(self.hidden_to_out(x)).squeeze()
        
        
        return h_out, theta
        
class DynamicRiskParity():
    
    def __init__(self, Simulator : Simulator_OU, X0=1, B = 0, alpha=0.8, p=0.5, beta=None):
        
        # set the device to use
        if torch.cuda.device_count() > 0:
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')
            
        if beta is not None:
            self.beta = torch.tensor(beta).float().to(self.device)
        else:
            self.beta = beta            
        
        self.Simulator = Simulator
        self.X0 = X0        
        self.T = Simulator.T
        self.d = Simulator.n
        self.dt = Simulator.dt
        self.alpha = alpha
        self.p = p
        self.gamma = lambda u : self.p*(u>self.alpha)/(1-alpha) + (1-self.p)
        
        if B == 0:
            self.B = torch.ones((self.T-1, 1, self.d))/((self.T-1)*self.d)
        else:
            self.B = torch.tensor(B).float()
            
        self.B = self.B.to(self.device)
        
        #
        # the percentage of wealth in each asset: 
        # states = past asset prices (encoded in a GRU), current time and wealth
        #
        self.theta = ThetaNet(nIn=self.d+2,
                            nOut=self.d,
                            gru_hidden=self.Simulator.n,
                            gru_layers=5,
                            linear_hidden=32,
                            linear_layers=5,
                            device=self.device)
        self.theta_optimizer = optim.AdamW(self.theta.parameters(),
                                           lr = 0.001)
        
        self.theta_scheduler = optim.lr_scheduler.StepLR(self.theta_optimizer,
                                                         step_size=10,
                                                         gamma=0.99)
        
        self.__initialize_CVaR_VaR_Mean_Net__(self.device)
        
        # for storing losses
        self.VaR_CVaR_mean_loss = []
        self.F_loss = []
        self.theta_loss = []

        self.eta = 0.1 # the Lagrange multiplier factor        
        
        self.RC = []
        self.V_0 = []
        
    def __initialize_CVaR_VaR_Mean_Net__(self, device='cpu'):
    
        # Need to elicit both conditional VaR (using network psi) and conditional 
        # CVaR (using network chi)
        # VaR = psi( states )
        # CVaR = psi( states ) + chi( states )
        # the states consists of the output of the gru layers
        
        def get_optimizer_scheduler(net):
            
            optimizer = optim.AdamW(net.parameters(), lr=0.001)
            scheduler = optim.lr_scheduler.StepLR(optimizer,
                                                  step_size=10,
                                                  gamma=0.99)
            
            return optimizer, scheduler
        
        gru_total_hidden = self.theta.gru.num_layers*self.theta.gru.hidden_size
        n_in = gru_total_hidden+2+self.Simulator.n
        
        # terminal percentage of wealth
        if self.beta is None:
            self.beta = Net(n_in=n_in, n_out=1, nNodes=32, nLayers=5, device=self.device)
            self.beta_optimizer, self.beta_scheduler = get_optimizer_scheduler(self.beta)
        else:
            self.beta_optimizer = None
            self.beta_scheduler  = None
        
        # for conditional cdf of c_t + V_{t+1}
        self.F = Net(n_in=n_in+1, n_out=1, nNodes=16, nLayers=3, out_activation='sigmoid', device=self.device)
        self.F_optimizer, self.F_scheduler = get_optimizer_scheduler(self.F)
        
        # for conditional mean
        self.mu = Net(n_in=n_in, n_out=1, nNodes=32, nLayers=5, device=self.device)
        self.mu_target = copy.deepcopy(self.mu)
        self.mu_optimizer, self.mu_scheduler = get_optimizer_scheduler(self.mu)    
        
        # for conditional VaR
        self.psi = Net(n_in=n_in, n_out=1, nNodes=32, nLayers=5, device=self.device)
        self.psi_target = copy.deepcopy(self.psi)
        self.psi_optimizer, self.psi_scheduler = get_optimizer_scheduler(self.psi)  
        
        # for increment from conditional VaR to conditional CVaR
        # i.e., CVaR = chi + psi
        self.chi = Net(n_in=n_in, n_out=1, nNodes=32, nLayers=5, out_activation='softplus', device=self.device)
        self.chi_target = copy.deepcopy(self.chi)
        self.chi_optimizer, self.chi_scheduler = get_optimizer_scheduler(self.chi)
        
        
        self.VaR = lambda h, Y : self.psi(h, Y)
        self.CVaR = lambda h, Y : self.psi(h, Y) + self.chi(h, Y)
        
        self.risk_measure = lambda h, Y :  (self.p * self.CVaR(h, Y) 
                                            + (1.0-self.p)* self.mu(h, Y))
        
        self.VaR_target = lambda h, Y : self.psi_target(h, Y)
        self.CVaR_target = lambda h, Y : self.psi_target(h, Y) + self.chi_target(h, Y)
        self.risk_measure_target = lambda h, Y :  (self.p * self.CVaR_target(h, Y) 
                                                   + (1.0-self.p)* self.mu_target(h, Y))
        
        
    def __RunEpoch__(self, batch_size = 256):
        
        # grab a simulation of the market prices
        X = torch.tensor(self.Simulator.Simulate(batch_size)).float().transpose(0,1).to(self.device)
        
        #
        # store the hidden states from all layers and each time
        #
        h = torch.zeros((self.T+1,
                         self.theta.gru.num_layers, 
                         batch_size, 
                         self.theta.gru.hidden_size)).to(self.device)
        
        # concatenate t_i, wealth, and asset prices
        Y = torch.zeros((self.T, batch_size, 1, self.Simulator.n+2)).to(self.device)
        ones = torch.ones((batch_size,1)).to(self.device)
        zeros = 0*ones
        Y[0,...] = torch.cat( (zeros, 
                               self.X0*ones, 
                               X[0,:,:]), axis=1).unsqueeze(axis=1)
        
        # push through the neural-net to get weights
        theta = torch.zeros((self.T, batch_size,  self.d)).to(self.device)
        theta_hat = torch.zeros((self.T, batch_size,  self.d)).to(self.device)
        beta = torch.zeros((self.T, batch_size,  self.d)).to(self.device)
        
        h[1,...], theta[0,:,:] = self.theta(h[0,...], Y[0,...])
        theta_hat[0,:,:] = theta[0,:,:].clone()
    
        # stores wealth process
        wealth = torch.zeros((self.T+1, batch_size)).to(self.device)
        wealth[0,:] = torch.sum(theta_hat[0,:,:] * X[0,:,:], axis=1)
        
        beta[0,:,:] = theta_hat[0,:,:].clone() * X[0,...] / wealth[0,:].unsqueeze(axis=1).clone()
        
        for t in range(1, self.T):
            
            # new wealth at period end
            wealth[t,:] = torch.sum( theta_hat[t-1,:,:] * X[t,:,:], axis=1).clone()
            
            # concatenate t_i, and asset prices    
            Y[t,...] = torch.cat(((t*self.dt)*ones, 
                                  wealth[t,:].reshape(-1,1).clone(),
                                  X[t,:,:]), axis=1).unsqueeze(axis=1)
            
            # at last time step replace theta with beta
            if t == self.T-1:
                
                if self.beta_optimizer is None:
                    beta[t,:,:] = self.beta.unsqueeze(axis=0)
                else:
                    beta[t,:,:] = self.beta(h[t,...].detach(),
                                            Y[t,...].detach())
                    
                theta[t,:,:] = wealth[t,:].unsqueeze(axis=1).clone() * beta[t,:,:] / X[t,:,:]
                theta_hat[t,:,:] = theta[t,:,:].clone() 
             
            else:
                
                # push through the neural-net to get new number of assets
                h[t+1,...], theta[t,:,:] = self.theta(h[t,...].detach(),
                                                      Y[t,...].detach())
                
                weight = torch.sum( theta[t-1,:,:] * X[t,:,:], axis=1).clone() \
                    / torch.sum( theta[t,:,:] * X[t,:,:], axis=1).clone()  
                 
                theta_hat[t,:,:] = theta[t,:,:].clone() * weight.reshape(-1,1).detach()
            
            if t < self.T-1:
               
                beta[t,:,:] = theta_hat[t,:,:].clone() * X[t,...] / wealth[t,:].unsqueeze(axis=1).clone()
                
        wealth[-1,:] = torch.sum( theta_hat[-1,:,:] * X[-1,:,:], axis=1).clone()
            
        costs = -torch.diff(wealth, axis=0)
        
        # costs = torch.zeros((self.T, batch_size)).to(self.device)
        # costs[0,:] = torch.sum(theta[0,:,:] * (X[0,:,:]-X[1,:,:]),axis=1)
        # for t in range(1, self.T):
        #     costs[t,:] = torch.sum(theta[t-1,:,:] *X[t,:,:], axis=1)\
        #         /torch.sum(theta[t,:,:] *X[t,:,:], axis=1) \
        #             * torch.sum(theta[t,:,:] * (X[t,:,:]-X[t+1,:,:]),axis=1)
        
        
        return costs, h, Y, beta, theta, theta_hat, X, wealth    
    
    def __V_Score__(self, VaR, CVaR, mu, X):
        
        xmax = torch.max(torch.abs(X))
        if xmax > 5:
            C = 2*xmax
        else:
            C = 5
            
        A = ((X<=VaR)*1.0-self.alpha)*VaR + (X>VaR)*X
        B = (CVaR+C)*(1.0-self.alpha)
        
        # for conditional VaR & CVaR
        score = torch.mean(torch.log( (CVaR+C)/(X+C)) - (CVaR/(CVaR+C)) + A/B)
        
        # for conditional mean
        score += torch.mean((mu-X)**2)
        
        return score
    
    def __update_valuefunction__(self, N_iter  = 100, batch_size = 256, n_print=100):
        
        for j in range(N_iter):
        
            costs, h, Y, beta, theta, theta_hat, X, wealth = self.__RunEpoch__(batch_size)
            
            h = h.detach()
            Y = Y.detach()
            
            costs = costs.unsqueeze(axis=2)
            
            self.psi_optimizer.zero_grad()
            self.chi_optimizer.zero_grad()
            self.mu_optimizer.zero_grad()

            loss = 0.0
            
            for t in range(0, self.T):

                cost_plus_V_one_ahead = 1.0* costs[t,:]
                
                if t < self.T-1:
                    cost_plus_V_one_ahead += self.risk_measure_target(h[t+1,...].transpose(0,1), 
                                                                      Y[t+1,...])
                
                loss += self.__V_Score__(self.VaR(h[t,...].transpose(0,1), Y[t,...]),
                                       self.CVaR(h[t,...].transpose(0,1), Y[t,...]),
                                       self.mu(h[t,...].transpose(0,1), Y[t,...]),
                                       cost_plus_V_one_ahead.detach())

            loss.backward()
            
            self.psi_optimizer.step()
            self.chi_optimizer.step()
            self.mu_optimizer.step()
            
            self.psi_scheduler.step()
            self.chi_scheduler.step()
            self.mu_scheduler.step()            
            
            self.VaR_CVaR_mean_loss.append(loss.item())
            
        self.chi_target = copy.deepcopy(self.chi)
        self.psi_target = copy.deepcopy(self.psi)
        self.mu_target = copy.deepcopy(self.mu)
        
    def __F_Score__(self, h, Y, X):
        
        max_z = 10
        
        if torch.max(X) > max_z:
            print('X too large')
            pdb.set_trace()
        
        if torch.min(X) < -max_z:
            print('X too small')
            pdb.set_trace()
        
        N = 501
        z = torch.linspace(-max_z, max_z,N).reshape(N,1,1,1).repeat(1,Y.shape[0], Y.shape[1], 1).to(self.device)
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
        
        
    def __update_F__(self, N_iter  = 100, batch_size = 256, n_print=100):
        
        for j in range(N_iter):
        
            costs, h, Y, beta, theta, theta_hat, X, wealth = self.__RunEpoch__(batch_size)
            
            costs = costs.unsqueeze(axis=2).detach()
            
            h = h.detach()
            Y = Y.detach()
            
            loss = 0.0
            
            self.F_optimizer.zero_grad()
            
            for t in range(0, self.T):

                cost_plus_V_one_ahead = 1.0* costs[t,:]
                
                if t < self.T-1:
                    cost_plus_V_one_ahead += self.risk_measure_target(h[t+1,...].transpose(0,1), 
                                                                      Y[t+1,...])
                
                loss += self.__F_Score__(h[t,...].transpose(0,1), 
                                         Y[t,...], 
                                         cost_plus_V_one_ahead.detach())

            loss.backward()
            
            self.F_optimizer.step()
            self.F_scheduler.step()
            
            self.F_loss.append(loss.item())
            
    def __get_v_gamma__(self, costs, h, Y):
        
            batch_size = costs.shape[1]
            
            # compute the value function at points in time            
            V = torch.zeros((self.T+1, batch_size)).to(self.device)
            for t in range(self.T):
                V[t,:] = self.risk_measure_target(h=h[t,...].transpose(0,1), 
                                                  Y=Y[t,...]).squeeze()
            
            costs_plus_V_onestep_ahead  = (costs + V[1:,:])

            # evalute the Gamma processes
            U = torch.zeros((self.T, batch_size)).to(self.device)
            
            Z = torch.concat((Y,
                              costs_plus_V_onestep_ahead.unsqueeze(axis=2).unsqueeze(axis=3)), 
                             axis=3)
            
            for t in range(self.T):
                U[t,:] = self.F(h[t,...].transpose(0,1), 
                                Z[t,...])[:,0]
                
            Gamma = self.gamma(U)
                
            cumprod_Gamma = torch.cumprod(Gamma, axis=0)
            
            return V, Gamma, cumprod_Gamma
        
    def __get_gateaux_terms__(self, batch_size=2048):
        
        costs, h, Y, beta, theta_grad, theta_hat, X, wealth = self.__RunEpoch__(batch_size)
        
        V,  Gamma, cumprod_Gamma = self.__get_v_gamma__(costs.detach(),
                                                        h.detach(), 
                                                        Y.detach())
        
        Delta_X = -torch.diff(X, axis=0)

        theta = theta_grad.detach()
        cumprod_Gamma = cumprod_Gamma.detach()
        
        Z = torch.zeros(theta[:-1,...].shape).to(self.device)

        for t in range(self.T-1): 

            if t > 0:
                
                theta_t_X_t = torch.sum(theta[t,:,:]  * X[t,:,:],axis=1)
                theta_t_dX_t = torch.sum(theta[t,:,:]  * Delta_X[t,:,:],axis=1)
                theta_tm1_X_t = torch.sum(theta[t-1,:,:]  * X[t,:,:],axis=1)                
            
                
            theta_tp1_X_tp1 = torch.sum(theta[t+1,:,:]  * X[t+1,:,:],axis=1)
            theta_tp1_dX_tp1 = torch.sum(theta[t+1,:,:]  * Delta_X[t+1,:,:],axis=1)

            for i in range(self.d):

                if t > 0:
                
                    Z[t,:,i] = ( (theta_tm1_X_t/theta_t_X_t) * Delta_X[t,:,i]\
                                - (theta_tm1_X_t/(theta_t_X_t**2)) * theta_t_dX_t  * X[t,:,i]) * cumprod_Gamma[t,:]
                else:
                    
                    Z[0,:,i] = Delta_X[0,:,i] * cumprod_Gamma[0,:]

                    
                Z[t,:,i] += X[t+1,:,i]* (theta_tp1_dX_tp1 /theta_tp1_X_tp1) * cumprod_Gamma[t+1,:]
                    
        Z *= theta_grad[:-1,...]
                
        return Z, theta_grad, V
        
    def RiskContributions(self, batch_size=2048):
                
        Z, _, _ = self.__get_gateaux_terms__(batch_size)
        
        RC = torch.mean(Z, axis=1)
                
        RC_err  = torch.std( Z, axis=1 )/np.sqrt(batch_size)

        return RC, RC_err  
    
    def __update_policy__(self, N_iter=100, batch_size = 256):
        """
        this updates the policy -- here policy = theta
        """
        
        count = 0
        for k in range(N_iter):
            
            t_rnd = np.random.permutation(np.arange(self.T-1))
            for t in t_rnd:
                
                i_rnd = np.random.permutation(np.arange(self.d))
                
                for i in i_rnd:
                    
                    Z, theta_grad, V = self.__get_gateaux_terms__(batch_size)
                    
                    loss = torch.mean(Z[t,:,i]) \
                        - self.eta*torch.mean( self.B[t,:,i] * torch.log(theta_grad[t,:,i]))
        
                    loss.backward()
        
                    self.theta_optimizer.step()
                    self.theta_scheduler.step()
            
            if self.beta_optimizer is not None:
                
                t = self.T-1
                for i in i_rnd:
                    
                    Z, theta_grad, V = self.__get_gateaux_terms__(batch_size)
                    
                    loss = torch.mean(Z[t,:,i]) \
                        - self.eta*torch.mean( self.B[t,:,i] * torch.log(theta_grad[t,:,i]))
        
                    loss.backward()
        
                    self.beta_optimizer.step()
                    self.beta_scheduler.step()
            
            self.V_0.append(V[0,0].item())
            
            count+=1
            if np.mod(count, 100) == 0:
                
                plt.plot(self.beta_loss)
                plt.yscale('log')
                plt.show()
                
            RC, RC_err = self.RiskContributions(500)
            self.RC.append(RC.cpu().detach().numpy())  
                
    def Train(self, n_iter=10_000, n_print = 10, M_value_iter = 10, M_policy_iter=1, batch_size=256, name=""):
        
        torch.autograd.set_detect_anomaly(False)
        
        print("training value function on initialization...")
        self.__update_valuefunction__(N_iter= 10, n_print=500, batch_size=batch_size)
        print("training F on initialization...")
        self.__update_F__(N_iter= 10, n_print= 500, batch_size=batch_size)
        
        print("main training...")
        self.PlotPaths(500)

        count = 0
        for i in tqdm(range(n_iter)):
            
            # this updates mu, psi, chi
            self.__update_valuefunction__(N_iter=M_value_iter, n_print=500, batch_size=batch_size)
            
            # this udpates F
            self.__update_F__(N_iter=10, n_print=500, batch_size=batch_size)
            
            # this updates beta and w_0
            self.__update_policy__(N_iter=M_policy_iter, batch_size=batch_size)
            
            count += 1
            
            if np.mod(count, n_print)==0:
                
                self.PlotSummary()
                self.PlotPaths(500)
                self.PlotHist()
                
                date_time = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
                dill.dump(self, open(name + '_' + date_time + '.pkl','wb'))   
                
    def PlotHist(self, batch_size = 10_000):
        
        costs, h, Y, beta, theta, theta_hat, S, X = self.__RunEpoch__(batch_size)

        V,  Gamma,  cumprod_Gamma = self.__get_v_gamma__(costs, h, Y)
        
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
        
        V_bins=np.linspace(-0.05, 0.05, 51)
        V_kde_bins = np.linspace(-0.05, 0.05, 501)
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
            plt.hist(V[i,:], bins=V_bins, density=True, alpha=0.5)
            f = kde(V[i,:], V_kde_bins)
            plt.plot(V_kde_bins,f,linewidth=1,color='r')
            plt.xlabel(r'$V_'+str(i) + '$', fontsize=16)
            
        plt.tight_layout()
        plt.show()
            
    def MovingAverage(self,x, n):
        
        y = np.zeros(len(x))
        
        for i in range(len(x)):
            if i < n:
                y[i] = np.mean(x[:i])
            else:
                y[i] = np.mean(x[i-n:i])
                
        return y
        
    def PlotSummary(self):
        
        plt.subplot(2,3,1)
        plt.plot(self.MovingAverage(self.theta_loss,10))
        plt.yscale('symlog')
        plt.ylim(1e-6, 0.01)
        plt.title(r'$V_0-\sum \mathbb{E}\log \theta$')
        
        plt.subplot(2,3,2)
        plt.plot(self.MovingAverage(self.VaR_CVaR_mean_loss,10))
        plt.yscale('symlog')
        plt.ylim(0.01, 0.1)
        plt.title(r'$S(VaR,CVaR)$')
        
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
        plt.axhline(self.eta, linestyle='--', color='k')
        plt.ylim(0,1.5*self.eta)
        plt.legend(fontsize=8)
        
        plt.tight_layout()
        plt.show()      
        
        plt.figure(figsize=(10,4))
        idx = 1
          
        for k in range(self.d):
            for j in range(self.T-1):
        
                # RC_target =self.MovingAverage(RC[:,j,k],100)[-1]
                # target = self.B[j,0,k].cpu().numpy() *self.eta
                # RC_ma = (self.MovingAverage(RC[:,j,k],50) / RC_target) * target
                
                # plt.subplot(self.d, self.T, idx)
                # plt.plot((RC[:,j,k] / RC_target)*target,alpha=0.5)
                # plt.plot(RC_ma)
                
                RC_ma = (self.MovingAverage(RC[:,j,k],50)) 
                
                plt.subplot(self.d, self.T, idx)
                plt.plot(RC[:,j,k],alpha=0.5)
                plt.plot(RC_ma)                
                
                plt.ylim(0,self.B[0,0,0].cpu().numpy()*self.eta*2)
                plt.ylabel(r'$RC_{' + str(j) + ','+str(k)+ '}$')
                plt.axhline(self.B[j,0,k].cpu()*self.eta, linestyle='--', color='k')
                idx += 1
                
        plt.suptitle(r'$RC$')
        plt.tight_layout()
        plt.show()      
                
       
    def PlotPaths(self, batch_size=1, title=None):
        
        costs, h, Y, beta, theta, theta_hat, S, wealth = self.__RunEpoch__(batch_size)
        
        costs = costs.cpu().detach().numpy()
        h = h.cpu().detach().numpy()
        Y = Y.cpu().detach().numpy()
        beta = beta.cpu().detach().numpy()
        S = S.cpu().detach().numpy()
        wealth = wealth.cpu().detach().numpy()
        
        
        fig, ax = plt.subplots(nrows=self.d,ncols=2)
        
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