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
    
    def __init__(self, Simulator : Simulator_OU, X0=1, B = 0, alpha=0.8, p=0.5):
        
        # set the device to use
        if torch.cuda.device_count() > 0:
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')
            
        self.env = Simulator
        self.X0 = X0        
        self.T = Simulator.T
        self.d = Simulator.n
        self.dt = Simulator.dt
        self.alpha = alpha
        self.p = p
        self.gamma = lambda u : self.p*(u>self.alpha)/(1-alpha) + (1-self.p)
        
        if B == 0:
            self.B = torch.ones((self.T, 1, self.d))/(self.T*self.d)
        else:
            self.B = torch.tensor(B).float()
            
        self.B /= torch.sum(self.B,axis=2).unsqueeze(axis=2)
            
        self.B = self.B.to(self.device)
        
        #
        # the percentage of wealth in each asset: 
        # states = past asset prices (encoded in a GRU), current time and wealth
        #
        self.theta = []
        self.theta_optimizer = []
        self.theta_scheduler = []
        for t in range(self.T):
            self.theta.append(ThetaNet(nIn=self.d+2,
                                nOut=self.d,
                                gru_hidden=self.env.n,
                                gru_layers=5,
                                linear_hidden=32,
                                linear_layers=5,
                                device=self.device))
            
            if t > 0:
                self.theta[-1].gru = self.theta[0].gru
            
            self.theta_optimizer.append(optim.AdamW(self.theta[t].parameters(),
                                                    lr = 0.001))
            
            self.theta_scheduler.append(optim.lr_scheduler.StepLR(self.theta_optimizer[t],
                                                             step_size=10,
                                                             gamma=0.99))
        
        self.__initialize_CVaR_VaR_Mean_Net__(self.device)
        
        # for storing losses
        self.VaR_CVaR_mean_loss = []
        self.F_loss = []
        self.theta_loss = []

        self.eta = 0.1 # the Lagrange multiplier factor        
        
        self.RC = []
        self.V = []
        
    def __initialize_CVaR_VaR_Mean_Net__(self, device='cpu'):
    
        # Need to elicit both conditional VaR (using network psi) and conditional 
        # CVaR (using network chi)
        # VaR = psi( states )
        # CVaR = psi( states ) + chi( states )
        # the states consists of the output of the gru layers
        
        gru_total_hidden = self.theta[0].gru.num_layers*self.theta[0].gru.hidden_size
        n_in = gru_total_hidden+2+self.env.n
        
        def get_opt_sched(net):
            opt = optim.AdamW(net.parameters(), lr=0.001)
            sched = optim.lr_scheduler.StepLR(opt,
                                              step_size=10,
                                              gamma=0.99)
            return opt, sched
        
        self.F = []
        self.mu = []
        self.psi = []
        self.chi = []
        
        self.F_optimizer = []
        self.F_scheduler = []
        self.mu_optimizer = []
        self.mu_scheduler = []
        self.psi_optimizer = []
        self.psi_scheduler = []
        self.chi_optimizer = []
        self.chi_scheduler = []
        
        for t in range(self.T):
            
            # for conditional cdf of c_t + V_{t+1}
            self.F.append(Net(n_in=n_in+1, n_out=1, nNodes=16, nLayers=3, out_activation='sigmoid', device=self.device))
            o, s = get_opt_sched(self.F[-1])
            self.F_optimizer.append(o)
            self.F_scheduler.append(s)
            
            # for conditional mean
            self.mu.append(Net(n_in=n_in, n_out=1, nNodes=32, nLayers=5, device=self.device))
            o, s = get_opt_sched(self.mu[-1])
            self.mu_optimizer.append(o)
            self.mu_scheduler.append(s)
            
            # for conditional VaR
            self.psi.append(Net(n_in=n_in, n_out=1, nNodes=32, nLayers=5, device=self.device))
            o, s = get_opt_sched(self.psi[-1])
            self.psi_optimizer.append(o)
            self.psi_scheduler.append(s)
            
            # for increment from conditional VaR to conditional CVaR
            # i.e., CVaR = chi + psi
            self.chi.append(Net(n_in=n_in, n_out=1, nNodes=32, nLayers=5, out_activation='softplus', device=self.device))
            o, s = get_opt_sched(self.chi[-1])
            self.chi_optimizer.append(o)
            self.chi_scheduler.append(s)
        
        self.mu_target = copy.deepcopy(self.mu)
        self.psi_target = copy.deepcopy(self.psi)
        self.chi_target = copy.deepcopy(self.chi)
        
        self.VaR = lambda t, h, Y : self.psi[t](h, Y)
        self.CVaR = lambda t, h, Y : self.psi[t](h, Y) + self.chi[t](h, Y)
        
        self.risk_measure = lambda t, h, Y :  (self.p * self.CVaR(t, h, Y)
                                               + (1.0-self.p)* self.mu[t](h, Y))
        
        self.VaR_target = lambda t, h, Y : self.psi_target[t](h, Y)
        self.CVaR_target = lambda t, h, Y : self.psi_target[t](h, Y) + self.chi_target[t](h, Y)
        self.risk_measure_target = lambda t, h, Y :  (self.p * self.CVaR_target(t, h, Y) 
                                                      + (1.0-self.p)* self.mu_target[t](h, Y))       
        
        
    def __run_epoch__(self, batch_size = 256):
        
        # grab a simulation of the market prices
        X = torch.tensor(self.env.Simulate(batch_size)).float().transpose(0,1).to(self.device)
        
        #
        # store the hidden states from all layers and each time
        #
        h = torch.zeros((self.T+1,
                         self.theta[0].gru.num_layers, 
                         batch_size, 
                         self.theta[0].gru.hidden_size)).to(self.device)
        
        # concatenate t_i, wealth, and asset prices
        Y = torch.zeros((self.T, batch_size, 1, self.env.n+2)).to(self.device)
        ones = torch.ones((batch_size,1)).to(self.device)
        zeros = 0*ones
        Y[0,...] = torch.cat( (zeros, 
                               self.X0*ones, 
                               X[0,:,:]), axis=1).unsqueeze(axis=1)
        
        # push through the neural-net to get weights
        theta = torch.zeros((self.T, batch_size,  self.d)).to(self.device)
        var_theta = torch.zeros((self.T, batch_size,  self.d)).to(self.device)
        beta = torch.zeros((self.T, batch_size,  self.d)).to(self.device)
        
        h[1,...], theta[0,:,:] = self.theta[0](h[0,...], Y[0,...])
        var_theta[0,:,:] = theta[0,:,:].clone()
    
        # stores wealth process
        wealth = torch.zeros((self.T+1, batch_size)).to(self.device)
        wealth[0,:] = torch.sum(var_theta[0,:,:] * X[0,:,:], axis=1)
        
        beta[0,:,:] = var_theta[0,:,:].clone() * X[0,...] / wealth[0,:].unsqueeze(axis=1).clone()
        
        w = torch.zeros((self.T, batch_size)).to(self.device)

        for t in range(1, self.T):
            
            # new wealth at period end
            wealth[t,:] = torch.sum( var_theta[t-1,:,:] * X[t,:,:], axis=1).clone()
            
            # concatenate t_i, and asset prices    
            Y[t,...] = torch.cat(((t*self.dt)*ones, 
                                  w[t-1,:].reshape(-1,1).detach(),
                                  X[t,:,:]), axis=1).unsqueeze(axis=1)
            
            # push through the neural-net to get new number of assets
            h[t+1,...], theta[t,:,:] = self.theta[t](h[t,...].detach(),
                                                     Y[t,...].detach())
            
            w[t-1,:] = torch.sum(theta[t-1,:,:].detach() * X[t,:,:], axis=1) \
                / torch.sum(theta[t,:,:].detach() * X[t,:,:], axis=1) 
            # w[t-1,:] = 1.0
            
            var_theta[t,:,:] = theta[t,:,:].clone() * torch.prod(w[:t,:], axis=0).reshape(-1,1)
            
            beta[t,:,:] = var_theta[t,:,:].detach() * X[t,...] / wealth[t,:].unsqueeze(axis=1).detach()
            
        wealth[-1,:] = torch.sum( var_theta[-1,:,:] * X[-1,:,:], axis=1).clone()
            
        costs = -torch.diff(wealth, axis=0)
        
        return costs, h, Y, beta, theta, var_theta, w, X, wealth
    
    def __V_Score__(self, VaR, CVaR, mu, X):
        
        C = 2
        xmax = torch.max(torch.abs(X))
        if xmax > C:
            C = 2*xmax
            
        A = ((X<=VaR)*1.0-self.alpha)*VaR + (X>VaR)*X
        B = (CVaR+C)*(1.0-self.alpha)
        
        # for conditional VaR & CVaR
        score = torch.mean(torch.log( (CVaR+C)/(X+C)) - (CVaR/(CVaR+C)) + A/B)
        
        # for conditional mean
        score += torch.mean((mu-X)**2)
        
        return score
    
    def __update_valuefunction__(self, N_iter  = 100, batch_size = 256, n_print=100):
        
        for j in range(N_iter):
            
            costs, h, Y, beta, theta, var_theta, w, X, wealth = self.__run_epoch__(batch_size)
        
            for t in range(self.T-1, -1, -1):
                
                self.psi_optimizer[t].zero_grad()
                self.chi_optimizer[t].zero_grad()
                self.mu_optimizer[t].zero_grad()
                
                dX = -torch.diff(X, axis=0)
                theta = theta.detach()
                h = h.detach()
                Y = Y.detach()

                # A = theta_t' dX_t + w_t V_{t+1}
                A = torch.sum(theta[t,:,:] * dX[t,:,:], axis=1).reshape(-1,1)
                
                if t < self.T-1:
                    A += w[t,:].reshape(-1,1) \
                        * self.risk_measure_target(t+1,
                                                   h[t+1,...].transpose(0,1),
                                                   Y[t+1,...])
                
                loss = self.__V_Score__(self.VaR(t,
                                                 h[t,...].transpose(0,1),
                                                 Y[t,...]),
                                       self.CVaR(t,
                                                 h[t,...].transpose(0,1),
                                                 Y[t,...]),
                                       self.mu[t](h[t,...].transpose(0,1),
                                                  Y[t,...]),
                                       A.detach())

                loss.backward()
                
                self.psi_optimizer[t].step()
                self.chi_optimizer[t].step()
                self.mu_optimizer[t].step()
                
                self.psi_scheduler[t].step()
                self.chi_scheduler[t].step()
                self.mu_scheduler[t].step()            
            
        self.chi_target = copy.deepcopy(self.chi)
        self.psi_target = copy.deepcopy(self.psi)
        self.mu_target = copy.deepcopy(self.mu)
        
    def __F_Score__(self, t, h, Y, X):
        
        max_z = 1
        
        N = 501
        z = torch.linspace(-max_z, max_z,N).reshape(N,1,1,1).repeat(1,Y.shape[0], Y.shape[1], 1).to(self.device)
        dz = z[1,0,0,0]-z[0,0,0,0]              
        
        Z = torch.concat((Y.unsqueeze(axis=0).repeat(N,1,1,1), 
                          z), 
                         axis=3)
        
        F = self.F[t](h.unsqueeze(axis=0).repeat(N,1,1,1), Z)
        
        score = torch.mean( torch.sum((F-1.0*(z[...,0]>=X.unsqueeze(axis=0).repeat(N,1,1)))**2*dz, 
                                      axis=0) )
        
        # add increasing penalty
        d_dz_F = torch.diff(F, axis=0)/dz
        score += torch.mean( torch.sum(d_dz_F**2*(d_dz_F<0)*dz), axis=0 )
        
        return score          
        

    def __update_F__(self, N_iter  = 100, batch_size = 256, n_print=100):
        
        for j in range(N_iter):
            
            costs, h, Y, beta, theta, var_theta, w, X, wealth = self.__run_epoch__(batch_size)
        
            for t in range(self.T-1, -1, -1):
                
                dX = -torch.diff(X, axis=0)
                theta = theta.detach()
                h = h.detach()
                Y = Y.detach()
                
                self.F_optimizer[t].zero_grad()
                
                A = torch.sum( theta[t,:,:] * dX[t,:,:], axis=1).reshape(-1,1)
                        
                if t < self.T-1:
                    A += w[t,:].reshape(-1,1)*\
                        self.risk_measure_target(t+1,
                                                 h[t+1,...].transpose(0,1),
                                                 Y[t+1,...])
                
                loss = self.__F_Score__(t,
                                        h[t,...].transpose(0,1),
                                        Y[t,...],
                                        A.detach())                
                
                loss.backward()
                
                self.F_optimizer[t].step()
                self.F_scheduler[t].step()
            
    def __get_v_gamma__(self, theta, dX, w, h, Y):
        
            batch_size = theta.shape[1]
            
            # compute the value function at points in time
            V = torch.zeros((self.T+1, batch_size)).to(self.device)
            for t in range(self.T):
                V[t,:] = self.risk_measure_target(t,
                                                  h=h[t,...].transpose(0,1), 
                                                  Y=Y[t,...]).squeeze()
            
            U = torch.zeros((self.T, batch_size)).to(self.device)
            
            for t in range(self.T):
                
                A = torch.sum( theta[t,:,:] * dX[t,:,:], axis=1).reshape(-1,1) \
                    + w[t,:].reshape(-1,1) * V[t+1,:].reshape(-1,1)
                
                Z = torch.concat((Y[t,...],
                                  A.unsqueeze(axis=2)),
                                 axis=2)
                
                U[t,:] = self.F[t](h[t,...].transpose(0,1), Z)[:,0]
                
            Gamma = self.gamma(U)
                
            return V, Gamma
        
    def __get_gateaux_terms__(self, batch_size=2048):
        
        costs, h, Y, beta, theta_grad, var_theta, w, X, wealth = self.__run_epoch__(batch_size)

        theta = theta_grad.detach()
        dX = -torch.diff(X, axis=0)
        h = h.detach()
        Y = Y.detach()
        
        V,  Gamma = self.__get_v_gamma__(theta,
                                         dX,
                                         w,
                                         h.detach(),
                                         Y.detach()) 
        
        Gamma = Gamma.detach()
        V = V.detach()
        
        Z = torch.zeros(theta.shape).to(self.device)
        for t in range(self.T):
            
            Z[t,:,:] =  dX[t,:,:]
            
            if t < self.T-1:
                Z[t,:,:] += (X[t+1,:,:]/ torch.sum(theta[t+1,:,:] * X[t+1,:,:], axis=1).reshape(-1,1)) \
                    * V[t+1,:].reshape(-1,1)
                
            Z[t,:,:] *= theta_grad[t,:,:] * Gamma[t,:].reshape(-1,1)
                
        return Z, theta_grad, V
        
    def RiskContributions(self, batch_size=2048):
                
        Z, theta_grad, V = self.__get_gateaux_terms__(batch_size)
        
        Z /= self.eta
        
        RC = torch.mean(Z, axis=1)
                
        RC_err  = torch.std( Z, axis=1 )/np.sqrt(batch_size)

        return RC, RC_err  
    
    def __update_policy__(self, N_iter=100, batch_size = 256):
        """
        this updates the policy -- here policy = theta
        """
        
        for k in range(N_iter):
            
            for t in range(self.T-1, -1, -1):
                
                i_rnd = np.random.permutation(np.arange(self.d))
                
                loss = 0
                
                Z, theta_grad, V = self.__get_gateaux_terms__(batch_size)
                
                for i in i_rnd:
                    
                    self.theta_optimizer[t].zero_grad()
                    
                    loss += torch.mean(Z[t,:,i]) \
                        - self.eta*torch.mean( self.B[t,:,i] * torch.log(theta_grad[t,:,i]) )
        
                loss.backward()
    
                self.theta_optimizer[t].step()
                self.theta_scheduler[t].step()
            
            self.V.append(torch.mean(V, axis=1).detach().cpu().numpy())
            
            RC, RC_err = self.RiskContributions(500)
            self.RC.append(RC.cpu().detach().numpy())  
                
    def Train(self, n_iter=10_000, n_print = 10, M_value_iter = 10, M_policy_iter=1, batch_size=256, name=""):
        
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
            self.__update_F__(N_iter=5, n_print=500, batch_size=batch_size)
            
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
        
        costs, h, Y, beta, theta, var_theta, w, X, wealth = self.__run_epoch__(batch_size)
        
        V,  Gamma = self.__get_v_gamma__(theta.detach(),
                                         -torch.diff(X, axis=0),
                                         w,
                                         h,
                                         Y)

        costs = costs.cpu().detach().numpy()
        V = V.cpu().detach().numpy()
        
        def kde(x, bins):
            h = np.std(x)*1.06*(len(x))**(-1/5)
            
            h += 1e-6
            dx = x.reshape(1,-1)-bins.reshape(-1,1)
            
            f = np.sum(np.exp(-0.5*dx**2/h**2)/np.sqrt(2*np.pi*h**2), axis=1)/len(x)
            
            return f
        
        bins=np.linspace(-0.5, 0.5, 51)
        kde_bins = np.linspace(-0.5, 0.5, 501)
        
        V_bins=np.linspace(0, 0.25, 51)
        V_kde_bins = np.linspace(0, 0.25, 501)
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
        
        
        fig, ax = plt.subplots(nrows=self.T,ncols=self.d, figsize=(8,8))
        
        for t in range(self.T):
            
            for i in range(self.d):
                
                ax[t,i].hist(theta[t,:,i].detach().cpu().numpy())
                
        plt.tight_layout()
        plt.show()
        
        
        zeta = beta.cpu().detach().numpy()
        for t in range(self.T):
            plt.subplot(1,self.T,t+1)
            plt.title(r'$\beta_{' + str(t) +'}$')
            for i in range(self.d):
                plt.hist(zeta[t,:,i], alpha=0.6, bins=np.linspace(0,1,51), density=True)
                if t > 0:
                    plt.axvline(np.mean(zeta[t-1,:,i]),linestyle='--', color='k')
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
        
        fig = plt.figure(figsize=(10,4))
        
        plt.subplot(1,3,1)
        RC = np.array(self.RC)
        RC_flat = RC.reshape(len(self.RC),-1)
        for i in range(RC_flat.shape[1]):
            plt.plot(self.MovingAverage(RC_flat[:,i],100))
        plt.ylabel('RC')
        
        
        plt.subplot(1,3,2)
        for t in range(self.T):
            plt.plot(self.MovingAverage(np.sum(RC[:,t,:], axis=1),100), 
                     label=r'$\sum \frac{RC_{' + str(t) + ',i}}{V_{' + str(t) + ',i}}$', linewidth=1) 
            
        plt.axhline(1.0, linestyle='--', color='k')
        plt.ylim(0,2)
        plt.legend(fontsize=12)

        plt.subplot(1,3,3)
        V = np.array(self.V)
        for t in range(self.T):
            plt.plot(V[:,t], label=r'$V_{'+str(t) + '}$', linewidth=1)
        plt.axhline(self.eta, linestyle='--', color='k')
        plt.ylim(0,2*self.eta)
        plt.legend(fontsize=12)
        
        plt.tight_layout()
        plt.show()      
        
        plt.figure(figsize=(10,4))
        idx = 1
        
        for k in range(self.d):
            for j in range(self.T):
        
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
                
                plt.ylim(0, 2*self.B[j,0,k].cpu())
                plt.ylabel(r'$RC_{' + str(j) + ','+str(k)+ '}$')
                # plt.axhline(self.B[j,0,k].cpu()*self.eta/torch.sum(self.B[j,0,:]).cpu(), linestyle='--', color='k')
                plt.axhline(self.B[j,0,k].cpu(), linestyle='--', color='k')
                idx += 1
                
        plt.suptitle(r'$RC$')
        plt.tight_layout()
        plt.show()      
                
       
    def PlotPaths(self, batch_size=1, title=None):
        
        costs, h, Y, beta, theta, var_theta, w, S, wealth = self.__run_epoch__(batch_size)
        
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