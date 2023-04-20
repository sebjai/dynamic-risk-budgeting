# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 17:02:13 2023


@author: jaimunga
"""

import torch
import torch.nn as nn
import torch.optim as optim

from heston import heston
from tqdm import tqdm

import pdb
import numpy as np
import matplotlib.pyplot as plt

import copy

import dill
from datetime import datetime

class net(nn.Module):
    
    def __init__(self, 
                 nIn, 
                 nOut, 
                 gru_hidden=5, 
                 gru_layers=5, 
                 linear_hidden = 36, 
                 linear_layers=5,
                 device='cpu',
                 out_activation=None,
                 out_factor =1):
        super(net, self).__init__()

        self.out_activation = out_activation
        self.out_factor = out_factor
        self.device = device
        self.gru = torch.nn.GRU(input_size=nIn, 
                                hidden_size=gru_hidden, 
                                num_layers=gru_layers, 
                                batch_first=True).to(self.device)
        
        self.gru_to_hidden = nn.Linear(gru_hidden*gru_layers+nIn, linear_hidden).to(self.device)
        self.linear_hidden_to_hidden = nn.ModuleList([nn.Linear(linear_hidden, linear_hidden).to(self.device) for i in range(linear_layers-1)])
        self.hidden_to_out = nn.Linear(linear_hidden, nOut).to(self.device)
        
        self.g = nn.SiLU()
        self.softmax = nn.Softmax(dim=1)
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()
        
        
    def forward(self, h, Y, z = None):
        
        self.gru.flatten_parameters()
        
        _, h_out = self.gru(Y.clone(), h.clone())
        
        if z is None:
            x = torch.cat((h.transpose(0,1).flatten(start_dim=-2), 
                           Y.flatten(start_dim=-2)),
                          axis=-1)
        else:
            
            if z.shape[:-1] == Y.shape[:-1]:
                h_flat = h.transpose(0,1).flatten(start_dim=-2)
                Y_flat = Y.flatten(start_dim=-2)
                z_flat = z.flatten(start_dim=-2)
            else:
                h_flat = h.transpose(0,1).flatten(start_dim=-2)
                h_flat = h_flat.unsqueeze(axis=1).repeat(1,z.shape[0],1)
                
                Y_flat = Y.flatten(start_dim=-2)
                Y_flat = Y_flat.unsqueeze(axis=1).repeat(1,z.shape[0],1)
                
                z_flat = z.unsqueeze(axis=0).repeat(Y_flat.shape[0],1).unsqueeze(axis=-1)
                
            x = torch.cat((h_flat, 
                           Y_flat,
                           z_flat),
                          axis=-1)
        
        x = self.gru_to_hidden(x)
        x = self.g(x)
        
        for linear in self.linear_hidden_to_hidden:
            x = self.g(linear(x))
            
        output = self.hidden_to_out(x).squeeze()
        
        if self.out_activation == 'softplus':
            output = self.softplus(output)
        elif self.out_activation == 'sigmoid':
            output = self.sigmoid(output)
        elif self.out_activation == 'softmax':
            output = self.softmax(output)
            
        output = self.out_factor * output
        
        if z is not None:
            output = output.unsqueeze(axis=1)
        
        return h_out, output
    
        
class dynamic_risk_budget():
    
    def __init__(self, env : heston, X0=1, B = 0, alpha=0.8, p=0.5):
        
        # set the device to use
        if torch.cuda.device_count() > 0:
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')
            
        self.env = env
        self.wealth_0 = X0        
        self.T = env.T
        self.d = env.n
        self.dt = env.dt
        self.alpha = alpha
        self.p = p
        self.gamma = lambda u : self.p*(u>self.alpha)/(1-alpha) + (1-self.p)
        
        if B == 0:
            self.B = torch.ones((self.T, 1, self.d))/(self.T*self.d)
        else:
            self.B = torch.tensor(B).float()
            
        self.B /= torch.sum(self.B,axis=2).unsqueeze(axis=2)
        self.B = self.B.to(self.device)
        
        self.__initialize_nets__(self.device)
        
        # for storing losses
        self.VaR_CVaR_mean_loss = []
        self.F_loss = []
        self.theta_loss = []

        self.eta = 0.1 # the Lagrange multiplier factor        
        
        self.RC = []
        self.V = []
        
    def __get_optim_sched__(self, A):
        
        optimizer = optim.AdamW(A.parameters(),
                                lr=0.001)
                    
        scheduler = optim.lr_scheduler.StepLR(optimizer,
                                              step_size=100,
                                              gamma=0.99)
    
        return optimizer, scheduler

    def __initialize_nets__(self, device='cpu'):
        
        # the theta network...
        self.theta = []
        self.theta_optimizer = []
        self.theta_scheduler = []
        for t in range(self.T):
            self.theta.append(net(nIn=self.d+2,
                                nOut=self.d,
                                gru_hidden=self.env.n,
                                gru_layers=5,
                                linear_hidden=32,
                                linear_layers=5,
                                device=self.device,
                                out_activation='sigmoid',
                                out_factor=10))
            
            if t > 0:
                self.theta[-1].gru = self.theta[0].gru
            
            optimizer, scheduler = self.__get_optim_sched__(self.theta[t])
            self.theta_optimizer.append(optimizer)
            self.theta_scheduler.append(scheduler)        
    
        # for conditional cdf of c_t + V_{t+1} 
        self.F = net(nIn=self.d+2, nOut=1, 
                     gru_hidden=self.env.n, gru_layers=5,
                     linear_hidden=32, linear_layers=5,
                     out_activation='sigmoid', device=self.device)
        # the gru to hidden has to take in one extra feature = z
        self.F.gru_to_hidden = nn.Linear(self.F.gru.hidden_size*self.F.gru.num_layers+self.d+3, 
                                         self.F.hidden_to_out.in_features).to(self.device)
        
        self.F_optimizer, self.F_scheduler = self.__get_optim_sched__(self.F)
        
        # for conditional mu, VaR, CVaR-VaR -- need to apply softplus to 3rd output
        self.cond_RM = net(nIn=self.d+2, nOut=3,
                               gru_hidden=self.env.n, gru_layers=5,
                               linear_hidden=32, linear_layers=5,
                               device=self.device)
        self.cond_RM.gru = self.F.gru
        self.cond_RM_target = copy.deepcopy(self.cond_RM)
        self.cond_RM_optimizer, self.cond_RM_scheduler = self.__get_optim_sched__(self.cond_RM)
                
        self.risk_measure = lambda mu, CVaR :  (self.p * CVaR + (1.0-self.p)* mu)
        
    def __strip_cond_RM__(self, cond_RM):
        
        mu = cond_RM[...,0]
        psi = cond_RM[...,1]
        chi = self.cond_RM.softplus( cond_RM[...,2] )
        
        return mu, psi, psi+chi
        
    def __run_epoch__(self, batch_size = 256):
        
        # grab a simulation of the market prices
        X, _ = self.env.simulate(batch_size)
        X = torch.tensor(X).float().transpose(0,1).to(self.device)
        
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
                               self.wealth_0*ones, 
                               X[0,:,:]), axis=1).unsqueeze(axis=1)
        
        # push through the neural-net to get weights
        theta = torch.zeros((self.T, batch_size,  self.d)).to(self.device)
        var_theta = torch.zeros((self.T, batch_size,  self.d)).to(self.device)
        beta = torch.zeros((self.T, batch_size,  self.d)).to(self.device)
        
        h[1,...], theta[0,:,:] = self.theta[0](h[0,...],
                                               Y[0,...])
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
            # Y[t,...] = torch.cat(((t*self.dt)*ones, 
            #                       torch.zeros(batch_size,1).to(self.device), #wealth[t,:].reshape(-1,1).detach(),
            #                       X[t,:,:]), axis=1).unsqueeze(axis=1)
            
            Y[t,...] = torch.cat(((t*self.dt)*ones, 
                                  wealth[t,:].reshape(-1,1).detach(),
                                  X[t,:,:]), axis=1).unsqueeze(axis=1)
            
            # push through the neural-net to get new number of assets
            h[t+1,...], theta[t,:,:] = self.theta[t](h[t,...].detach(),
                                                     Y[t,...].detach())
            
            w[t-1,:] = torch.sum(theta[t-1,:,:].detach() * X[t,:,:], axis=1) \
                / torch.sum(theta[t,:,:].detach() * X[t,:,:], axis=1) 
            
            var_theta[t,:,:] = theta[t,:,:].clone() * torch.prod(w[:t,:], axis=0).reshape(-1,1)
            
            beta[t,:,:] = var_theta[t,:,:].detach() * X[t,...] / wealth[t,:].unsqueeze(axis=1).detach()
            
        wealth[-1,:] = torch.sum( var_theta[-1,:,:] * X[-1,:,:], axis=1).clone()
            
        costs = -torch.diff(wealth, axis=0)
        
        return costs, h, Y, beta, theta, var_theta, w, X, wealth
    
    def __V_Score__(self, VaR, CVaR, mu, X):
        
        C = 1
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

    def get_risk_measure(self, Y, target = False):
        
        h = torch.zeros((self.T+1,
                         self.cond_RM.gru.num_layers, 
                         Y.shape[1], 
                         self.cond_RM.gru.hidden_size)).to(self.device)
        
        mu = torch.zeros(self.T, Y.shape[1]).to(self.device)
        VaR = mu.clone()
        CVaR = mu.clone()
        
        for t in range(self.T):
            
            if not target:
                h[t+1,...], cond_RM = self.cond_RM(h[t,...], Y[t,...])
            else:
                h[t+1,...], cond_RM = self.cond_RM_target(h[t,...], Y[t,...])
                
            mu[t,...], VaR[t,...], CVaR[t,...] = self.__strip_cond_RM__(cond_RM)
            
        mu = mu.unsqueeze(axis=-1)    
        VaR = VaR.unsqueeze(axis=-1)
        CVaR = CVaR.unsqueeze(axis=-1)
        
        return mu, VaR, CVaR
    
    
    def __update_risktogo__(self, N_iter  = 100, batch_size = 256, n_print=100):
        
        for j in range(N_iter):
            
            costs, _, Y, beta, theta, var_theta, w, X, wealth = self.__run_epoch__(batch_size)
            
            dX = -torch.diff(X, axis=0)
            theta = theta.detach()            
            Y = Y.detach()
            
            mu_target, VaR_target, CVaR_target = self.get_risk_measure(Y, target=True)
            mu_target = mu_target.detach()
            CVaR_target = CVaR_target.detach()
            
            mu, VaR, CVaR = self.get_risk_measure(Y, target=False)

            self.cond_RM_optimizer.zero_grad()
            
            loss = 0
            for t in range(self.T):
                
                # A = theta_t' dX_t + w_t V_{t+1}
                A = torch.sum(theta[t,:,:] * dX[t,:,:], axis=1).reshape(-1,1)
                
                if t < self.T-1:
                    A += w[t,:].reshape(-1,1) *  self.risk_measure(mu_target[t+1,...],
                                                                   CVaR_target[t+1,...])
                    
                loss += self.__V_Score__(VaR[t,...], 
                                         CVaR[t,...], 
                                         mu[t,...], 
                                         A.detach())

            loss.backward()
            
            self.cond_RM_optimizer.step()
            self.cond_RM_scheduler.step()
            
            self.VaR_CVaR_mean_loss.append(loss.item())
            
        self.cond_RM_target = copy.deepcopy(self.cond_RM)
        
    def get_F(self, Y):
        
        h = torch.zeros((self.T+1,
                         self.F.gru.num_layers, 
                         Y.shape[1], 
                         self.F.gru.hidden_size)).to(self.device)
        
        max_z = 5
        
        N = 501
        z = torch.linspace(-max_z, max_z,N).to(self.device)
        
        F = torch.zeros(self.T, Y.shape[1], Y.shape[2], N).to(self.device)
        
        for t in range(self.T):
            
            h[t+1,...], F[t,...] = self.F(h[t,...],
                                          Y[t,...],
                                          z)
            
        return z, F
        
    def __F_Score__(self, z, F, X):
        
        dz = z[1]-z[0]
                
        # score for eliciting conditional cdf
        score = torch.mean( torch.sum((F[:,0,:]-1.0*(z.unsqueeze(axis=0)>=X))**2*dz, 
                                      axis=1) )
        
        
        # add increasing penalty
        d_dz_F = torch.diff(F, axis=-1)/dz
        score += torch.mean( torch.sum(d_dz_F**2*(d_dz_F<0)*dz, axis=-1))
        
        return score          
        

    def __update_F__(self, N_iter  = 100, batch_size = 256, n_print=100):
        
        for j in range(N_iter):
        
            costs, _, Y, beta, theta, var_theta, w, X, wealth = self.__run_epoch__(batch_size)
            
            dX = -torch.diff(X, axis=0)
            theta = theta.detach()
            Y = Y.detach()
            
            mu_target, VaR_target, CVaR_target = self.get_risk_measure(Y, target=True)
            
            mu_target = mu_target.detach()
            CVaR_target = CVaR_target.detach()
            
            z, F = self.get_F(Y)
            
            self.F_optimizer.zero_grad()
            
            loss = 0.0
            
            for t in range(self.T):
                
                A = torch.sum( theta[t,:,:] * dX[t,:,:], axis=1).reshape(-1,1)
                        
                if t < self.T-1:
                    A += w[t,:].reshape(-1,1)* self.risk_measure(mu_target[t+1,...],
                                                                 CVaR_target[t+1,...])
                
                loss += self.__F_Score__(z, F[t,...], A.detach())                
                
            loss.backward()
            
            self.F_optimizer.step()
            self.F_scheduler.step()
            
    def __get_v_gamma__(self, theta, dX, w, Y):

        
        batch_size = theta.shape[1]
        
        mu, VaR, CVaR = self.get_risk_measure(Y, target=True)
        
        # compute the value function at points in time
        V = torch.zeros((self.T+1, batch_size)).to(self.device)
        for t in range(self.T):
            V[t,:] = self.risk_measure(mu[t,...], CVaR[t,...]).squeeze()
        
        U = torch.zeros((self.T, batch_size)).to(self.device)

        h = torch.zeros((self.T+1,
                         self.F.gru.num_layers, 
                         Y.shape[1], 
                         self.F.gru.hidden_size)).to(self.device)        
        for t in range(self.T):
            
            A = torch.sum( theta[t,:,:] * dX[t,:,:], axis=1).reshape(-1,1)
            A += w[t,:].reshape(-1,1) * V[t+1,:].reshape(-1,1)
            
            h[t+1,...], u = self.F(h[t,...],
                                   Y[t,...],
                                   A.unsqueeze(axis=-1))
            U[t,:] = u.squeeze()
            
        Gamma = self.gamma(U)
            
        return V, Gamma
        
    def __get_gateaux_terms__(self, batch_size=2048):

        costs, h, Y, beta, theta_grad, var_theta, w, X, wealth = self.__run_epoch__(batch_size)

        theta = theta_grad.detach()
        dX = -torch.diff(X, axis=0)
        h = h.detach()
        Y = Y.detach()
        
        V,  Gamma = self.__get_v_gamma__(theta, dX, w, Y) 
        
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
        
        #Z /= V[:-1,:].reshape(self.T, batch_size,1)
        Z /= self.eta
        
        RC = torch.mean(Z, axis=1)
                
        RC_err  = torch.std( Z, axis=1 )/np.sqrt(batch_size)

        return RC, RC_err  
    
    def __update_policy__(self, N_iter=100, batch_size = 256):
        """
        this updates the policy -- here policy = theta
        """
        
        for k in range(N_iter):
            
            t_rnd = np.random.permutation(np.arange(self.T))
            for t in t_rnd:
                
                i_rnd = np.random.permutation(np.arange(self.d))
                
                for i in i_rnd:
                    
                    Z, theta_grad, V = self.__get_gateaux_terms__(batch_size)
                    
                    self.theta_optimizer[t].zero_grad()
                    
                    loss = torch.mean(Z[t,:,i]) \
                        - self.eta*torch.mean( self.B[t,:,i] * torch.log(theta_grad[t,:,i]) )
        
                    loss.backward()
        
                    self.theta_optimizer[t].step()
                    self.theta_scheduler[t].step()
            
            self.V.append(torch.mean(V, axis=1).detach().cpu().numpy())
            
            RC, RC_err = self.RiskContributions(500)
            self.RC.append(RC.cpu().detach().numpy())  
            
    def print_state(self, n_iter, M_value_iter, M_F_iter, M_policy_iter, batch_size):
        
        print('T : ' , self.T)
        print('d : ', self.d)
        print('n_iter : ', n_iter)
        print('M_value_iter : ', M_value_iter)
        print('M_F_iter : ', M_F_iter)
        print('M_policy_iter : ', M_policy_iter)
        print('batch_size : ', batch_size)
        
        
            
    def train(self, 
              n_iter=10_000, 
              n_print = 10, 
              M_value_iter = 10, 
              M_F_iter=10, 
              M_policy_iter=1, 
              batch_size=256, name=""):
        
        self.print_state(n_iter, M_value_iter, M_F_iter, M_policy_iter, batch_size)
        
        print("training value function on initialization...")
        self.__update_risktogo__(N_iter= 10, n_print=500, batch_size=batch_size)
        print("training F on initialization...")
        self.__update_F__(N_iter= 10, n_print= 500, batch_size=batch_size)

        print("main training...")
        self.plot_paths(500)

        count = 0
        for i in tqdm(range(n_iter)):
            
            # this updates mu, psi, chi
            self.__update_risktogo__(N_iter=M_value_iter, n_print=500, batch_size=batch_size)
            
            # this udpates F
            self.__update_F__(N_iter=M_F_iter, n_print=500, batch_size=batch_size)
        
            # this updates beta and w_0
            self.__update_policy__(N_iter=M_policy_iter, batch_size=batch_size)
            
            count += 1
            
            if np.mod(count, n_print)==0:
                
                self.plot_summary()
                self.plot_paths(500)
                self.plot_hist()
                
                date_time = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
                dill.dump(self, open(name + '_' + date_time + '.pkl','wb'))   
                
    def plot_hist(self, batch_size = 10_000):
        
        costs, h, Y, beta, theta, var_theta, w, X, wealth = self.__run_epoch__(batch_size)
        
        V,  Gamma = self.__get_v_gamma__(theta.detach(),
                                         -torch.diff(X, axis=0),
                                         w, Y)

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
        
        V_bins=np.linspace(0.05, 0.15, 51)
        V_kde_bins = np.linspace(0.05, 0.15, 501)
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
            if i > 0:
                plt.hist(V[i,:], bins=V_bins, density=True, alpha=0.5)
                f = kde(V[i,:], V_kde_bins)
                plt.plot(V_kde_bins,f,linewidth=1,color='r')
            else:
                plt.axvline(V[0,0],linewidth=1,color='r')
            plt.xlabel(r'$V_'+str(i) + '$', fontsize=16)
            
        plt.tight_layout()
        plt.show()
        
        
        fig, ax = plt.subplots(nrows=self.T,ncols=self.d, figsize=(8,8))
        
        if self.T > 1:
            for t in range(self.T):
                
                for i in range(self.d):
                    
                    ax[t,i].hist(theta[t,:,i].detach().cpu().numpy())
                    
        else:
            
            for i in range(self.d):
                
                ax[i].hist(theta[0,:,i].detach().cpu().numpy())
                
        plt.tight_layout()
        plt.show()
            
    def moving_average(self,x, n):
        
        y = np.zeros(len(x))
        y[0] = np.nan
        
        for i in range(1,len(x)):
            if i < n:
                y[i] = np.mean(x[:i])
            else:
                y[i] = np.mean(x[i-n:i])
                
        return y
        
    def plot_summary(self):

        RC = np.array(self.RC)
        
        fig = plt.figure(figsize=(10,4))
        
        plt.subplot(1,2,1)
        for t in range(self.T):
            plt.plot(self.moving_average(np.sum(RC[:,t,:], axis=1), 100), 
                     label=r'$\frac{1}{\eta}\sum RC_{' + str(t) + ',i}$', linewidth=1) 
            
        plt.axhline(1.0, linestyle='--', color='k')
        plt.ylim(0.5, 2)
        plt.legend(fontsize=12,loc='upper right')
        plt.yticks(fontsize=14)
        plt.xticks(fontsize=14)
        
        plt.subplot(1,2,2)
        V = np.array(self.V)
        for t in range(self.T):
            plt.plot(V[:,t], linewidth=1, alpha=0.25)
            plt.plot(self.moving_average(V[:,t],100), label=r'$V_{'+str(t) + '}$', linewidth=1.5)
        plt.axhline(self.eta, linestyle='--', color='k')
        plt.ylim(0.5*self.eta,2*self.eta)
        plt.legend(fontsize=12)
        plt.yticks(fontsize=14)
        plt.xticks(fontsize=14)
        
        plt.tight_layout(pad=1.2)
        
        plt.savefig('sumRC_V.pdf', format='pdf')
        plt.show()      
        
        plt.figure(figsize=(8,4))
        idx = 1
        
        for k in range(self.d):
            for j in range(self.T):
        
                RC_ma = (self.moving_average(RC[:,j,k],50)) 
                
                plt.subplot(self.d, self.T, idx)
                plt.plot(RC[:,j,k],alpha=0.5)
                plt.plot(RC_ma)                
                
                plt.ylim(0, 1)
                plt.ylabel(r'$RC_{' + str(j) + ','+str(k)+ '}$')
                # plt.axhline(self.B[j,0,k].cpu()*self.eta/torch.sum(self.B[j,0,:]).cpu(), linestyle='--', color='k')
                plt.axhline(self.B[j,0,k].cpu(), linestyle='--', color='k')
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                idx += 1
                
        plt.suptitle(r'$RC$')
        plt.tight_layout()
        plt.savefig('RC.pdf', format='pdf')
        plt.show()      
                
       
    def plot_paths(self, batch_size=1, title=None):
        
        costs, h, Y, beta, theta, var_theta, w, S, wealth = self.__run_epoch__(batch_size)
        
        costs = costs.cpu().detach().numpy()
        h = h.cpu().detach().numpy()
        Y = Y.cpu().detach().numpy()
        beta = beta.cpu().detach().numpy()
        S = S.cpu().detach().numpy()
        wealth = wealth.cpu().detach().numpy()
        
        
        fig, ax = plt.subplots(nrows=self.d,ncols=1, figsize=(3.2,5.6))
        
        qtl_S = np.quantile(S, [0.1,0.9], axis=1)
        qtl_beta = np.quantile(beta, [0.1,0.9], axis=1)
        
        for j in range(self.d):
                # ax[j,0].set_ylabel(r'$S^{0:2d}$'.format(j+1))
                # ax[j,0].plot(S[:,:,j], alpha=0.1)
                # ax[j,0].plot(S[:,0,j], color='r', linewidth=1)
                # ax[j,0].plot(qtl_S[:,:,j].T, color='k', linewidth=1)
                # ax[j,0].set_ylim(0.7, 1.4)
                # ax[j,1].set_xticks([0,1,2])
                
                ax[j].set_ylabel(r'$\beta^{0:2d}$'.format(j+1))
                ax[j].plot(beta[:,:,j], alpha=0.1)
                ax[j].plot(beta[:,0,j], color='r', linewidth=1)
                ax[j].plot(qtl_beta[:,:,j].T, color='k', linewidth=1)
                ax[j].set_ylim(0, 2.0/self.d)
                ax[j].set_xticks(np.arange(self.T))
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                
                
        if title is not None:
            plt.suptitle(title)
            plt.tight_layout()
            
            fig.savefig(title)
        
        plt.tight_layout()
        plt.show()
        
        for j in range(1, self.T):
            
            fig = plt.figure(figsize=(16,6.4))
            
            for k in range(self.d):
                
                ax = plt.subplot(1,self.d,k+1)
                
                plt.title(r'$\beta_{0:1d}^{1:1d}$'.format(j,k+1), fontsize=20)
                im1=plt.scatter(S[j,:,0], S[j,:,1], 
                                s=10, alpha=0.6, c=beta[j,:,k], 
                                cmap='brg', vmin=0, vmax=2.0/self.d)
                plt.scatter(S[j,0,0],S[j,0,1],
                            s=100, 
                            c = beta[j,0,k], 
                            facecolors=None,
                            edgecolors='k')
                plt.xlabel(r'$S_{0:1d}^1$'.format(j),fontsize=20)
                plt.ylabel(r'$S_{0:1d}^2$'.format(j),fontsize=20)
                plt.xticks(np.linspace(0.8,1.2,5), fontsize=14)
                plt.yticks(np.linspace(0.8,1.2,5), fontsize=14)
                plt.xlim(0.8, 1.2)
                plt.ylim(0.8, 1.2)
                
            plt.tight_layout(pad=2)
            
            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.82, 0.2, 0.01, 0.65])
            cbar = fig.colorbar(im1, cax=cbar_ax, shrink=0.6)
            cbar.ax.set_yticks(np.linspace(0,2.0/self.d,5))
            cbar.ax.set_yticklabels(["{:.1%}".format(i) for i in np.linspace(0,2.0/self.d,5)], fontsize=12)
        
            plt.savefig('beta-' + str(j) + '.pdf', format='pdf')
            plt.show()
        
        return S, beta