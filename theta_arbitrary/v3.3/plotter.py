# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 10:08:18 2023

@author: jaimunga
"""

import numpy as np
import torch
from dynamic_risk_budget import dynamic_risk_budget
from heston import heston
import matplotlib.pyplot as plt
from scipy.stats import norm

import pdb

class plotter():
    
    def __init__(self, drb : dynamic_risk_budget):
        
        self.drb = drb
        plt.style.use('paper.mplstyle')
        
        self.kde = lambda x, x_d, h: np.mean(norm.pdf( (x.reshape(-1,1)- x_d.reshape(1,-1))/h)/h, axis=1)
        
    def moving_average(self, x, n):
        
        y = np.zeros(len(x))
        y_err = np.zeros(len(x))
        y[0] = np.nan
        y_err[0] = np.nan
        
        for i in range(1,len(x)):
            
            if i < n:
                y[i] = np.mean(x[:i])
                y_err[i] = np.std(x[:i])
            else:
                y[i] = np.mean(x[i-n:i])
                y_err[i] = np.std(x[i-n:i])
                
        return y, y_err
        
    def plot_summary(self):

        RC = np.array(self.drb.RC)
        
        fig = plt.figure(figsize=(10,4))
        
        plt.subplot(1,2,1)
        for t in range(self.drb.T):
            
            mv, mv_err = self.moving_average(np.sum(RC[:,t,:], axis=1), 100)
            
            plt.fill_between(np.arange(len(mv)), mv-mv_err, mv+mv_err, alpha=0.2)
            plt.plot(mv, label=r'$\sum RC_{' + str(t) + ',i}$', linewidth=1) 
            
        plt.axhline(1.0, linestyle='--', color='k', alpha=0.8)
        
        plt.legend(fontsize=14,loc='upper right')
        plt.yticks(fontsize=14)
        plt.xticks(fontsize=14, ticks=[0, 500, 1000, 1500, 2000, 2500])
        
        plt.ylim(0.5, 2)
        plt.xlim(0,3000)
        
        plt.subplot(1,2,2)
        V = np.array(self.drb.V)
        for t in range(self.drb.T):
            # plt.plot(V[:,t], linewidth=1, alpha=0.2)
            
            mv, mv_err = self.moving_average(V[:,t]/self.drb.eta,100)
            
            plt.fill_between(np.arange(len(mv)), y1=mv-mv_err, y2=mv+mv_err, alpha=0.2)
            plt.plot(mv, label=r'$\mathfrak{R}_{'+str(t) + '}$', linewidth=1.5)
            
        plt.axhline(1, linestyle='--', color='k', alpha=0.8)
        
        plt.ylim(0.5, 2)
        plt.xlim(0,3000)
        
        plt.legend(fontsize=16)
        plt.yticks(fontsize=14)
        plt.xticks(fontsize=14, ticks=[0, 500, 1000, 1500, 2000, 2500])
        
        plt.tight_layout(pad=2)
        
        plt.savefig(self.drb.name + '_' + 'sumRC_V.pdf', format='pdf')
        plt.show()      
        
    def plot_RC(self):
        
        RC = np.array(self.drb.RC)
        
        fig, ax= plt.subplots(nrows=self.drb.d, ncols=self.drb.T,
                              sharex=True, sharey=True,
                              figsize=(8,6))
        
        B = self.drb.B.cpu().numpy()
        
        for k in range(self.drb.d):
            for j in range(self.drb.T):
        
                mv, mv_err = self.moving_average(RC[:,j,k],100)
                
                ax[k,j].fill_between(np.arange(len(mv)), y1=mv-mv_err, y2=mv+mv_err, alpha=0.5)
                ax[k,j].plot(mv, color='r')
                ax[k,j].axhline(self.drb.B[j,0,k].cpu(), linestyle='--', color='k', alpha=0.8)
                
                ax[k,j].set_ylim(0, 2*B[j,0,k])
                
                if j == self.drb.T-1:
                    
                    ax[k,j].set_ylabel(r'$i=' + str(k+1) +'$')
                    ax[k,j].yaxis.set_label_position('right')
                
                if k == 0:
                    ax[k,j].set_title(r'$t=' + str(j) +'$')
                    
                ax[k,j].set_xlim(0, 3000)
                ax[k,j].set_xticks([0, 1000, 2000], fontsize=12)
                ax[k,j].set_xticklabels([r'$0$',r'$1$',r'$2$'])
                ax[k,j].set_yticks([0.1, 0.3], fontsize=12)
                
        plt.tight_layout()
        plt.savefig(self.drb.name + '_RC.pdf', format='pdf')
        plt.show()
        
    def plot_wealth(self):
        
        print(self.drb.name)
        costs, Y, beta, theta, var_theta, w, X, wealth = self.drb.__run_epoch__(100_000)
        
        w = wealth.detach().cpu().numpy()
        
        w /= w[0,:]
        
        
        xmin = 0.7
        xmax = 1.3
        
        bins = np.linspace(xmin, xmax, 501)
        
        
        fig, ax = plt.subplots(1,self.drb.T-1)
        
        for t in range(1, self.drb.T):
            
            h = 1.06*np.std(w[t,:])*(len(w[t,:]))**(-1/5)
            kde = f(bins, w[t,:], h)
            
            ax[t-1].hist(w[t,:], np.linspace(xmin, xmax,101), density=True)
            ax[t-1].plot(bins, kde)
            ax[t-1].set_ylim(0, 9)
            
            qtl = np.quantile(w[t,:], 0.1)
            print(np.mean(w[t,:]), np.mean(w[t, w[t,:]<qtl]))
            
        plt.tight_layout(pad=2)
        plt.savefig(self.drb.name + '_wealth.pdf', format='pdf')
        plt.show()
        
        
    def plot_beta(self):
        
        costs, Y, beta, theta, var_theta, w, X, wealth = self.drb.__run_epoch__(5_000)
        
        beta=beta.detach().cpu().numpy()
        
        fig, ax= plt.subplots(nrows=self.drb.d, ncols=1,
                              sharex=True, sharey=True,
                              figsize=(4,7))
        
        xmin=0.05
        xmax=0.45
        bins = np.linspace(xmin,xmax,501)
        
        c = ['b', 'orange', 'g']
        
        for k in range(self.drb.d):
            
            for j in range(self.drb.T):
            
                if j > 0:
                    ax[k].hist(beta[j,:,k], np.linspace(xmin,xmax,101), density=True, alpha=0.5, color=c[j])
                    
                    h = 1.06*np.std(beta[j,:,k])*(len(beta[j,:,k]))**(-1/5)
                    f = self.kde(bins, beta[j,:,k], h)
                    ax[k].plot(bins, f, label=r'$t=' + str(j) + '$', color=c[j], linewidth=1.5)
                else:
                    ax[k].axvline(beta[j,0,k], label=r'$t=' + str(j) + '$', color=c[j], linewidth=1.5)
            
            ax[k].set_ylabel(r'$i='+str(k+1) + '$')
            if k == 0:
                ax[k].legend(fontsize=12)
            ax[k].set_ylim(0,55)
            ax[k].set_xlim(xmin,xmax)
                
        plt.tight_layout()
        plt.savefig(self.drb.name + '_beta.pdf', format='pdf')
        plt.show()        