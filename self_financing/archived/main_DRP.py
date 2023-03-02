# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 15:00:58 2022

@author: sebja
"""

from Simulator_OU import Simulator_OU
from DynamicRiskParity_v10 import DynamicRiskParity

import numpy as np
import torch
import matplotlib.pyplot as plt

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

#%%
Simulator = Simulator_OU(n_assets=2, T=2)

DRP = DynamicRiskParity(Simulator=Simulator, alpha=0.75, p=0.25)
# DRP.B[0,0,0] = 0.5
# DRP.B[0,0,1] = 0.2
# DRP.B[1,0,0] = 0.2
# DRP.B[1,0,1] = 0.1
# for i in range(10):
#     DRP.EstimateValueFunction(N_iter = 50)
#%%
DRP.Train(n_iter=10_000, 
          n_print=100, 
          M_value_iter=5, 
          M_policy_iter=1, 
          batch_size=4096,
          name="n2_T5_p25")

#%% various alpha levels
alpha = [0.7, 0.8, 0.9, 0.95]
DRP = []
for i, a in enumerate(alpha):
    
    print(a)

    # if i ==0 :
    #     DRP.append(copy.copy(DRP0))
    # else:
    #     DRP.append(copy.copy(DRP[-1]))

    # DRP.append(copy.copy(DRP0))
    DRP.append(DynamicRiskParity(Simulator=Simulator, alpha=a))
    DRP[-1].Train(n_iter=1000, n_print=10, batch_size=512)

#%%

DRP0 = DynamicRiskParity(Simulator=Simulator, alpha=0.7)
# DRP0.Train(n_iter=50)

# DRP0.__initialize_CVaR_VaR_Net__()
# DRP0.VaR_CVaR_loss = []
# DRP0.EstimateValueFunction(N_iter=5000, batch_size=128)



#%% naive CVaR at time 0

DRP0.EstimateValueFunction(N_iter=500, batch_size=512)


batch_size = 2048
costs, h, Y, beta, theta, S, wealth = DRP0.__RunEpoch__(batch_size=batch_size)

costs = costs.detach().numpy().squeeze()

qtl = np.quantile(costs, DRP0.alpha)

plt.hist(costs, density=True, bins=51, alpha=0.5)
plt.axvline(qtl, color='r', linestyle='--')
plt.show()


V = torch.zeros((DRP0.T+1, batch_size))
for j in range(DRP0.T):
    V[j,:] = DRP0.CVaR(t = j*DRP0.dt, h = h[j+1,...].transpose(0,1), Y=Y[j,...]).squeeze()

print(qtl, np.mean(costs[costs >= qtl]))
print(V[0,0])



#%%


for model in DRP:
    np.random.set_state(state)
    model.PlotPaths(4096, title='alpha_' + str(int(model.alpha*100)) )

#%%
state = np.random.get_state()
for model in DRP:
    np.random.set_state(state)
    model.PlotSummary()


#%%
# DRP0 = DynamicRiskParity(Simulator=Simulator, alpha=0.5)


# # #%% test running an epoch
# B = np.expand_dims(np.array([[0.1,0.5],[0.9,0.5]]), axis=1)
# DRP0.B = torch.tensor(B).float()

# DRP_B = []
# for a in alpha:
#     print(a)
#     DRP_B.append(copy.deepcopy(DRP0))
#     DRP_B[-1].alpha = a
#     DRP_B[-1].Train(n_iter=1_000, n_print=100, mini_batch_size=512)





#%%
def kde(x, bins):
    h = np.std(x)*1.06*(len(x))**(-1/5)
    
    h += 1e-6
    dx = x.reshape(1,-1)-bins.reshape(-1,1)
    
    f = np.sum(np.exp(-0.5*dx**2/h**2)/np.sqrt(2*np.pi*h**2), axis=1)/len(x)
    
    return f
        
def PlotCVaR(x, kde_bins, bins, alpha, trigger_level=1, trigger_period=2):
    
    plt.hist(x, bins=bins, density=True, alpha=0.5)
    qtl = np.quantile(x, alpha)
    
    plt.axvline(qtl,color='k',linestyle='--',linewidth=2, label=r"$VaR_{0.75}$")
    plt.axvline(np.mean(x[x>=qtl]),color='r',linestyle='--',linewidth=2, label=r"$CVaR_{0.75}$")
    
    plt.axvline(np.mean(x),color='b',linestyle='--',linewidth=2, label=r"$\mathbb{E}$")
    
    f = kde(x, kde_bins)
    plt.plot(kde_bins,f,linewidth=1,color='r')
    plt.fill_between(kde_bins[kde_bins>=qtl], f[kde_bins>=qtl], alpha=0.5, color='r')
    
    plt.legend(fontsize=14)
    plt.ylim(0,5)
    plt.xlim(0.7,1.4)
    plt.show()

costs, h, Y, beta, theta, S, wealth = DRP.__RunEpoch__(10_000)

wealth = wealth.detach().cpu().numpy()

bins=np.linspace(0.6, 1.4, 101)
kde_bins = np.linspace(0.6, 1.4, 501)

PnL = wealth[-1,:]/wealth[0,0]

PlotCVaR(PnL, kde_bins, bins, 0.75)

#%%
def PlotCVaR_trigger(x, kde_bins, bins, alpha, trigger_level=1, trigger_period=2):
    
    mask = (x[trigger_period,:]<=trigger_level)
    
    x= x[-1,mask]
    
    plt.hist(x, bins=bins, density=True, alpha=0.5)
    qtl = np.quantile(x, 1-alpha)
    
    plt.axvline(qtl,color='k',linestyle='--',linewidth=2, label=r"$VaR_{0.75}$")
    plt.axvline(np.mean(x[x<=qtl]),color='r',linestyle='--',linewidth=2, label=r"$CVaR_{0.75}$")
    
    plt.axvline(np.mean(x),color='b',linestyle='--',linewidth=2, label=r"$\mathbb{E}$")
    
    x = x[:50_000]
    
    f = kde(x, kde_bins)
    plt.plot(kde_bins,f,linewidth=1,color='r')
    plt.fill_between(kde_bins[kde_bins<=qtl], f[kde_bins<=qtl], alpha=0.5, color='r')
    
    plt.legend(fontsize=14)
    plt.ylim(0,7)
    plt.xlim(0.7,1.4)
    plt.show()
#%%
bins=np.linspace(0.6, 1.4, 101)
kde_bins = np.linspace(0.6, 1.4, 501)

_, _, _, _, _, _, wealth = DRP.__RunEpoch__(1_000_000)

wealth = wealth.detach().cpu().numpy()

_, _, _, wealth_const_beta = DRP.RunEpoch_const_beta(batch_size=1_000_000, const_beta=torch.tensor([0.45, 0.55]).to(DRP.device))
wealth_const_beta = wealth_const_beta.detach().cpu().numpy()

#%%
PlotCVaR_trigger(wealth/wealth[0,0], kde_bins, bins, 0.75,  trigger_level=0.9, trigger_period=2)
PlotCVaR_trigger(wealth_const_beta/wealth_const_beta[0,0], kde_bins, bins, 0.75,  trigger_level=0.9, trigger_period=2)