# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 15:00:58 2022

@author: sebja
"""

from Simulator_OU import Simulator_OU
from DynamicRiskParity import DynamicRiskParity

import numpy as np
import torch
import matplotlib.pyplot as plt

import copy

#%%
Simulator = Simulator_OU(n_assets=2, T=2)

DRP = DynamicRiskParity(Simulator=Simulator, alpha=0.75, p=0.7)
# DRP.B[0,0,0] = 0.5
# DRP.B[0,0,1] = 0.2
# DRP.B[1,0,0] = 0.2
# DRP.B[1,0,1] = 0.1
# for i in range(10):
#     DRP.EstimateValueFunction(N_iter = 50)
DRP.Train(n_iter=1_000, n_print=50, M_value_iter=5, M_policy_iter=5, batch_size=4096)

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


