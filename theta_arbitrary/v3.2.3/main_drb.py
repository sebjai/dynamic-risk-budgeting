# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 17:00:58 2023

@author: sebja
"""

from heston import heston
from Simulator_OU import Simulator_OU
from dynamic_risk_budget import dynamic_risk_budget
from plotter import plotter

import numpy as np
import torch
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

#%%
print("this case is the same as 3.2.2, but we now elicit rho directly")
env = heston(n_assets=5, T=3)

p=0.5
alpha=0.75

prefix = 'unequalB_'

drb = dynamic_risk_budget(env=env,
                          alpha=alpha,
                          p=p, 
                          name=prefix + "p" + str(int(100*p)) + '_alpha' + str(int(alpha*100))+ "_heston",
                          sched_step_size=250)

b = np.linspace(1,env.n, env.n)
b /= np.sum(b)
B = np.zeros((env.T,env.n))
for i in range(env.T):
    B[i,:] = b

drb.B = torch.tensor(B).to(drb.device).unsqueeze(axis=1)

drb.eta = 0.1

#%%
drb.train(n_iter=5_000, 
          n_print=500,   
          M_value_iter=5, 
          M_F_iter=5,
          M_policy_iter=1,  
          batch_size=2048)

pl = plotter(drb)

pl.plot_summary(len(drb.V))
pl.plot_RC(len(drb.V))
pl.plot_beta()

#%%
for p in [0.6, 0.7, 0.8, 0.9]:
    
    drb.name = prefix + "p" + str(int(100*p)) + '_alpha' + str(int(alpha*100))+ "_heston"
    drb.p = p
    
    drb.__reset_optim_sched__()
    
    drb.train(n_iter=1_000, 
              n_print=500,   
              M_value_iter=5, 
              M_F_iter=5,
              M_policy_iter=1,  
              batch_size=2048)
    
    pl = plotter(drb)

    pl.plot_summary(1_000)
    pl.plot_RC(1_000)
    pl.plot_beta()