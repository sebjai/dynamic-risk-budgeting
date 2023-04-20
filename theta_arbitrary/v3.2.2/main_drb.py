# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 17:00:58 2023

@author: sebja
"""

from heston import heston
from Simulator_OU import Simulator_OU
from dynamic_risk_budget import dynamic_risk_budget

import numpy as np
import torch
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

#%%
print("this case is the same as 3.2, but we add a scheduler step")
env = heston(n_assets=5, T=3)

p=1
alpha=0.75

drb = dynamic_risk_budget(env=env,
                          alpha=alpha,
                          p=p, 
                          name="p" + str(int(100*p)) + '_alpha' + str(int(alpha*100))+ "_heston",
                          sched_step_size=500)
drb.eta = 0.1

#%%
drb.train(n_iter=5_000, 
          n_print=500,   
          M_value_iter=5, 
          M_F_iter=5,
          M_policy_iter=1,  
          batch_size=2048)

#%%
for p in [0.6, 0.7, 0.8, 0.9]:
    drb.name = "p" + str(int(100*p)) + '_alpha' + str(int(alpha*100))+ "_heston"
    drb.p = p
    
    drb.train(n_iter=2_000, 
              n_print=500,   
              M_value_iter=5, 
              M_F_iter=5,
              M_policy_iter=1,  
              batch_size=2048)
    
    pl = plotter(drb)

    pl.plot_summary(5_000)
    pl.plot_RC(5_000)
    pl.plot_beta()