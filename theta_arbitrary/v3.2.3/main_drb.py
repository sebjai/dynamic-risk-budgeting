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
from plotter import plotter
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import dill

#%%
print("this case is the same as 3.2.2, but we use transfer learning")

drb = dill.load(open('p50_alpha75_heston_04-12-2023_16-15-05.pkl', 'rb'))

#%%
for alpha in [0.8, 0.85, 0.9, 0.95]:
    drb.name = "p" + str(int(100*drb.p)) + '_alpha' + str(int(alpha*100))+ "_heston"
    drb.alpha = alpha
    
    drb.__restart_sched__(100)
    
    drb.train(n_iter=1_000, 
              n_print=100,   
              M_value_iter=5, 
              M_F_iter=5,
              M_policy_iter=1,  
              batch_size=2048)
    
    pl = plotter(drb)

    pl.plot_summary(1_000)
    pl.plot_RC(1_000)
    pl.plot_beta()