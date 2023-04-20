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
env = Simulator_OU(n_assets=2, T=3)
# env = heston(n_assets=2, T=3)

drb = dynamic_risk_budget(env=env,
                          alpha=0.75,
                          p=0.75)
drb.eta = 0.1

#%%
drb.train(n_iter=20_000, 
          n_print=100,   
          M_value_iter=5, 
          M_F_iter=5,
          M_policy_iter=1,  
          batch_size=1024,
          name="n2_T3_p75_heston")