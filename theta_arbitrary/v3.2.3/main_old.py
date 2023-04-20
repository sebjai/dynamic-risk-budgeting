# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 17:30:40 2023

@author: jaimunga
"""

from Simulator_OU import Simulator_OU
from dynamic_risk_budget import dynamic_risk_budget

import numpy as np
import torch
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

#%%
env = Simulator_OU(n_assets=5, T=3)

drb = dynamic_risk_budget(Simulator=env, alpha=0.75, p=1.0)
drb.eta = 0.1

#%%
drb.Train(n_iter=10_000, 
          n_print=100,   
          M_value_iter=20, 
          M_F_iter=5,
          M_policy_iter=1,  
          batch_size=512,
          name="n5_T3_p100_mv_20")