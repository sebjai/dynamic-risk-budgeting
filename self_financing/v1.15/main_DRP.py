# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 15:00:58 2022

@author: sebja
"""

from Simulator_OU import Simulator_OU
from DynamicRiskParity import DynamicRiskParity

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

#%%
Simulator = Simulator_OU(n_assets=2, T=4)

DRP = DynamicRiskParity(Simulator=Simulator, alpha=0.75, p=0.25)
DRP.eta = 0.02
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
          batch_size=128,
          name="n2_T4_p25")

