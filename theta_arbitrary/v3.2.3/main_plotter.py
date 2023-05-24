# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 10:11:33 2023

@author: jaimunga
"""

from plotter import plotter
from heston import heston
from Simulator_OU import Simulator_OU
from dynamic_risk_budget import dynamic_risk_budget

import dill
import numpy as np
import torch
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


import matplotlib.pyplot as plt
plt.style.use('paper.mplstyle')

# pl = plotter(drb)

# pl.plot_summary(5_000)
# pl.plot_RC(5_000)
# pl.plot_beta()

#%%
filenames =['p50_alpha75_heston_05-01-2023_19-25-25.pkl',
            'p60_alpha75_heston_05-01-2023_20-09-03.pkl',
            'p70_alpha75_heston_05-01-2023_20-53-15.pkl',
            'p80_alpha75_heston_05-01-2023_21-37-05.pkl',
            'p90_alpha75_heston_05-01-2023_22-21-09.pkl',
            'unequalB_p50_alpha75_heston_05-01-2023_19-26-40.pkl',
            'unequalB_p60_alpha75_heston_05-01-2023_20-10-17.pkl',
            'unequalB_p70_alpha75_heston_05-01-2023_20-54-30.pkl',
            'unequalB_p80_alpha75_heston_05-01-2023_21-38-21.pkl',
            'unequalB_p90_alpha75_heston_05-01-2023_22-22-06.pkl']

# filenames= ['unequalB_p50_alpha75_heston_04-29-2023_16-29-15.pkl']

#%%
for i, file in enumerate(filenames):
    drb = dill.load(open(file,'rb'))
    pl = plotter(drb)
    
    # if i ==0 :
    #     n = len(drb.V)
    # else:
    #     n = 1000
    # pl.plot_summary(n)
    # pl.plot_RC(n)
    
    pl.plot_beta(35)
    

