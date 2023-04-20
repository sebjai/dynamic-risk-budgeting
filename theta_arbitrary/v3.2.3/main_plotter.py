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

#%%
filenames= ['p60_alpha75_heston_04-13-2023_12-08-01.pkl',
            'p70_alpha75_heston_04-13-2023_12-58-22.pkl',
            'p80_alpha75_heston_04-13-2023_13-48-40.pkl',
            'p90_alpha75_heston_04-13-2023_14-38-42.pkl']


filenames = ['p100_alpha95_heston_04-13-2023_21-30-00.pkl']
#%%
for file in filenames:
    drb = dill.load(open(file,'rb'))
    pl = plotter(drb)
    # pl.plot_summary(1_000)
    # pl.plot_RC(1_000)
    # pl.plot_beta()
    pl.plot_wealth()
    
#%%
for file in filenames:
    drb = dill.load(open(file,'rb'))
    pl = plotter(drb)
    pl.plot_wealth()
    
#%%
for file in filenames:
    drb = dill.load(open(file,'rb'))
    pl = plotter(drb)
    pl.plot_beta()
    
#%%
# pl = plotter(dill.load(open('p75_alpha90_heston_04-11-2023_17-34-22.pkl','rb')))
pl = plotter(dill.load(open('p75_alpha90_heston_04-11-2023_16-10-08.pkl','rb')))
#%%
pl.plot_summary(4_000)
pl.plot_RC(4_000)
pl.plot_beta()