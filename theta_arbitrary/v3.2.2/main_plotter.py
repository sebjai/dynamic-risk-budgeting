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

pl = plotter(drb)

pl.plot_summary(5_000)
pl.plot_RC(5_000)
pl.plot_beta()

#%%
filenames= ['p50_heston_04-07-2023_16-26-51.pkl',
            'p75_heston_04-07-2023_16-26-39.pkl',
            'p100_heston_04-07-2023_16-26-38.pkl']

#%%
for file in filenames:
    drb = dill.load(open(file,'rb'))
    pl = plotter(drb)
    pl.plot_summary()
    pl.plot_RC()
    
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