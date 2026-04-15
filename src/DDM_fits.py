# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 17:23:38 2023
DDM fitting of our data
@author: ihoxha
"""

# from ddm import Model, Fittable, InitialCondition, Overlay
# from ddm.solution import Solution
# from ddm.sample import Sample
# from ddm.models import NoiseConstant, BoundConstant, OverlayNonDecision, LossRobustLikelihood, Drift, ICPoint 
# from ddm.functions import fit_adjust_model, get_model_loss
# import ddm.plot
import pyddm
from pyddm.models.ic import ICPointRatio
from pyddm.models.overlay import OverlayNonDecision

import pandas as pd
import numpy as np

from joblib import Parallel, delayed

import itertools

#%% import data first
full_data=pd.read_csv('./cued_classification_and_behavior_4-8Hz.csv')
full_data['correct']=(full_data['Stimulus']==full_data['Response'])+0
# full_data['anticipation'] = (full_data['Stimulus']==full_data['Estimated class'])+0
full_data['anticipation'] = (full_data['Stimulus']==full_data['Cue'])+0
full_data['RT']=full_data['RT']/1000

#%% sepcify conditioning functions on drift an starting point
def drift_func(mu1, mu2, anticipation):
    m = mu1 * anticipation + mu2 * (1-anticipation)
    return m

def sp(x1,x2, anticipation):
    x = x1 * anticipation + x2 * (1-anticipation)
    return x

#%%specify the DDM that will be used: one drift per stimulus type
model = pyddm.gddm(drift = drift_func,
                    starting_position = sp,
                    nondecision = 0.3,
                    conditions=['anticipation'],
                    parameters={"mu1":(-5,5),"mu2":(-5,5),
                                "x1":(0,1),"x2":(0,1)},
                    T_dur = 2.001
                    )

  
#%% see now what happens when we fit just one DDM per condition
subjects = np.unique(full_data.Subject)
results_df = pd.DataFrame(columns=model.get_model_parameter_names(),
                          index = subjects)
for s in subjects:
    data = full_data[full_data.Subject==s]
    samples = pyddm.Sample.from_pandas_dataframe(data, choice_column_name='correct',
                                                 rt_column_name='RT')
    model.fit(sample = samples, verbose=False)
    results_df.loc[s] = [x+0 for x in model.get_model_parameters()]

#%% save
results_df.to_csv('./DDM_fits_cued_antiCorrectness.csv')
# %% try a more complex model, also accounting for stimulus differences
def drift_4_func(mu_vis, mu_aud, mu1, mu2, Stimulus, anticipation):
    stim = Stimulus-1
    m = (mu1 * anticipation + mu2 * (1-anticipation)) * (mu_aud*stim + mu_vis*(1-stim))
    return m

def sp_4(x_vis,x_aud,x1,x2,Stimulus, anticipation):
    stim = Stimulus-1
    x = (x1 * anticipation + x2 * (1-anticipation)) * (x_aud*stim + x_vis*(1-stim))
    return x

def nondectime(T1,T2,Stimulus):
    stim = Stimulus-1
    T = T2*stim + T1*(1-stim)
    return T

model2 = pyddm.gddm(drift = drift_4_func,
                    starting_position = sp_4,
                    nondecision=nondectime,
                    conditions=['Stimulus','anticipation'],
                    parameters={"mu1":(0,1),"mu2":(0,1),
                                "x1":(0,1),"x2":(0,1),
                                "mu_aud":(-5,5),"mu_vis":(-5,5),
                                "x_aud":(0,1),"x_vis":(0,1),
                                'T1':(0.1, 0.5),'T2':(0.1, 0.5)},
                    T_dur=2.001
                    )

results_df2 = pd.DataFrame(columns=model2.get_model_parameter_names(),
                          index = subjects)
for s in subjects:
    data = full_data[full_data.Subject==s]
    samples = pyddm.Sample.from_pandas_dataframe(data, choice_column_name='correct',
                                                 rt_column_name='RT')
    model2.fit(sample = samples, verbose=False)
    results_df2.loc[s] = [x+0 for x in model2.get_model_parameters()]
    
# save
results_df2.to_csv('./DDM_fits_cued_antiCorrectness_Stimulus.csv')

# %%
