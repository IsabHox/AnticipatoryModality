# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 16:10:35 2025
ERP analysis
@author: madln
"""
from preprocessing_function import preprocessing
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from mne.stats import spatio_temporal_cluster_1samp_test


#%% so that mne doesn't print it all
mne.set_log_level('ERROR')

#%% data import
# subjects=['897']
subjects=['pilot3','136','358','916','347','766','661','959','205','400',
          '756','897','420','804','207','196','295','402','722','720',
          '966','568','482','307','730','228','785','640','457','183',
          '759','278','732','441','930','755','689','877','312','751',
          '443','223']
#new participants since: 
# subjects = ['pilot3','136','358','897']#]
nblocks_calib=4
nblocks_test=3
event_id=[1,2]
stim_events=[11,12]
relevant_id=[11,12,1001,1002,1003]
tmin=-1
tmax=2

figpath='./presentations/figures/EEG paper/'

#%% where processing is cached
cache_dir = Path("../../data/expe1/cache_no_reref")
cache_dir.mkdir(exist_ok=True)

#%% structures to be used
noAntis = []
antis = []
anti1s = []
anti2s = []

#%% load an pre-process data
for subject in subjects:
    print(f"Processing subject {subject}")
    
    cache_file = cache_dir / f"{subject}-evokeds.npz"
    
    if cache_file.exists():
        print("Loading cached evoked")
        data = np.load(cache_file)
        noAntis.append(data["noAntis"])
        antis.append(data["antis"])
        anti1s.append(data["anti1s"])
        anti2s.append(data["anti2s"])
        continue
    datapath= '../../data/expe1/{}/{}_calib_000{}'.format(subject,{},{})
    datapath_test= '../../data/expe1/{}/{}_test_000{}'.format(subject,{},{})
    
    # filtering: 0.1-35 for all, 8-13 for alpha band specifically
    epochs, labels, _, _,eegdata=preprocessing(datapath,subject,nblocks_calib,event_id,relevant_id,tmin,tmax,cut_low=None,cut_high=35., reref = False) #previously: 8., no cut_low // previously: cut_high=35.
    eptest, test_labels, _,_, eegtest=preprocessing(datapath_test,subject,nblocks_test,[13],relevant_id,tmin,tmax,cut_low=None,cut_high=35., reref=False)

    
    # events, evt_info=mne.events_from_annotations(eegdata)
    # evt_ix=np.where((events[:,2]==1) | (events[:,2]==2))[0]
    # events[evt_ix,0]=events[evt_ix,0]-900
    # events[evt_ix,2]=15
    # stim_events=np.squeeze(events[evt_ix])
    # stim_events=np.sort(stim_events.view('int,int,int'), order=['f1'], axis=0).view(int)
    # metadata = {'event_time': stim_events[:, 0],
    #             'trial_number': range(len(stim_events))}#
    # metadata = pd.DataFrame(metadata)
    
    # ep_bl_late=mne.Epochs(eegdata, stim_events, event_id=[15], tmin=-1, tmax=2, preload=True,proj=True, baseline=None, detrend=1, metadata=metadata)
    
    # get stimulus labels in cued condition
    # stim_ix=np.where((events[:,2]==11) | (events[:,2]==12))[0]
    # stim_cued=events[stim_ix,2]-10
    #get stimulus labels in test condition
    # events, evt_info=mne.events_from_annotations(eegtest)
    # stim_ix=np.where((events[:,2]==11) | (events[:,2]==12))[0]
    # stim_uncued=events[stim_ix,2]-10
    #create the evoked objects
    anti1_evoked = epochs['1'].average().crop(0,0.9) #.crop()
    anti2_evoked = epochs['2'].average().crop(0,0.9) #.
    
    wcue_evoked = epochs.average().crop(0,0.9) #with cue
    ncue_evoked = eptest.average().crop(0,0.9) #without cue
    
    noAntis.append(ncue_evoked.data)
    antis.append(wcue_evoked.data)
    anti1s.append(anti1_evoked.data)
    anti2s.append(anti2_evoked.data)
    
    np.savez(
        cache_file,
        noAntis=ncue_evoked.data,
        antis=wcue_evoked.data,
        anti1s=anti1_evoked.data,
        anti2s=anti2_evoked.data
    )
    
    # if subjects.index(subject)==0:
    #     noAntis=np.expand_dims(ncue_evoked.get_data(),0)
    #     antis=np.expand_dims(wcue_evoked.get_data(),0)
    #     anti1s=np.expand_dims(anti1_evoked.get_data(),0)
    #     anti2s=np.expand_dims(anti2_evoked.get_data(),0)
    # else:
    #     noAntis=np.concatenate((noAntis, np.expand_dims(ncue_evoked.get_data(),0)), axis=0)
    #     antis=np.concatenate((antis, np.expand_dims(wcue_evoked.get_data(),0)), axis=0)
    #     anti1s=np.concatenate((anti1s, np.expand_dims(anti1_evoked.get_data(),0)), axis=0)
    #     anti2s=np.concatenate((anti2s, np.expand_dims(anti2_evoked.get_data(),0)), axis=0)

# #%% statistical specifications
# precluster_thresh=0.05
# adjacency, names=mne.channels.find_ch_adjacency(epochs.info,'eeg')
# comb_adjacency = mne.stats.combine_adjacency(antis.shape[1],adjacency)

#%% convert to list
noAntis = np.array(noAntis)
antis = np.array(antis)
anti1s = np.array(anti1s)
anti2s = np.array(anti2s)

#%%for cluster tests we need to swap the time and channel axes
noAntis=np.swapaxes(noAntis, 1, 2)
antis=np.swapaxes(antis, 1, 2)
anti1s=np.swapaxes(anti1s, 1, 2)
anti2s=np.swapaxes(anti2s, 1, 2)

#%% TFCE 
# from mne.stats import spatio_temporal_cluster_test
try:
    adjacency,_= mne.channels.find_ch_adjacency(epochs.info,'eeg')
except NameError:
    datapath= '../../data/expe1/{}/{}_calib_000{}'.format(subject,{},{})
    epochs, labels, _, _,eegdata=preprocessing(datapath,subject,nblocks_calib,event_id,relevant_id,tmin,tmax,cut_low=None,cut_high=35., reref = False) #previously: 8., no cut_low // previously: cut_high=35.
finally:
    adjacency,_= mne.channels.find_ch_adjacency(epochs.info,'eeg')


X=antis - noAntis#[noAntis,antis]
Y=anti1s-anti2s

tfce = dict(start=0.1, step=0.1)

t_obs, clusters, cluster_pv, h0 = spatio_temporal_cluster_1samp_test(
    X, tfce, adjacency=adjacency, n_jobs=-1)

#%% plotting
significant_points = cluster_pv.reshape(t_obs.shape).T < 0.025

# We need an evoked object to plot the image to be masked
evoked = mne.combine_evoked(
    [mne.EvokedArray(noAntis.mean(axis=0).T,epochs.info), 
     mne.EvokedArray(antis.mean(axis=0).T,epochs.info)], 
    weights=[1, -1])  # calculate difference wave
time_unit = dict(time_unit="s")
evoked.plot_joint(
    title="Anticipation vs. no anticipation", ts_args=time_unit, topomap_args=time_unit
)  # show difference wave

# Create ROIs by checking channel labels
selections = mne.channels.make_1020_channel_selections(evoked.info, midline="12z")

# Visualize the results
fig, axes = plt.subplots(nrows=3, figsize=(8, 8))
axes = {sel: ax for sel, ax in zip(selections, axes.ravel())}
evoked.plot_image(
    axes=axes,
    group_by=selections,
    colorbar=False,
    show=False,
    mask=significant_points,
    mask_cmap ='vlag',
    mask_alpha=0.9,
    show_names="all",
    titles=None,
    **time_unit,
)
plt.colorbar(axes["Left"].images[-1], ax=list(axes.values()), shrink=0.3, label="µV")

plt.show()
# fig.savefig('./figures/ERP_antiNoAnti.svg')

#%% all the same but between antis 1 and 2
t_obs2, clusters2, cluster_pv2, h02 = spatio_temporal_cluster_1samp_test(
    Y, tfce, adjacency=adjacency, n_jobs=-1)

#%% plotting
significant_points = cluster_pv2.reshape(t_obs2.shape).T < 0.025

# We need an evoked object to plot the image to be masked
evoked = mne.combine_evoked(
    [mne.EvokedArray(anti1s.mean(axis=0).T,epochs.info), 
     mne.EvokedArray(anti2s.mean(axis=0).T,epochs.info)], 
    weights=[1, -1])  # calculate difference wave
diff_evoked = mne.EvokedArray(np.mean(anti1s-anti2s,axis=0).T,epochs.info)
time_unit = dict(time_unit="s")
evoked.plot_joint(
    title="Visual vs. auditory anticipation", ts_args=time_unit, topomap_args=time_unit
)  # show difference wave

# Create ROIs by checking channel labels
selections = mne.channels.make_1020_channel_selections(evoked.info, midline="12z")

# Visualize the results
fig, axes = plt.subplots(nrows=3, figsize=(8, 8))
axes = {sel: ax for sel, ax in zip(selections, axes.ravel())}
fig2=evoked.plot_image(
    axes=axes,
    group_by=selections,
    colorbar=False,
    show=False,
    mask=significant_points,
    show_names="all",
    titles=None,
    mask_cmap ='vlag',
    mask_alpha=0.8,
    **time_unit,
)
plt.colorbar(axes["Left"].images[-1], ax=list(axes.values()), shrink=0.3, label="µV")

plt.show()
# fig.savefig('./figures/ERP_visAud.svg')

#%% ERP saving does not work directly in command line, try here
fig2=evoked.plot_joint(
    title="Visual vs. auditory anticipation", ts_args={'time_unit':'ms'}, topomap_args={'vlim':(-4, 4), 'time_unit':'ms'}
)
# fig2.savefig('./figures/jointplot_visAud.png', dpi=300)

#%% and the other one
evoked = mne.combine_evoked(
    [mne.EvokedArray(noAntis.mean(axis=0).T,epochs.info), 
     mne.EvokedArray(antis.mean(axis=0).T,epochs.info)], 
    weights=[1, -1])  # calculate difference wave
time_unit = dict(time_unit="s")
fig3=evoked.plot_joint(
    title="Cued vs. no uncued", ts_args={'time_unit':'ms'}, topomap_args={'vlim':(-10, 10), 'time_unit':'ms'}
)  # show difference wave
# fig3.savefig('./figures/jointplot_antiNoAnti.png', dpi=300)
#%% ERP analysis is very nice. Now we should look at oscillatory activity
# fig, ax = plt.subplots(figsize=(5, 5), sharey=True, layout="constrained")
# freqs = np.arange(2.0, 35.0, 3.0)
# vmin, vmax = -3.0, 3.0 
# n_cycles = freqs / 2.0

# Xtf = np.swapaxes(X, 1, 2)
# power = epochs.compute_tfr(
#     method="morlet", freqs=freqs, n_cycles=n_cycles, return_itc=False, average=True
# )
# power.plot(
#     [0],
#     # baseline=(0.0, 0.1),
#     mode="mean",
#     # vlim=(vmin, vmax),
#     axes=ax,
#     show=False,
#     colorbar=True,)
# ax.set_title(f"Sim: Using Morlet wavelet")

# %%
