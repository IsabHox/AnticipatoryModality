# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 15:41:34 2025
preprocessing file
@author: madln
"""
import mne 
import numpy as np
import pandas as pd
from collections import Counter

def preprocessing(datapath,subject,nblock,event_id,relevant_id,tmin,tmax,
                  balanced=False,cut_low=None,cut_high=35.,
                  baseline = None, reref = True):
    '''This is the same as preprocessing in the steps, but the file name format changes '''
    #%% load and concatenate data 
    raws = []
    for b in range(nblock):
        filename = datapath.format(subject, b + 1)
        intraw = mne.io.read_raw_brainvision(filename + '.vhdr', preload=True)
        raws.append(intraw)
    raw = mne.concatenate_raws(raws)

    #%% re-reference to common average
    # raw.set_eeg_reference('average', projection=False) #TODO: check here if we get the channel extra
    
    #%% detect bad channels and interpolation
    # raw.info['bads'] = []  # reset
    # raw.plot()
    # # raw.pick_types(eeg=True)  # drop non-EEG channels
    
    # #automatic bad channel detection (based on high variance)
    # # raw_copy = raw.copy().filter(1, 30)  # broad filter to make detection easier
    # auto_bad = mne.preprocessing.find_bad_channels_maxwell(raw, cross_talk=None, calibration=None)
    # raw.info['bads'].extend(auto_bad['bads'])
    # print(f"There were {len(auto_bad['bads'])} bad channels. Interpolated")
    # raw.interpolate_bads(reset_bads=True)
    raw = mne.add_reference_channels(raw,ref_channels=['Fz'])
    raw.set_montage('standard_1020')
    #%% causal band_pass filtering
    raw.filter(cut_low, cut_high, fir_design='firwin', phase='minimum') #beware that it can introduce distortions
    
    #%% event detection
    events, _ = mne.events_from_annotations(raw)
    event_ix = np.isin(events[:, 2], event_id)
    stim_events = events[event_ix]

    subevent_ix = np.isin(events[:, 2], relevant_id)
    subevents = events[subevent_ix]
    

    #%% Epoching
    metadata = pd.DataFrame({
        'event_time': stim_events[:, 0],
        'trial_number': np.arange(len(stim_events)),
        'label': stim_events[:, 2]
    })
    epochs = mne.Epochs(raw, stim_events, event_id, tmin, tmax,
                         proj=True, baseline=baseline, metadata=metadata,
                         detrend=1, preload=True, event_repeated='merge')

    #%% electrode rejection and common average re-referencing
    rej_trials,rej_channels=get_bad_channels_trials(raw,stim_events)
    # rej_channels = mne.preprocessing.find_bad_channels_lof(raw)
    # print(f'ACTUALLY REJECTED: {len(rej_channels)}')
    epochs.info['bads'] = rej_channels
    epochs.info['bads'].append('Fz')
    epochs.interpolate_bads()
    # epochs.set_eeg_reference('average')
    if reref:
        epochs.set_eeg_reference(['TP10'])
    # rej_trials,rej_channels=get_bad_channels_trials(raw,stim_events)
    epochs.drop(rej_trials)
    # reject_criteria =  # 150 µV threshold
    # 

    #%% extract labels
    evt_sorted=np.sort(epochs.events.view('int,int,int'), order=['f1'], axis=0).view(int)
    labels = evt_sorted[:, -1]
    subevt_sorted=np.sort(subevents.view('int,int,int'), order=['f1'], axis=0).view(int)
    timings=np.diff(subevt_sorted[:,0])
    #timings=timings[timings<3000] #this is a bit of tinkering...find a more elegant way of doing
    timings=timings[timings>0]
    

    #%% if we want balanced condition, filter out here
    if balanced:
        red_epochs=epochs.copy()
        [red_epochs,rej_ix]=red_epochs.equalize_event_counts(['10','20'])
        rej_ix=rej_ix.tolist()
        return epochs,labels,timings, red_epochs, rej_ix, raw
    print(f'There are {len(epochs)} epochs')
    
    return epochs, labels, timings, rej_trials, raw

def get_bad_channels_trials(eegdata,stim_events,thresh_trial=200e-6,thresh_chans=0.2,tmin=0,tmax=2,reject_tmin=0,reject_tmax=2):
    '''Returns bad channels and trials given the specified thresholds.
    Inputs:
        thresh_trial: (default 150e-6) threshold value from which an epoch should be rejected (in V)
        thresh_chans: (default 0.15) proportion of trials rejected to decide to reject channels. Should be between 0 and 1
    Outputs:
        rej_trials: list of trial indices that should be rejected
        rej_channels: list of str of the rejected channels'''
    rej_dict=dict(eeg=thresh_trial)
    ep = mne.Epochs(eegdata, events=stim_events, baseline=None,tmin=tmin, tmax=tmax,reject=rej_dict,reject_tmin=reject_tmin,reject_tmax=reject_tmax, preload=True)
    drop_log=list(ep.drop_log)
    stats=len(get_rej_trials(drop_log))/len(stim_events) #ep.drop_log_stats()
    stat_details=get_bad_stats(drop_log)
    rej_trials=get_rej_trials(drop_log)
    rej_channels=[]
    ch=0
    while stats>thresh_chans and ch<eegdata.info['nchan']:
        chan_to_rej=stat_details.most_common()[0][0]
        rej_channels.extend([chan_to_rej])
        for i in range(len(drop_log)):
            if chan_to_rej in drop_log[i]:
                new_log=list(drop_log[i])
                new_log.remove(chan_to_rej)
                if new_log is None:
                    drop_log[i]=()
                else:
                    drop_log[i]=tuple(new_log)
        stats=len(get_rej_trials(drop_log))/len(stim_events)
        stat_details=get_bad_stats(drop_log)
        ch+=1
    rej_trials=get_rej_trials(drop_log)
    # print(f'Dropping {len(rej_trials)} trials')
    print(f'Dropping {len(rej_channels)} channels')
    return rej_trials,rej_channels

def get_rej_trials(drop_log):
    '''From a drop log, get the index of trials that have been rejected'''
    ix_list=[]
    for i,t in enumerate(drop_log):
        if len(t)!=0:
            ix_list.append(i)
    return ix_list

def get_bad_stats(drop_log):
    scores = Counter([ch for d in drop_log for ch in d])
    return scores

def get_stim(eegdata,rej_ix=[]):
    events, event_dict= mne.events_from_annotations(eegdata)
    stim_ix=np.where((events[:,2]==11) | (events[:,2]==12))
    stim_events=events[stim_ix,2]-11
    if len(rej_ix)!=0:
        stim_events=np.delete(stim_events,rej_ix)
    return stim_events

def get_rt(eegdata, rej_ix=[]):
    events, event_dict= mne.events_from_annotations(eegdata)
    subevt_ix=np.where((events[:,2]==11) | (events[:,2]==12)| (events[:,2]==1001)| (events[:,2]==1002)| (events[:,2]==1003))[0]
    stim_ix=np.where((events[:,2]==11) | (events[:,2]==12))[0]
    if len(rej_ix)!=0:
        stim_ix=np.delete(stim_ix,rej_ix)
    
    _,which_timings,_=np.intersect1d(subevt_ix, stim_ix, return_indices=True)
    stim_and_resp_events=events[subevt_ix,0]
    rts=np.diff(stim_and_resp_events)
    rts=rts[which_timings]
    return rts

def get_response(eegdata,rej_ix=[]):
    events, event_dict= mne.events_from_annotations(eegdata)
    subevt_ix=np.where((events[:,2]==1001)| (events[:,2]==1002)| (events[:,2]==1003))[0]
    if len(rej_ix)!=0:
        subevt_ix=np.delete(subevt_ix,rej_ix)
    resp=events[subevt_ix,2]-1000
    return resp